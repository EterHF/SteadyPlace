"""Paired pilot of timed and current-state release on the moving tray."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json
from math import ceil
from pathlib import Path

import mujoco as mj
import numpy as np

from sim.stack import BLOCK_COUNT, BLOCK_WIDTH, HOLD, OPEN, TABLE_Z, StackRun, cycle_is_stable


PERIODS = (1.30, 1.34, 1.38, 1.42, 1.50, 2.00)
PHASES_DEG = (0, 90, 180, 270)
MIN_SUPPORT_MARGIN_M = 0.004
GATE_INTERVAL_S = 0.020
SCREEN_YAW_DEG = 0.3


class RuleStackRun(StackRun):
    """Align to the supporting block and release at positive current support margin."""

    def __init__(self, period, *, tray_phase, align_landing=True):
        super().__init__(period, tray_phase=tray_phase)
        self.align_landing = align_landing
        self.release_checks = []

    def _landing_target(self, layer, landing):
        if not self.align_landing:
            return landing
        # Simulation pose is a noiseless proxy for a visual estimate at the standoff.
        support = self.tray if layer == 0 else self.blocks[layer - 1]
        offset = self.data.xpos[support, :2] - self.data.xpos[self.blocks[layer], :2]
        corrected = landing.copy()
        corrected[:2] += np.clip(offset, -BLOCK_WIDTH / 4, BLOCK_WIDTH / 4)
        return corrected

    def _support_margin(self, layer):
        # Project the mass above each interface along effective gravity. Footprints
        # are approximated by axis-aligned boxes in this instantaneous rule.
        bodies = self.blocks[:layer + 1]
        centers = self.data.xpos[bodies]
        masses = self.model.body_mass[bodies]
        displacement, _ = self.tray_reference()
        accel = -self.omega**2 * displacement
        half = BLOCK_WIDTH / 2
        margin = np.inf
        for i, block in enumerate(centers):
            if i == 0:
                support_xy = self.data.xpos[self.tray, :2]
                support_half = self.model.geom_size[self.supports[0], :2]
            else:
                support_xy = centers[i - 1, :2]
                support_half = half
            low = np.maximum(block[:2] - half, support_xy - support_half)
            high = np.minimum(block[:2] + half, support_xy + support_half)
            com = np.average(centers[i:], axis=0, weights=masses[i:])
            supported_height = com[2] - (block[2] - half)
            projection = com[:2].copy()
            projection[1] -= supported_height * accel / 9.81
            margin = min(margin, float(np.min(np.minimum(projection - low, high - projection))))
        return margin

    def _release_ready(self, layer, landing, payload_kg, measure):
        start = self.data.time
        checks = ceil(self.period / GATE_INTERVAL_S)
        for check in range(checks + 1):
            margin = self._support_margin(layer)
            ready = margin >= MIN_SUPPORT_MARGIN_M
            if ready or check == checks:
                self.release_checks.append({
                    "layer": layer + 1, "ready": ready,
                    "wait_s": self.data.time - start, "margin_mm": 1000 * margin,
                })
                return ready
            duration = min(GATE_INTERVAL_S, self.period - check * GATE_INTERVAL_S)
            self.move(f"{layer + 1}/{BLOCK_COUNT}  Wait for support margin", landing, duration,
                      follow=True, grip=HOLD, payload_kg=payload_kg,
                      vertical_stiffness=620.0, measure=measure)

    def run(self):
        result = super().run()
        result["control"] = ("pose_alignment_and_current_support_margin"
                             if self.align_landing else "current_support_margin")
        result["release_checks"] = self.release_checks
        return result


def aligned_tower_check(period, phase):
    """Check a settled, centered tower starting at the timed controller's retreat phase."""
    run = StackRun(period, tray_phase=phase)
    model, data = run.model, run.data
    tray_speed = run.tray_velocity
    tray_shift = float(data.qpos[run.slide_q])
    qjoints = model.body_jntadr[run.blocks]
    velocities = model.jnt_dofadr[qjoints]
    for layer, q in enumerate(model.jnt_qposadr[qjoints]):
        # Break exact face/corner coincidence without an appreciable change in support area.
        half_yaw = np.radians((-1)**layer * SCREEN_YAW_DEG) / 2
        data.qpos[q:q+7] = [
            *data.xpos[run.tray, :2], TABLE_Z + (layer + 0.5)*BLOCK_WIDTH,
            np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw),
        ]
    data.qvel[run.slide_v] = 0.0
    data.ctrl[run.gripper] = OPEN
    mj.mj_forward(model, data)
    data.ctrl[:7] = data.qfrc_bias[:7]
    for _ in range(round(0.60 / run.dt)):
        data.ctrl[run.tray_drive] = 1100 * (tray_shift - data.qpos[run.slide_q]) - 62 * run.tray_velocity
        mj.mj_step(model, data)
    data.qvel[run.slide_v] = tray_speed
    for v in velocities:
        data.qvel[v:v+6] = [0.0, tray_speed, 0.0, 0.0, 0.0, 0.0]
    data.time = 0.0
    data.qacc_warmstart[:] = 0.0
    mj.mj_forward(model, data)
    initial_xy = data.xpos[run.blocks, :2] - data.xpos[run.tray, :2]
    initial_offset_mm = 1000 * float(np.max(np.linalg.norm(initial_xy, axis=1)))
    initial_tilt = float(np.degrees(np.arccos(np.clip(data.xmat[run.blocks, 8], -1, 1))).max())
    cycles = [run.verify_cycle() for _ in range(2)]
    return {
        "initial_max_offset_mm": initial_offset_mm,
        "initial_max_tilt_deg": initial_tilt,
        "stable": all(cycle_is_stable(cycle) for cycle in cycles),
        "cycles": cycles,
    }


def evaluate_case(case):
    period, phase_deg = case
    phase = np.radians(phase_deg)
    fixed = StackRun(period, tray_phase=phase).run()
    margin_only = RuleStackRun(period, tray_phase=phase, align_landing=False).run()
    rule = RuleStackRun(period, tray_phase=phase).run()
    final_phase = phase + 2 * np.pi * fixed["task_finished_s"] / period
    ideal = aligned_tower_check(period, final_phase)
    return {
        "period_s": period, "start_phase_deg": phase_deg,
        "ideal_at_timed_retreat": ideal, "timed": fixed, "margin_only": margin_only, "rule": rule,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/stack_baselines"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = list(product(PERIODS, PHASES_DEG))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(evaluate_case, cases))
    result = {
        "screen": {"horizon_periods": 2, "phase": "timed_controller_final_retract",
                   "alternating_yaw_deg": SCREEN_YAW_DEG},
        "rule": {"min_support_margin_mm": 1000 * MIN_SUPPORT_MARGIN_M,
                 "gate_check_interval_s": GATE_INTERVAL_S,
                 "max_wait_periods_per_layer": 1, "pose": "noiseless_simulator_pose"},
        "cases": rows,
    }
    report = args.output / "summary.json"
    temp = report.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(report)
    for row in rows:
        print(f"{row['period_s']:.2f} s, phase {row['start_phase_deg']:3d} deg  "
              f"ideal={row['ideal_at_timed_retreat']['stable']}  "
              f"timed={row['timed']['success']}  margin={row['margin_only']['success']}  "
              f"aligned_rule={row['rule']['success']}")


if __name__ == "__main__":
    main()
