"""Run paired scripted pilots. Set MUJOCO_GL=egl when exporting PNGs headlessly."""

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path

from gymnasium.utils.env_checker import check_env
import numpy as np
from PIL import Image

from sim.carry_place import CarryPlaceEnv, Scenario


TRACE_HEADER = ("time_s", "x_m", "y_m", "z_m", "ref_x_m", "ref_y_m", "ref_z_m", "target_fz_n",
                "filtered_fz_n", "disturbance_x_n", "disturbance_y_n", "kx_y_n_m", "kz_n_m", "touched", "released")


def rollout(env, scenario, image_dir=None):
    env.reset(seed=27, options={"scenario": scenario})
    trace = []
    if image_dir is not None:
        Image.fromarray(env.render()).save(image_dir / "grasp.png")
    saved_pulse = saved_touch = False
    while True:
        _, _, terminated, truncated, info = env.step(np.zeros(2))
        trace.extend(env.last_trace)
        if image_dir is not None:
            if not saved_pulse and np.linalg.norm(env.data.xfrc_applied[env.payload, :2]) > 0.0:
                Image.fromarray(env.render()).save(image_dir / "disturbance.png")
                saved_pulse = True
            if not saved_touch and info["first_touch_s"] is not None:
                Image.fromarray(env.render()).save(image_dir / "contact.png")
                saved_touch = True
        if terminated or truncated:
            break
    if image_dir is not None:
        if not (saved_pulse and saved_touch and info["is_success"]):
            raise AssertionError(f"Could not capture all stages of the pilot: {info}")
        Image.fromarray(env.render()).save(image_dir / "settled.png")
    return info, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/smoke"))
    parser.add_argument("--render", action="store_true", help="Export four actual MuJoCo frames")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = Scenario()
    scenarios = {"no_pulse": replace(base, pulse=0.0), "standard_pulse": base,
                 "raised_table": replace(base, height=0.02, pulse=12.0),
                 "lowered_table": replace(base, height=-0.02, pulse=12.0)}
    records = []
    for baseline in ("fixed", "phase"):
        env = CarryPlaceEnv(baseline=baseline, render_mode="rgb_array" if args.render else None)
        try:
            if baseline == "fixed":
                check_env(env, skip_render_check=True)
            for name, scenario in scenarios.items():
                capture = args.render and baseline == "phase" and name == "standard_pulse"
                info, trace = rollout(env, scenario, args.output if capture else None)
                records.append(dict(method=baseline, case=name, scenario=asdict(scenario), **info))
                if name == "standard_pulse":
                    with (args.output / f"{baseline}_trace.csv").open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.writer(handle)
                        writer.writerow(TRACE_HEADER)
                        writer.writerows(trace)
                print(f"{baseline:5s} {name:14s} {info['status']:16s} touch={info['first_touch_s']} "
                      f"Fpeak={info['peak_force_n']} N", flush=True)
        finally:
            env.close()
    negative = CarryPlaceEnv()
    try:
        failure, _ = rollout(negative, replace(base, pulse=100.0, angle_deg=0.0, duration=0.12))
    finally:
        negative.close()
    summary = dict(episodes=records, injected_drop_check=failure, physics_dt_s=0.002,
                   action_dt_s=0.02, gymnasium_check="passed")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not all(row["is_success"] and row["contact_window_complete"] for row in records):
        raise SystemExit("At least one paired pilot failed; see summary.json")
    if failure["status"] != "dropped" or failure["first_touch_s"] is not None:
        raise SystemExit(f"Drop check did not produce the expected failure: {failure}")


if __name__ == "__main__":
    main()
