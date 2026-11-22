"""Paired action interventions with model uncertainty and cross-stage loading."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from sim.carry_place import CARRY_TIME, DT, Scenario
from sim.coupled_place import (DEADLINE_S, MAX_BRAKE, MAX_SPEED, MIN_BRAKE, MIN_SPEED, PEAK_LIMIT_N,
                              PROFILES, REFERENCE_TRAVEL_M, CoupledCarryPlaceEnv, disturbance_program)
from sim.evaluate_disturbance import METHODS


SEED = 6314
BRAKE_ONSET_M = .085
PACES = {"slow": .016, "middle": .022, "fast": .050}
STRONG = "fixed_520_300"


def action_for(speed, brake=BRAKE_ONSET_M, grip=0.):
    return np.array([0., 0., 2 * (speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED) - 1,
                     2 * (brake - MIN_BRAKE) / (MAX_BRAKE - MIN_BRAKE) - 1, grip])


def cases_for(context):
    i = context["id"]
    params = {k: v for k, v in context.items() if k != "id"}
    return {profile: Scenario(**params, **disturbance_program(profile, np.random.default_rng(SEED + i * 3 + j)))
            for j, profile in enumerate(PROFILES)}


def replay(env, case, action, on_frame=None, *, seed=SEED):
    obs, _ = env.reset(seed=seed, options={"scenario": case})
    assert env.observer.mass_kg == .25 and obs.shape == (42,)
    plant, estimate = [], []
    if on_frame is not None:
        on_frame(env)
    while True:
        command = action(obs) if callable(action) else action
        obs, _, ended, truncated, info = env.step(command)
        assert np.isfinite(obs).all()
        plant.extend(env.last_trace)
        estimate.extend(env.last_observer_trace)
        if on_frame is not None:
            on_frame(env)
        if ended or truncated:
            return info, np.asarray(plant), np.asarray(estimate)


def measurements(env, info, plant, estimate):
    t = plant[:, 0]
    np.testing.assert_array_equal(t, estimate[:, 0])
    f = np.linalg.norm(np.c_[plant[:, 9:11], estimate[:, 10]], axis=1)
    touched, detected, opened = (info[k] for k in ("first_touch_s", "estimated_touch_s", "release_s"))
    held = t <= opened if opened is not None else np.ones(len(t), dtype=bool)
    anchored = np.isfinite(estimate[:, 16]) & held
    offset = (float(1000 * np.max(np.abs(plant[anchored, 6] - estimate[anchored, 16])))
              if anchored.any() else None)
    if offset is not None:
        assert offset <= REFERENCE_TRAVEL_M * 1000 + 1e-8
    shift_window = (t <= CARRY_TIME) & (t >= .3)
    at_crossing = f[t == CARRY_TIME]
    at_touch = f[t == touched]
    carry_z_error = plant[shift_window, 3] - plant[shift_window, 6]
    peak_sample = np.argmax(plant[:, 7])
    return dict(**info, force_peak_n=float(f.max()), force_at_phase_switch_n=float(at_crossing.max()) if at_crossing.size else None,
                disturbance_at_touch_n=float(at_touch.max()) if at_touch.size else None,
                force_during_landing_s=float(DT * np.count_nonzero((t >= CARRY_TIME) & (f > .2))),
                force_during_contact_s=float(DT * np.count_nonzero((t >= touched) & held & (f > .2))) if touched is not None else None,
                placement_peak_s=float(t[peak_sample]) if touched is not None else None,
                estimated_contact_early=detected is not None and (touched is None or detected < touched),
                opened_early=opened is not None and (touched is None or opened < touched),
                max_local_z_reference_mm=offset, mass_update_steps=env.mass_updates, contact_rejections=env.contact_rejections,
                carry_vertical_rms_mm=float(1000 * np.sqrt(np.mean(carry_z_error**2))) if carry_z_error.size else None,
                max_z_separation_mm=float(1000 * np.max(np.abs(plant[held, 3] - estimate[held, 15]))),
                mean_carry_feedforward_n=float(np.mean(estimate[t < CARRY_TIME, 14])))


def evaluate(job):
    method, context = job
    ctrl, gains = METHODS[method]
    env = CoupledCarryPlaceEnv(baseline=ctrl, fixed_gains=gains)
    cases = cases_for(context)
    modes = list(PACES) if method == STRONG else ["middle"]
    trials = [("main", profile, name, True, action_for(PACES[name])) for name in modes for profile in PROFILES]
    if method == STRONG:
        trials.append(("feedforward", "sustained", "nominal_mass", False, action_for(PACES["middle"])))
        if context["height"] >= .02:
            trials.extend(("braking", profile, "late_brake", True, action_for(PACES["slow"], .054))
                          for profile in ("quiet", "sustained"))
        if context["id"] in (0, 3, 16, 19):
            trials.extend(("grip", "sustained", name, True, action_for(PACES["middle"], grip=value))
                          for name, value in (("tight", -1.), ("loose", 1.)))
    try:
        rows = []
        for group, profile, pace, ff, action in trials:
            env.estimated_mass_feedforward = ff
            scenario = cases[profile]
            info, plant, estimate = replay(env, scenario, action)
            rows.append(dict(group=group, context=context["id"], method=method, profile=profile, pace=pace,
                             action=action.tolist(), estimated_mass_feedforward=ff, scenario=asdict(scenario),
                             **measurements(env, info, plant, estimate)))
        return rows
    finally:
        env.close()


def summary(rows):
    contact = [r for r in rows if r["first_touch_s"] is not None]
    complete = [r for r in rows if r["status"] in ("settled", "impact_limit", "overhang", "premature_release")]
    peaks = [r["placement_peak_force_n"] for r in contact]
    return dict(n=len(rows), outcome=dict(Counter(r["status"] for r in rows)), touched=len(contact), completed=len(complete),
                mean_peak_n=float(np.mean(peaks)) if peaks else None,
                peak_p90_n=float(np.quantile(peaks, .90)) if peaks else None,
                median_completed_s=float(np.median([r["elapsed_s"] for r in complete])) if complete else None,
                median_touch_speed_mm_s=float(1000*np.median([abs(r["contact_speed_z_m_s"]) for r in contact])) if contact else None,
                overlapped_touch=sum(r["disturbance_at_touch_n"] > .2 for r in contact),
                impact_exceeded=sum(bool(r["placement_peak_force_n"] > PEAK_LIMIT_N) for r in rows),
                early_contact=sum(r["estimated_contact_early"] for r in rows), early_open=sum(r["opened_early"] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("results/coupled_pilot"))
    args = parser.parse_args()
    contexts = []
    rng = np.random.default_rng(SEED)
    for h_i, height in enumerate((-.04, -.02, 0., .02, .04)):
        for s_i, arm_scale in enumerate((.96, .98, 1.02, 1.04)):
            contexts.append(dict(id=len(contexts), height=height, arm_scale=arm_scale, mass=(.18, .26, .34)[(h_i+s_i) % 3],
                                 friction=float(rng.uniform(.46, .79)), payload_com_m=rng.uniform([-.008, -.008, -.005], [.008, .008, .005]).tolist()))
    rows = []
    jobs = [(method, context) for method in METHODS for context in contexts]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, batch in enumerate(pool.map(evaluate, jobs), 1):
            rows.extend(batch)
            if i % 10 == 0 or i == len(jobs):
                print(f"{len(rows)} episodes complete", flush=True)
    primary = [r for r in rows if r["group"] == "main"]
    stats = {f"{method}/{pace}": summary([r for r in primary if (r["method"], r["pace"]) == (method, pace)])
             for method in METHODS for pace in (PACES if method == STRONG else ("middle",))}
    args.output.mkdir(parents=True, exist_ok=True)
    run = dict(config=dict(seed=SEED, deadline_s=DEADLINE_S, peak_limit_n=PEAK_LIMIT_N,
                          fixed_brake_onset_m=BRAKE_ONSET_M, speed_choices_m_s=PACES, contexts=contexts), episodes=rows)
    for name, obj in (("runs", run), ("summary", stats)):
        (args.output / f"{name}.json").write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
