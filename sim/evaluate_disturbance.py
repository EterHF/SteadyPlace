"""Measure scripted baselines under matched horizontal disturbance schedules."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from sim.carry_place import CARRY_TIME, DT, CarryPlaceEnv, Scenario
from sim.demo import rollout


SEED = 2718
METHODS = {
    "fixed_360_300": ("fixed", (360.0, 300.0)),
    "phase_rule": ("phase", (360.0, 300.0)),
    "fixed_520_300": ("fixed", (520.0, 300.0)),
}
# Intervals are offsets from a common force-end time. The three nonzero schedules
# deliver the same 0.64 N s at the 2 ms physics rate.
PROFILES = {
    "none": (0.0, ()),
    "single": (8.0, ((-0.08, 0.0),)),
    "triple": (8.0 / 3.0, ((-0.72, -0.64), (-0.40, -0.32), (-0.08, 0.0))),
    "constant": (8.0 / 9.0, ((-0.72, 0.0),)),
}
TIMINGS = {"carry": (1.30, (-20, 0, 20)), "descent": (2.80, (0, 20))}


def run_context(job):
    method, context, base_scenario = job
    baseline, gains = METHODS[method]
    env = CarryPlaceEnv(baseline=baseline, fixed_gains=gains)
    records = []
    try:
        for timing, (force_end_s, heights_mm) in TIMINGS.items():
            for height_mm in heights_mm:
                for profile, (amplitude, offsets) in PROFILES.items():
                    if timing == "descent" and profile == "none":
                        continue  # The unforced reference is shared across timings.
                    intervals = tuple((round(force_end_s + start, 3), round(force_end_s + end, 3))
                                      for start, end in offsets)
                    scenario = replace(base_scenario, height=height_mm / 1000, pulse=amplitude,
                                       force_intervals=intervals)
                    info, trace = rollout(env, scenario)
                    samples = np.asarray(trace)
                    time = samples[:, 0]
                    first_touch = info["first_touch_s"]
                    descent_end = first_touch if first_touch is not None else info["elapsed_s"]
                    descending = (time > CARRY_TIME) & (time <= descent_end)
                    xy_error = samples[descending, 1:3] - samples[descending, 4:6]
                    descent_rms = float(np.sqrt(np.mean(np.sum(xy_error**2, axis=1)))) if xy_error.size else None
                    force = np.linalg.norm(samples[:, 9:11], axis=1)
                    applied_impulse = float(DT * force.sum())
                    active_steps = int(np.count_nonzero(force))
                    if profile == "none" or info["elapsed_s"] >= force_end_s:
                        expected_steps = sum(round((end - start) / DT) for start, end in intervals)
                        expected_impulse = 0.0 if profile == "none" else 0.64
                        if active_steps != expected_steps or not np.isclose(applied_impulse, expected_impulse):
                            raise AssertionError(f"Unexpected measured disturbance for {method}/{timing}/{profile}")
                    load_past_touch = first_touch is not None and bool(np.any(force[time > first_touch] > 0))
                    records.append(dict(method=method, context=context, timing=timing, profile=profile,
                                        height_mm=height_mm, disturbance_impulse_n_s=applied_impulse,
                                        disturbance_steps=active_steps, load_past_touch=load_past_touch,
                                        rms_descent_error_m=descent_rms, **info))
    finally:
        env.close()
    return records


def stats(rows):
    contacts = [row for row in rows if row["contact_window_complete"]]
    successful = [row for row in rows if row["is_success"]]
    values = {
        "carry_rms_mm": [row["rms_carry_error_m"] * 1000 for row in rows],
        "carry_peak_mm": [row["max_carry_error_m"] * 1000 for row in rows],
        "descent_rms_mm": [row["rms_descent_error_m"] * 1000 for row in contacts
                           if row["rms_descent_error_m"] is not None],
        "touch_speed_mm_s": [abs(row["contact_speed_z_m_s"]) * 1000 for row in contacts],
        "touch_peak_n": [row["peak_force_n"] for row in contacts],
        "touch_impulse_n_s": [row["impulse_100ms_n_s"] for row in contacts],
        "settling_time_s": [row["elapsed_s"] for row in successful],
        "final_xy_mm": [row["final_xy_error_m"] * 1000 for row in successful],
        "final_tilt_deg": [row["final_tilt_deg"] for row in successful],
    }
    result = dict(attempts=len(rows), successes=len(successful), complete_contact_windows=len(contacts),
                  load_past_touch=sum(row["load_past_touch"] for row in rows),
                  status_counts=dict(Counter(row["status"] for row in rows)))
    for name, samples in values.items():
        result[name] = (dict(mean=float(np.mean(samples)), p90=float(np.quantile(samples, 0.90)),
                             max=float(np.max(samples))) if samples else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/disturbance_benchmark"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    angles = np.linspace(-157.5, 157.5, 8)
    contexts = [Scenario(mass=float(mass), friction=float(friction), angle_deg=float(angle))
                for mass, friction, angle in zip(rng.uniform(0.18, 0.34, len(angles)),
                                                rng.uniform(0.45, 0.80, len(angles)), angles)]
    jobs = [(method, index, context) for method in METHODS for index, context in enumerate(contexts)]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for batch in pool.map(run_context, jobs):
            records.extend(batch)
            print(f"{len(records):3d} episodes complete", flush=True)

    groups = defaultdict(list)
    for row in records:
        method, timing, profile, height = (row[key] for key in ("method", "timing", "profile", "height_mm"))
        groups[method, "all", "all", "all"].append(row)
        groups[method, timing, profile, "all"].append(row)
        groups[method, timing, profile, height].append(row)
    summary = [dict(method=method, timing=timing, profile=profile, height_mm=height, **stats(rows))
               for (method, timing, profile, height), rows in groups.items()]
    config = dict(seed=SEED, dt_s=DT, carry_duration_s=CARRY_TIME, methods=METHODS, profiles=PROFILES,
                  timing=TIMINGS, contexts=[asdict(context) for context in contexts])
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in (("runs", dict(config=config, episodes=records)), ("summary", summary)):
        (args.output / f"{name}.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n",
                                                  encoding="utf-8")


if __name__ == "__main__":
    main()
