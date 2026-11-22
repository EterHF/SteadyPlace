"""Cross-touch disturbances and precision placement on a small support surface."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from sim.carry_place import CARRY_TIME, DT, CarryPlaceEnv, Scenario
from sim.demo import rollout
from sim.evaluate_disturbance import METHODS, stats


SEED = 7319
SURFACES = {"wide": None, "pad_64mm": 0.064}
HEIGHTS_MM = (0, 20)
START_GAP_M = 0.020
# For descent cases, intervals are relative to the first crossing of START_GAP_M.
CASES = {
    "none": (0.0, None, ()),
    "carry_constant_8n": (8.0, None, ((0.58, 1.30),)),
    "descent_single_8n": (8.0, START_GAP_M, ((0.00, 0.08),)),
    "descent_triple_8n": (8.0, START_GAP_M, ((0.00, 0.08), (0.32, 0.40), (0.64, 0.72))),
    "descent_constant_4n": (4.0, START_GAP_M, ((0.00, 0.72),)),
    "descent_constant_8n": (8.0, START_GAP_M, ((0.00, 0.72),)),
}


def evaluate_episode(env, scenario):
    info, trace = rollout(env, scenario)
    samples = np.asarray(trace)
    time = samples[:, 0]
    touch_s = info["first_touch_s"]
    descent_end = touch_s if touch_s is not None else info["elapsed_s"]
    descending = (time > CARRY_TIME) & (time <= descent_end)
    xy_error = samples[descending, 1:3] - samples[descending, 4:6]
    descent_rms = float(np.sqrt(np.mean(np.sum(xy_error**2, axis=1)))) if xy_error.size else None

    force = np.linalg.norm(samples[:, 9:11], axis=1)
    active = time[force > 0.0]
    force_end = info["elapsed_s"]
    if scenario.approach_force_gap_m is not None and info["release_s"] is not None:
        force_end = info["release_s"]
    expected_steps = 0
    if env.force_origin is not None:
        expected_steps = sum(round(max(0.0, min(env.force_origin + stop, force_end) - env.force_origin - start) / DT)
                             for start, stop in scenario.force_intervals)
    if active.size != expected_steps or np.any(active > force_end):
        raise AssertionError(f"Force schedule diverged from the requested intervals for {scenario}")

    # Projection of an oriented box onto each of the horizontal world axes.
    rotation = env.data.geom_xmat[env.payload_geom].reshape(3, 3)
    projected_half_width = np.abs(rotation[:2]) @ env.model.geom_size[env.payload_geom]
    displacement = env.data.geom_xpos[env.payload_geom, :2] - env.data.geom_xpos[env.target_geom, :2]
    edge_margin = env.model.geom_size[env.target_geom, :2] - np.abs(displacement) - projected_half_width
    margin_mm = float(np.min(edge_margin)) * 1000
    release_xy = samples[time == info["release_s"], 1:3]
    release_axis_offset_mm = (float(np.abs(release_xy[0] - env.data.geom_xpos[env.target_geom, :2]).max()) * 1000
                              if release_xy.size else None)
    full_support = info["is_success"] and margin_mm >= 0.0
    outcome = ("full_support" if full_support else "overhang") if info["is_success"] else info["status"]

    return dict(**info, outcome=outcome, full_support=full_support, footprint_margin_mm=margin_mm,
                release_axis_offset_mm=release_axis_offset_mm,
                rms_descent_error_m=descent_rms, load_at_touch=bool(np.any(active == touch_s)),
                load_past_touch=bool(touch_s is not None and np.any(active > touch_s)),
                disturbance_impulse_n_s=float(force.sum() * DT), force_steps=int(active.size),
                force_origin_s=env.force_origin if scenario.approach_force_gap_m is not None else None,
                force_start_s=float(active[0] - DT) if active.size else None,
                force_end_s=float(active[-1]) if active.size else None)


def evaluate_group(job):
    method, surface, context_index, context = job
    baseline, gains = METHODS[method]
    env = CarryPlaceEnv(baseline=baseline, fixed_gains=gains, target_width_m=SURFACES[surface])
    records = []
    try:
        for height_mm in HEIGHTS_MM:
            for condition, (force_n, gap_m, intervals) in CASES.items():
                scenario = replace(context, height=height_mm / 1000, pulse=force_n,
                                   force_intervals=intervals, approach_force_gap_m=gap_m)
                measurements = evaluate_episode(env, scenario)
                records.append(dict(method=method, surface=surface, context=context_index,
                                    height_mm=height_mm, condition=condition, **measurements))
    finally:
        env.close()
    return records


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        method, surface, height, condition = (row[key] for key in ("method", "surface", "height_mm", "condition"))
        groups[method, surface, height, condition].append(row)
        groups[method, surface, "all", condition].append(row)
    summary = []
    for (method, surface, height, condition), rows in groups.items():
        measurements = stats(rows)
        fitted = [row for row in rows if row["full_support"]]
        stable = [row for row in rows if row["is_success"]]
        measurements.update(
            full_support=len(fitted), outcome_counts=dict(Counter(row["outcome"] for row in rows)),
            load_at_touch=sum(row["load_at_touch"] for row in rows),
            force_impulse_mean_n_s=float(np.mean([row["disturbance_impulse_n_s"] for row in rows])),
            mean_full_support_time_s=float(np.mean([row["elapsed_s"] for row in fitted])) if fitted else None,
            min_stable_footprint_margin_mm=min((row["footprint_margin_mm"] for row in stable), default=None),
        )
        summary.append(dict(method=method, surface=surface, height_mm=height, condition=condition, **measurements))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/challenge_benchmark"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    angles = np.linspace(-157.5, 157.5, 8)
    contexts = [Scenario(mass=float(mass), friction=float(friction), angle_deg=float(angle))
                for mass, friction, angle in zip(rng.uniform(0.18, 0.34, len(angles)),
                                                rng.uniform(0.45, 0.80, len(angles)), angles)]
    jobs = [(method, surface, i, context) for method in METHODS for surface in SURFACES
            for i, context in enumerate(contexts)]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for batch in pool.map(evaluate_group, jobs):
            records.extend(batch)
            if len(records) % 96 == 0:
                print(f"{len(records):3d} episodes complete", flush=True)

    config = dict(seed=SEED, dt_s=DT, surfaces_m={"wide": (0.38, 0.34), "pad_64mm": (0.064, 0.064)},
                  heights_mm=HEIGHTS_MM, cases=CASES, methods=METHODS, contexts=[asdict(item) for item in contexts])
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in (("runs", dict(config=config, episodes=records)), ("summary", summarize(records))):
        (args.output / f"{name}.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n",
                                                  encoding="utf-8")


if __name__ == "__main__":
    main()
