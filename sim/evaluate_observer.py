"""Paired feasibility tests for proprioceptive carry and placement."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from sim.carry_place import CARRY_TIME, DT, OBSERVER_DEADLINE, PLACEMENT_PEAK_LIMIT_N, SUPPORT_TARGET_N, CarryPlaceEnv, Scenario
from sim.evaluate_disturbance import METHODS
from sim.momentum_observer import CONTACT_DEBOUNCE_S, CONTACT_LEVEL_N, NOMINAL_LOAD_KG, OBSERVER_GAIN, WRENCH_LAG_S, WRENCH_NOISE_STD


HEIGHTS_MM = (-40, -25, 0, 25, 40)
MASSES_KG = (0.18, 0.26, 0.34)
PROFILES = ("quiet", "sustained", "random_pulses")
SENSOR_SEED = 27


def make_case(height_mm, mass_kg, profile):
    params = dict(height=height_mm / 1000, mass=mass_kg, friction=0.62)
    if profile == "quiet":
        return Scenario(**params, pulse=0.0)
    if profile == "sustained":
        return Scenario(**params, pulse=2.0, angle_deg=36.0, force_z_n=-2.0,
                        force_intervals=((0.72, 1.62),), force_ramp_s=0.09)
    if profile != "random_pulses":
        raise ValueError(profile)
    rng = np.random.default_rng(1729 + height_mm + round(1000 * mass_kg))
    starts = np.round(rng.uniform([.51, 1.04, 1.54], [.65, 1.20, 1.74]) / DT) * DT
    ends = starts + np.round(rng.uniform(.07, .18, 3) / DT) * DT
    magnitudes = rng.uniform(1.0, 3.5, size=3)
    angles = rng.uniform(-np.pi, np.pi, size=3)
    xy = magnitudes[:, None] * np.stack((np.cos(angles), np.sin(angles)), axis=1)
    return Scenario(**params, pulse=0.0, force_z_n=2.2, force_ramp_s=0.018,
                    force_intervals=tuple((round(a, 6), round(b, 6)) for a, b in zip(starts, ends)),
                    force_vectors_xy_n=tuple(tuple(map(float, v)) for v in xy))


def replay(env, scenario, frame=None):
    observation, _ = env.reset(seed=SENSOR_SEED, options={"scenario": scenario})
    assert observation.shape == (39,) and np.isfinite(observation).all()
    truth, estimates = [], []
    action = np.zeros(2)
    if frame is not None:
        frame(env)
    while True:
        observation, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(observation).all() and np.isfinite(reward)
        truth.extend(env.last_trace)
        estimates.extend(env.last_observer_trace)
        if frame is not None:
            frame(env)
        if terminated or truncated:
            break
    return info, np.asarray(truth), np.asarray(estimates)


def measures(info, truth, estimate):
    time = truth[:, 0]
    np.testing.assert_array_equal(time, estimate[:, 0])
    if info["first_touch_s"] is not None:
        assert np.isclose(truth[:, 7].max(), info["placement_peak_force_n"])
    touch, detected, opening = (info[k] for k in ("first_touch_s", "estimated_touch_s", "release_s"))
    in_carry = time <= CARRY_TIME
    # Over the same 80 ms interval that gates opening in the controller.
    pre_open = ((np.round(opening - time, 9) < .08) & (time <= opening)
                if opening is not None else np.zeros(len(time), dtype=bool))
    return dict(**info, contact_delay_s=round(detected - touch, 9) if touch is not None and detected is not None else None,
                detection_before_touch=detected is not None and (touch is None or detected < touch),
                opened_before_touch=opening is not None and (touch is None or opening < touch),
                true_support_before_open_n=float(truth[pre_open, 7].mean()) if pre_open.any() else None,
                estimated_support_before_open_n=float(estimate[pre_open, 7].mean()) if pre_open.any() else None,
                carry_vertical_rms_mm=float(1000 * np.sqrt(np.mean((truth[in_carry, 3] - truth[in_carry, 6])**2))),
                applied_force_z_peak_n=float(np.max(np.abs(estimate[:, 10]))))


def evaluate_group(job):
    group, method, gains, height_mm, mass_kg, profiles, model_scale = job
    baseline = METHODS[method][0] if method in METHODS else "fixed"
    env = CarryPlaceEnv(baseline=baseline, fixed_gains=gains, target_width_m=0.064, observer_feedback=True)
    if model_scale != 1.0:
        observer = env.observer
        # Deliberately perturb the estimator's arm model. Motor control still
        # runs the independent nominal arm model used in the main experiment.
        arm = [body for body in range(1, observer.model.nbody) if body != observer.model.body("nominal_load").id]
        observer.model.body_mass[arm] *= model_scale
        observer.model.body_inertia[arm] *= model_scale
        mj.mj_setConst(observer.model, observer.data)
    rows = []
    calm_z = None
    try:
        for profile in profiles:
            case = make_case(height_mm, mass_kg, profile)
            info, truth, estimate = replay(env, case)
            point = measures(info, truth, estimate)
            z = truth[truth[:, 0] <= CARRY_TIME, 3]
            if profile == "quiet":
                calm_z = z
            if calm_z is not None:
                n = min(len(z), len(calm_z))
                point["peak_additional_carry_z_mm"] = float(1000 * np.max(np.abs(z[:n] - calm_z[:n])))
            rows.append(dict(group=group, method=method, fixed_gains_n_m=gains if baseline == "fixed" else None,
                             observer_arm_scale=model_scale, height_mm=height_mm, mass_kg=mass_kg, profile=profile,
                             scenario=asdict(case), **point))
    finally:
        env.close()
    return rows


def summarize(rows):
    delays_ms = [1000 * r["contact_delay_s"] for r in rows if r["contact_delay_s"] is not None]
    peaks_n = [r["placement_peak_force_n"] for r in rows if r["first_touch_s"] is not None]
    times_s = [r["elapsed_s"] for r in rows if r["is_success"]]
    held = [(r["true_support_before_open_n"], r["estimated_support_before_open_n"]) for r in rows
            if r["release_s"] is not None]
    mean_abs_mass_error_g = np.mean([abs(r["estimated_mass_kg"] - r["mass_kg"]) * 1000 for r in rows])
    return dict(n=len(rows), status_counts=dict(Counter(r["status"] for r in rows)),
                mass_mae_g=float(mean_abs_mass_error_g), detections_with_touch=len(delays_ms),
                false_contacts=sum(r["detection_before_touch"] for r in rows),
                early_opens=sum(r["opened_before_touch"] for r in rows),
                contact_delay_range_ms=[float(min(delays_ms)), float(max(delays_ms))] if delays_ms else None,
                contact_delay_p50_ms=float(np.median(delays_ms)) if delays_ms else None,
                peak_force_p90_n=float(np.quantile(peaks_n, .90)) if peaks_n else None,
                max_peak_force_n=float(max(peaks_n)) if peaks_n else None,
                mean_success_time_s=float(np.mean(times_s)) if times_s else None,
                mean_true_support_before_open_n=float(np.mean([a for a, _ in held])) if held else None,
                support_mae_before_open_n=float(np.mean([abs(a-b) for a, b in held])) if held else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/observer_pilot"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    jobs = [("main", method, gains, height, mass, PROFILES, 1.0) for method, (_, gains) in METHODS.items()
            for height in HEIGHTS_MM for mass in MASSES_KG]
    jobs += [("kz_sweep", f"fixed_520_{kz}", (520, kz), height, mass, ("quiet", "sustained"), 1.0)
             for kz in (140, 520) for height in (-40, 40) for mass in (.18, .34)]
    jobs += [("observer_bias", "fixed_520_300", (520, 300), height, mass, ("sustained",), scale)
             for scale in (.98, 1.02) for height in (-40, 0, 40) for mass in (.18, .34)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, batch in enumerate(pool.map(evaluate_group, jobs), 1):
            rows.extend(batch)
            if i % 12 == 0 or i == len(jobs):
                print(f"{len(rows)} episodes complete", flush=True)
    main_rows = [r for r in rows if r["group"] == "main"]
    stats = {method: summarize([r for r in main_rows if r["method"] == method]) for method in METHODS}
    stats["all_main"] = summarize(main_rows)
    stats["observer_mass_bias"] = {str(k): summarize([r for r in rows if r["group"] == "observer_bias" and r["observer_arm_scale"] == k])
                                  for k in (.98, 1.02)}
    config = dict(dt_s=DT, sensor_seed=SENSOR_SEED, nominal_load_kg=NOMINAL_LOAD_KG,
                  observer_gain_per_s=OBSERVER_GAIN, force_noise_sigma=WRENCH_NOISE_STD.tolist(),
                  output_lag_s=WRENCH_LAG_S, contact_threshold_n=CONTACT_LEVEL_N, contact_debounce_s=CONTACT_DEBOUNCE_S,
                  target_support_n=SUPPORT_TARGET_N, deadline_s=OBSERVER_DEADLINE, peak_limit_n=PLACEMENT_PEAK_LIMIT_N)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in (("runs", dict(config=config, episodes=rows)), ("summary", stats)):
        (args.output / f"{name}.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
