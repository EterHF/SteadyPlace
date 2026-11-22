"""Freeze matched evaluation cases and score the current run's SAC checkpoints."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import lru_cache
from itertools import product
import json
from pathlib import Path
from time import sleep

import numpy as np

from sim.carry_place import Scenario
from sim.coupled_place import DEADLINE_S, PEAK_LIMIT_N, PROFILES, CoupledCarryPlaceEnv, disturbance_program
from sim.evaluate_coupled import action_for, measurements, replay, summary


SPLIT_FILE = Path("split.json")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, contents):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(contents, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def sample_contexts(seed, contexts_per_height_bin):
    rng = np.random.default_rng(seed)
    edges = np.linspace(-.04, .04, 6)
    contexts = []
    for band, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        for _ in range(contexts_per_height_bin):
            plant = dict(height=float(rng.uniform(low, high)), mass=float(rng.uniform(.18, .34)),
                         friction=float(rng.uniform(.46, .79)), arm_scale=float(rng.uniform(.96, 1.04)),
                         payload_com_m=tuple(map(float, rng.uniform([-.008, -.008, -.005], [.008, .008, .005]))))
            cases = []
            for profile in PROFILES:
                case = Scenario(**plant, **disturbance_program(profile, rng))
                cases.append(dict(profile=profile, scenario=asdict(case)))
            contexts.append(dict(id=len(contexts), height_band=band, reset_seed=int(rng.integers(2**31)), cases=cases))
    return contexts


def constant_specs(protocol):
    reference = protocol["fixed_reference"]
    fixed = dict(name="fixed_520_300", baseline="fixed", gains=reference["gains_n_m"],
                 action=action_for(reference["near_speed_mm_s"] / 1000, reference["brake_mm"] / 1000,
                                   reference["grip_action"]).tolist())
    grid = protocol["static_grid"]
    candidates = []
    for speed, brake, gains in product(grid["near_speed_mm_s"], grid["brake_mm"], grid["gain_residuals"]):
        action = action_for(speed / 1000, brake / 1000, grid["grip_action"])
        action[:2] = gains
        candidates.append(dict(name=f"constant_{len(candidates) + 1:02d}", baseline="phase", action=action.tolist(),
                               near_speed_mm_s=speed, brake_mm=brake, gain_residuals=gains))
    return fixed, candidates


@lru_cache(maxsize=1)
def load_policy(path, step):
    import torch
    from stable_baselines3 import SAC

    torch.set_num_threads(1)
    model = SAC.load(path, device="cpu")
    if model.num_timesteps != step or model.observation_space.shape != (42,) or model.action_space.shape != (5,):
        raise ValueError(f"Unexpected checkpoint for step {step}")
    return model


def evaluate_context(job):
    spec, context = job
    env = CoupledCarryPlaceEnv(baseline=spec["baseline"], fixed_gains=spec.get("gains", (360., 300.)))
    try:
        if "checkpoint" in spec:
            policy = load_policy(spec["checkpoint"], spec["step"])

            def action(obs):
                return policy.predict(obs, deterministic=True)[0]
        else:
            action = np.asarray(spec["action"], dtype=float)
        rows = []
        for item in context["cases"]:
            case = Scenario(**item["scenario"])
            info, plant, estimate = replay(env, case, action, seed=context["reset_seed"])
            rows.append(dict(method=spec["name"], context=context["id"], height_band=context["height_band"],
                             profile=item["profile"], **measurements(env, info, plant, estimate)))
        return rows
    finally:
        env.close()


def evaluate(pool, specs, contexts, filename):
    path = Path(filename)
    if path.exists():
        previous = read_json(path)
        if previous["specs"] != specs:
            raise ValueError(f"Configuration differs from existing {filename}")
        return previous
    rows = []
    for batch in pool.map(evaluate_context, product(specs, contexts)):
        rows.extend(batch)
    if len(rows) != len(specs) * len(contexts) * len(PROFILES):
        raise ValueError("Incomplete paired evaluation")
    result = dict(specs=specs, episodes=rows)
    write_json(path, result)
    print(f"Saved {filename} with {len(rows)} episodes.", flush=True)
    return result


def score(rows):
    stats = summary(rows)
    stats["successful"] = stats["outcome"].get("settled", 0)
    successful = [r["elapsed_s"] for r in rows if r["is_success"]]
    stats["median_success_s"] = float(np.median(successful)) if successful else None
    stats["mean_carry_rms_mm"] = float(1000 * np.mean([r["rms_carry_error_m"] for r in rows]))
    return stats


def paired(policy, baseline):
    def indexed(rows):
        return {(r["context"], r["profile"]): r for r in rows}

    p, b = indexed(policy), indexed(baseline)
    if p.keys() != b.keys() or len(p) != len(policy) or len(b) != len(baseline):
        raise ValueError("Policy and baseline must be scored on the same unique cases")
    pairs = [(p[key], b[key]) for key in p]
    new_pass = sum(a["is_success"] and not b["is_success"] for a, b in pairs)
    lost_pass = sum(b["is_success"] and not a["is_success"] for a, b in pairs)
    force = [a["placement_peak_force_n"] - b["placement_peak_force_n"] for a, b in pairs
             if a["first_touch_s"] is not None and b["first_touch_s"] is not None]
    elapsed = [a["elapsed_s"] - b["elapsed_s"] for a, b in pairs if a["is_success"] and b["is_success"]]
    return dict(gained_success=new_pass, lost_success=lost_pass, delta_success_pp=100 * (new_pass - lost_pass) / len(pairs),
                both_touched=len(force), paired_median_peak_delta_n=float(np.median(force)) if force else None,
                both_successful=len(elapsed), paired_median_time_delta_s=float(np.median(elapsed)) if elapsed else None)


def report(policy, baselines):
    combined = {"sac": policy}
    for method in dict.fromkeys(r["method"] for r in baselines):
        combined[method] = [r for r in baselines if r["method"] == method]
    return dict(
        methods={name: dict(overall=score(rows),
                            profiles={p: score([r for r in rows if r["profile"] == p]) for p in PROFILES},
                            height_bands={str(i): score([r for r in rows if r["height_band"] == i]) for i in range(5)})
                 for name, rows in combined.items()},
        paired={name: paired(policy, rows) for name, rows in combined.items() if name != "sac"},
    )


def test_checkpoint(pool, step, config, val, baselines, split_name):
    filename = "policy.zip" if step == config["steps"] else f"checkpoint_{step}.zip"
    train_pid = Path("train.pid").read_text().strip()
    while not Path(filename).is_file():
        if Path("train.exit").exists() or not Path("/proc", train_pid).is_dir():
            if Path(filename).is_file():
                break
            raise RuntimeError(f"Training exited before saving {filename}; see train.log")
        sleep(10)
    policy = dict(name="sac", baseline="phase", checkpoint=filename, step=step)
    tag = f"{split_name}_{step}"
    data = evaluate(pool, [policy], val, f"{tag}_runs.json")
    stats = report(data["episodes"], baselines)
    write_json(f"{tag}.json", stats)
    print(f"{tag}: SAC {stats['methods']['sac']['overall']['successful']}/{len(data['episodes'])} successes.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Only freeze the splits")
    args = parser.parse_args()
    protocol, config = read_json("protocol.json"), read_json("config.json")
    criteria = protocol["criteria"]
    if (criteria["deadline_s"] != DEADLINE_S or criteria["peak_force_n"] != PEAK_LIMIT_N
            or not config["coupled_approach"] or not config["force_feedback"]
            or config.get("target_width_m", .064) * 1000 != criteria["target_width_mm"]):
        raise ValueError("Protocol differs from the simulated task")
    if not SPLIT_FILE.is_file():
        write_json(SPLIT_FILE, {name: sample_contexts(**protocol[name]) for name in ("validation", "test")})
    if args.prepare:
        return

    split = read_json(SPLIT_FILE)
    fixed, candidates = constant_specs(protocol)
    validation, test = split["validation"], split["test"]
    milestones = [*config["checkpoint_steps"], config["steps"]]
    with ProcessPoolExecutor(max_workers=protocol["workers"]) as pool:
        fixed_val = evaluate(pool, [fixed], validation, "validation_fixed.json")["episodes"]
        early = milestones[0]
        if Path(f"checkpoint_{early}.zip").is_file():
            test_checkpoint(pool, early, config, validation, fixed_val, "early")

        grid = evaluate(pool, candidates, validation, "validation_grid.json")["episodes"]
        grid_stats = {s["name"]: score([r for r in grid if r["method"] == s["name"]]) for s in candidates}

        def selection_key(candidate):
            stats = grid_stats[candidate["name"]]
            return (-stats["successful"], stats["outcome"].get("timeout", 0), stats["impact_exceeded"])

        chosen = min(candidates, key=selection_key)
        write_json("baseline_selection.json", dict(selected=chosen, fixed=score(fixed_val), grid=grid_stats))
        print(f"Selected constant baseline {chosen['name']}.", flush=True)
        val_baselines = fixed_val + [r for r in grid if r["method"] == chosen["name"]]
        test_baselines = evaluate(pool, [fixed, chosen], test, "test_baselines.json")["episodes"]
        for step in milestones:
            test_checkpoint(pool, step, config, validation, val_baselines, "validation")
        test_checkpoint(pool, config["steps"], config, test, test_baselines, "test")


if __name__ == "__main__":
    main()
