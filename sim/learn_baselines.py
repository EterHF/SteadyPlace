"""Train and evaluate residual SAC and TD3 on the fourth placement."""

from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json
from pathlib import Path

import gymnasium as gym
import mujoco as mj
import numpy as np

from sim.evaluate_baselines import aligned_tower_check
from sim.stack import BLOCK_COUNT, BLOCK_WIDTH, TABLE_Z, Placement, StackRun


TRAIN_CASES = tuple(product((1.36, 1.40, 1.44, 1.48, 1.52), range(0, 360, 30)))
TEST_CASES = tuple(product((1.38, 1.42, 1.46, 1.50), range(15, 360, 60)))
STATE = mj.mjtState.mjSTATE_INTEGRATION
FIXED_ACTION = np.array([0.0, 0.0, -1.0], dtype=np.float32)
OFFSET_LIMIT_M = 0.010
FEATURES = 4 + 10 * BLOCK_COUNT


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_snapshot(case):
    period, phase_deg = case
    run = StackRun(period, tray_phase=np.radians(phase_deg))
    layers = [run.place(i) for i in range(BLOCK_COUNT - 1)]
    p = run.prepare_placement(BLOCK_COUNT - 1)
    state = np.empty(mj.mj_stateSize(run.model, STATE))
    mj.mj_getState(run.model, run.data, state, STATE)
    return {
        "period_s": period,
        "start_phase_deg": phase_deg,
        "layers": layers,
        "placement": {
            "layer": p.layer, "landing": p.landing.tolist(), "above_tray": p.above_tray.tolist(),
            "payload_kg": p.payload_kg, "grasped": p.grasped, "lift_height_mm": p.lift_height_mm,
        },
        "state": state.tolist(),
    }


class LandingEnv(gym.Env):
    """One residual placement decision, followed by a full physical stability check."""

    def __init__(self, snapshots):
        self.snapshots = snapshots
        self.run = StackRun(snapshots[0]["period_s"])
        joints = self.run.model.body_jntadr[self.run.blocks]
        self.dofs = self.run.model.jnt_dofadr[joints][:, None] + np.arange(6)
        self.action_space = gym.spaces.Box(-1, 1, (3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-10, 10, (FEATURES,), dtype=np.float32)
        self.current = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        index = options["case_index"] if options else self.np_random.integers(len(self.snapshots))
        self.current = self.snapshots[index]
        run = self.run
        run.period = self.current["period_s"]
        run.omega = 2 * np.pi / run.period
        run.tray_phase = np.radians(self.current["start_phase_deg"])
        raw = self.current["placement"]
        self.placement = Placement(**{**raw, "landing": np.array(raw["landing"]),
                                      "above_tray": np.array(raw["above_tray"])})
        run.reference = self.placement.above_tray.copy()
        run.follow = True
        mj.mj_resetData(run.model, run.data)
        mj.mj_setState(run.model, run.data, np.array(self.current["state"]), STATE)
        mj.mj_forward(run.model, run.data)
        return self._observation(), {}

    def _observation(self):
        run = self.run
        angle = run.omega * run.data.time + run.tray_phase
        tray = run.data.xpos[run.tray]
        body_pos = run.data.xpos[run.blocks] - [tray[0], tray[1], TABLE_Z]
        body_pos[:, :2] /= BLOCK_WIDTH
        body_pos[:, 2] = (body_pos[:, 2] - (np.arange(BLOCK_COUNT) + 0.5) * BLOCK_WIDTH) / 0.10
        velocity = run.data.qvel[self.dofs].copy()
        velocity[:, 1] -= run.tray_velocity
        tilt = run.data.xmat[run.blocks].reshape(BLOCK_COUNT, 3, 3)[:, :2, 2] / 0.20
        clock = [np.sin(angle), np.cos(angle), (run.period - 1.44) / 0.10, run.tray_velocity / 0.50]
        body = np.column_stack((body_pos, velocity[:, :3] / 0.25, tilt, velocity[:, 3:5] / 3.0))
        return np.clip(np.r_[clock, body.ravel()], -10, 10).astype(np.float32)

    def step(self, action):
        if self.current is None:
            raise RuntimeError("reset() must precede a placement decision")
        action = np.clip(np.asarray(action, dtype=float), -1, 1)
        offset = action[:2] * OFFSET_LIMIT_M
        wait = (action[2] + 1.0) * self.run.period / 2
        layer = self.run.complete_placement(self.placement, xy_offset=offset, pre_descent_wait=wait)
        metrics = self.run.evaluate_stack([*self.current["layers"], layer])
        post = metrics["post_release_cycle"]
        impulse = layer["touch_impulse_ns_100ms"] or 0.0
        reward = (
            3.0 * metrics["success"] + 0.6 * post["fraction_all_layers_supported"]
            - 0.5 * min(post["max_block_tilt_deg"] / 90, 1.0)
            - 0.08 * min(impulse / 0.15, 2.0) - 0.10 * wait / self.run.period
        )
        self.current = None
        info = {
            "metrics": metrics, "success": metrics["success"],
            "offset_xy_mm": (offset * 1000).tolist(), "pre_descent_wait_s": float(wait),
        }
        return self._observation(), float(reward), True, False, info


def prepare(config):
    cases = [*TRAIN_CASES, *TEST_CASES]
    with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
        snapshots = list(pool.map(make_snapshot, cases))
    model = LandingEnv(snapshots)
    model.reset(options={"case_index": 0})
    fixed = model.step(FIXED_ACTION)[-1]["metrics"]
    reference = StackRun(cases[0][0], tray_phase=np.radians(cases[0][1])).run()
    if fixed != reference:
        raise RuntimeError("Zero-residual rollout from the saved state disagrees with the full script")
    write_json(Path(config["bank"]), {"train": snapshots[:len(TRAIN_CASES)], "test": snapshots[len(TRAIN_CASES):]})
    print(f"Prepared {len(TRAIN_CASES)} train and {len(TEST_CASES)} paired test states; replay verified.", flush=True)


def train(config):
    from stable_baselines3 import SAC, TD3
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.vec_env import SubprocVecEnv
    import torch

    torch.set_num_threads(1)
    snapshots = json.loads(Path(config["bank"]).read_text())["train"]
    workers = config["workers"]
    steps = config["steps"]
    if steps <= 128 or steps % workers:
        raise ValueError("Training steps must exceed warmup and be divisible by the environment count")

    class Trace(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rewards = []
            self.successes = []
            self.history = []

        def _on_step(self):
            self.rewards.extend(float(x) for x in self.locals["rewards"])
            self.successes.extend(int(info["success"]) for info in self.locals["infos"])
            if len(self.rewards) >= 128 or self.num_timesteps == steps:
                record = {"decisions": self.num_timesteps, "train_successes": sum(self.successes),
                          "samples": len(self.rewards), "mean_reward": float(np.mean(self.rewards))}
                self.history.append(record)
                write_json(Path("progress.json"), self.history)
                print(record, flush=True)
                self.rewards.clear()
                self.successes.clear()
            return True

    env = SubprocVecEnv([lambda: LandingEnv(snapshots) for _ in range(workers)], start_method="fork")
    try:
        kwargs = dict(policy="MlpPolicy", env=env, learning_rate=3e-4, batch_size=128,
                      learning_starts=128, buffer_size=steps + 128,
                      train_freq=(1, "step"), gradient_steps=workers, policy_kwargs={"net_arch": [128, 128]},
                      seed=config["seed"], device="cpu")
        if config["algorithm"] == "sac":
            model = SAC(**kwargs, ent_coef="auto_0.1")
        elif config["algorithm"] == "td3":
            model = TD3(**kwargs, action_noise=NormalActionNoise(np.zeros(3), np.full(3, 0.1)))
        else:
            raise ValueError("Expected sac or td3")
        trace = Trace()
        model.learn(steps, callback=trace)
        model.save("policy.zip")
        if model._n_updates < 1:
            raise RuntimeError("Training completed without updating the networks")
        write_json(Path("training.json"), {
            "algorithm": config["algorithm"], "seed": config["seed"],
            "decisions": model.num_timesteps, "updates": model._n_updates,
            "workers": workers, "training_case_count": len(snapshots), "history": trace.history,
        })
    finally:
        env.close()


def test_case(item):
    snapshot, choices = item
    env = LandingEnv([snapshot])
    cases = {}
    for method, action in choices.items():
        env.reset()
        _, reward, _, _, info = env.step(action)
        cases[method] = {**info, "reward": reward}
    period = snapshot["period_s"]
    phase_deg = snapshot["start_phase_deg"]
    phase = np.radians(phase_deg) + 2 * np.pi * cases["timed"]["metrics"]["task_finished_s"] / period
    preparation = aligned_tower_check(period, phase)
    return {"period_s": period, "start_phase_deg": phase_deg, "prepared_tower": preparation, "methods": cases}


def evaluate(config):
    from stable_baselines3 import SAC, TD3
    import torch

    torch.set_num_threads(1)
    bank = json.loads(Path(config["bank"]).read_text())
    train_keys = {(s["period_s"], s["start_phase_deg"]) for s in bank["train"]}
    test_keys = {(s["period_s"], s["start_phase_deg"]) for s in bank["test"]}
    if train_keys & test_keys:
        raise ValueError("Training and evaluation cases overlap")
    states = bank["test"]
    env = LandingEnv(states)
    observations = np.stack([env.reset(options={"case_index": i})[0] for i in range(len(states))])
    rng = np.random.default_rng(config["random_seed"])
    choices = {"timed": np.tile(FIXED_ACTION, (len(states), 1)),
               "random": rng.uniform(-1, 1, (len(states), 3))}
    metadata = {}
    for key, folder in config["models"].items():
        algorithm = {"sac": SAC, "td3": TD3}[key]
        path = Path(folder)
        model = algorithm.load(path / "policy.zip", device="cpu")
        choices[key] = model.predict(observations, deterministic=True)[0]
        metadata[key] = json.loads((path / "training.json").read_text())
    items = [(state, {k: v[i] for k, v in choices.items()}) for i, state in enumerate(states)]
    with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
        results = list(pool.map(test_case, items))
    summary = {
        "train_cases": sorted(train_keys), "test_cases": sorted(test_keys),
        "offset_limit_mm": OFFSET_LIMIT_M * 1000,
        "max_wait_periods": 1, "random_seed": config["random_seed"],
        "training": metadata, "cases": results,
    }
    write_json(Path(config["output"]), summary)
    for key in choices:
        print(key, sum(r["methods"][key]["success"] for r in results), "/", len(results), flush=True)


def main():
    config = json.loads(Path("config.json").read_text(encoding="utf-8"))
    {"prepare": prepare, "train": train, "evaluate": evaluate}[config["stage"]](config)


if __name__ == "__main__":
    main()
