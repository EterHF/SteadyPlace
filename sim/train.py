"""Run one SAC seed from the current directory's config.json."""

import json
from pathlib import Path
from time import monotonic

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
import torch

from sim.carry_place import CarryPlaceEnv
from sim.coupled_place import CoupledCarryPlaceEnv


def save_policy(model, name):
    path = Path(name)
    temporary = path.with_suffix(".tmp.zip")
    model.save(temporary)
    temporary.replace(path)


def main():
    config = json.loads(Path("config.json").read_text(encoding="utf-8"))
    steps = config["steps"]
    checkpoints = sorted(set(config.get("checkpoint_steps", [])))
    if any(step <= 0 or step >= steps for step in checkpoints):
        raise ValueError("Checkpoint steps must be positive and earlier than the final step")
    torch.set_num_threads(1)
    device = config["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a GPU")
    coupled = config.get("coupled_approach", False)
    options = dict(force_feedback=config["force_feedback"], randomize=True,
                   target_width_m=config.get("target_width_m", .064 if coupled else None))
    task = CoupledCarryPlaceEnv(**options) if coupled else CarryPlaceEnv(
        observer_feedback=config.get("observer_feedback", False), **options)
    monitor_file = "monitor.csv" if config.get("record_episodes", False) else None
    env = Monitor(task, filename=monitor_file, info_keywords=("status", "is_success") if monitor_file else ())
    try:
        # For a scored deadline, bootstrapping from the final observation is invalid.
        replay_options = {"handle_timeout_termination": False} if config.get("terminal_deadline", False) else None
        model = SAC("MlpPolicy", env, learning_rate=3e-4, policy_kwargs={"net_arch": [128, 128]},
                    batch_size=256, buffer_size=config.get("buffer_size", 200_000),
                    learning_starts=config.get("learning_starts", 5_000), seed=config["seed"], device=device,
                    replay_buffer_kwargs=replay_options, verbose=config.get("verbose", 0))
        start = monotonic()
        for step in [*checkpoints, steps]:
            model.learn(total_timesteps=step - model.num_timesteps, reset_num_timesteps=False)
            if step != steps:
                save_policy(model, f"checkpoint_{step}.zip")
                print(f"Checkpoint at {step} steps, {model._n_updates} updates, {monotonic() - start:.1f} s", flush=True)
        if config.get("save_policy", True):
            save_policy(model, "policy.zip")
        report = {"device": str(model.device), "policy_steps": model.num_timesteps,
                  "updates": model._n_updates,
                  "actor_loss": model.logger.name_to_value.get("train/actor_loss"),
                  "critic_loss": model.logger.name_to_value.get("train/critic_loss"),
                  "elapsed_s": monotonic() - start,
                  "argv": Path("/proc/self/cmdline").read_bytes().decode().strip("\x00").split("\x00")}
        Path("training.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    finally:
        env.close()


if __name__ == "__main__":
    main()
