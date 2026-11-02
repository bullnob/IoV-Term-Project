from dataclasses import asdict
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch

from .config import Config
from .environment import OffloadingEnv
from .evaluation import evaluate, VALIDATION_SEEDS
from .learner import Learner
from .replay import ReplayBuffer, save_transitions


def train(config: Config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists():
        raise FileExistsError(f"Use a new run directory; {output} already contains a run")
    config.save(output / "config.json")
    c, t = config.environment, config.training
    learner = Learner(config)
    replay = ReplayBuffer(t.replay_size, t.seed + 1)
    history, assessments = [], []
    best_reward = -float("inf")
    start = time.monotonic()

    def assess(stage, episode_number):
        nonlocal best_reward
        result = evaluate(c, "trained", VALIDATION_SEEDS, learner)
        score = result["summary"]["reward"]["mean"]
        assessments.append({"stage": stage, "episode": episode_number, "result": result})
        if score > best_reward:
            best_reward = score
            learner.save(output / "best.pt", {"stage": stage, "episode": episode_number,
                                             "validation_reward": score, "selection": "validation only"})
        print(f"{stage:12s} episode={episode_number:3d} validation_reward={score:.4f} "
              f"deadline_success={result['summary']['deadline_success']['mean']:.3f}", flush=True)
        return score

    learner.save(output / "untrained.pt", {"stage": "untrained"})
    assess("untrained", 0)
    # Algorithm 1 supplies actual (s,a,r,s') rollouts, not static reward labels.
    env = OffloadingEnv(c)
    rollout_index = 0
    state = env.reset(10_000 + t.seed * 1_000)
    warmup_rows = []
    for _ in range(t.warmup_steps):
        action = env.matching_action()
        next_state, reward, terminated, truncated, _ = env.step(action)
        replay.add(state, action, reward, next_state, terminated)
        warmup_rows.append(replay.data[(replay.cursor - 1) % len(replay.data)])
        state = next_state
        if truncated:
            rollout_index += 1
            state = env.reset(10_000 + t.seed * 1_000 + rollout_index)
    if len(replay) < t.batch_size:
        raise ValueError("warmup_steps must be at least batch_size")
    save_transitions(output / "matching_data.npz", warmup_rows)
    for _ in range(t.imitation_steps):
        learner.imitate(replay.sample(t.batch_size))
    learner.sync_targets()
    learner.save(output / "warmstart.pt", {"stage": "matching imitation", "steps": t.imitation_steps})
    assess("warmstart", 0)

    steps = 0
    for episode_number in range(1, t.episodes + 1):
        state = env.reset(40_000 + t.seed * 1_000 + episode_number)
        rewards, energy, successes, queues, updates = [], [], [], [], []
        noise = t.noise_end + (t.noise_start - t.noise_end) * max(0, 1 - episode_number / (0.8 * t.episodes))
        for _ in range(c.horizon):
            action = learner.act(state, noise)
            next_state, reward, terminated, truncated, info = env.step(action)
            replay.add(state, action, reward, next_state, terminated)
            state = next_state
            steps += 1
            rewards.append(info["reward"])
            energy.append(info["energy_j"])
            successes.append(info["deadline_success"])
            queues.append(info["queue_mbit"])
            if steps % t.update_interval == 0:
                updates.append(learner.update(replay.sample(t.batch_size)))
            if truncated or terminated:
                break
        row = {"episode": episode_number, "reward": float(np.mean(rewards)),
               "energy_j": float(np.mean(energy)), "deadline_success": float(np.mean(successes)),
               "queue_mbit": float(np.mean(queues)), "noise": noise,
               "critic_loss": float(np.mean([x["critic_loss"] for x in updates])) if updates else 0.0,
               "actor_loss": float(np.mean([x["actor_loss"] for x in updates])) if updates else 0.0,
               "actor_gradient_norm": float(np.mean([x["actor_gradient_norm"] for x in updates])) if updates else 0.0}
        history.append(row)
        # Persist after every episode so an interrupted run retains its evidence.
        with (output / "history.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        if episode_number % t.eval_interval == 0 or episode_number == t.episodes:
            assess("RL", episode_number)
            (output / "validation.json").write_text(json.dumps(assessments, indent=2))
    learner.save(output / "final.pt", {"stage": "RL final", "episode": t.episodes})
    manifest = {"config": asdict(config), "elapsed_seconds": time.monotonic() - start,
                "rl_steps": steps, "critic_updates_per_agent": learner.updates,
                "actor_updates_per_agent": learner.updates, "imitation_updates_per_agent": t.imitation_steps,
                "matching_transitions": t.warmup_steps, "torch_version": str(torch.__version__),
                "validation_seeds": list(VALIDATION_SEEDS), "test_seeds_used_in_training": []}
    (output / "training_manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "validation.json").write_text(json.dumps(assessments, indent=2))
    return output

