from dataclasses import replace
import numpy as np

from .environment import OffloadingEnv

POLICIES = ("trained", "untrained", "random", "local", "mec_first", "matching")
VALIDATION_SEEDS = tuple(range(200_000, 200_006))
TEST_SEEDS = tuple(range(300_000, 300_020))


def episode(config, policy, seed, learner=None, *, frames=False, workload=1.0):
    c = replace(config, arrival_rate=config.arrival_rate * workload)
    env = OffloadingEnv(c)
    state = env.reset(seed)
    policy_rng = np.random.default_rng(seed + 900_000)
    records = []
    for _ in range(c.horizon):
        if policy in ("trained", "untrained"):
            if learner is None:
                raise ValueError("A neural policy needs a checkpoint")
            action = learner.act(state)
        elif policy == "matching":
            action = env.matching_action()
        elif policy == "local":
            action = np.zeros((c.agents, c.subtasks), dtype=np.int64)
        elif policy == "mec_first":
            scores = np.zeros((c.agents, c.subtasks, 5))
            scores[:, :, 3] = 10
            action = env.matcher.solve(scores, state.rsu, state.available)
        elif policy == "random":
            scores = policy_rng.uniform(-1, 1, (c.agents, c.subtasks, 5))
            action = env.matcher.solve(scores, state.rsu, state.available)
        else:
            raise ValueError(f"Unknown policy {policy}")
        state, _, _, _, info = env.step(action)
        records.append(info)
    total = lambda key: sum(row[key] for row in records)
    metrics = {
        "seed": seed,
        "reward": total("reward") / c.horizon,
        "energy_j_per_slot": total("energy_j") / c.horizon,
        "energy_j_per_completed_mbit": total("energy_j") / max(total("served_mbit"), 1e-9),
        "deadline_success": total("completed") / max(total("attempted"), 1),
        "queue_mbit": total("queue_mbit") / c.horizon,
        "queue_growth_mbit": total("queue_growth_mbit") / c.horizon,
        "latency_s": total("latency_s") / c.horizon,
        "completed_subtasks": total("completed"),
        "attempted_subtasks": total("attempted"),
    }
    return {"metrics": metrics, **({"frames": records} if frames else {})}


def evaluate(config, policy, seeds, learner=None):
    rows = [episode(config, policy, seed, learner)["metrics"] for seed in seeds]
    summary = {}
    for key in rows[0]:
        if key == "seed":
            continue
        values = np.array([row[key] for row in rows], dtype=float)
        summary[key] = {"mean": float(values.mean()),
                        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                        "ci95": float(1.96 * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0}
    return {"episodes": rows, "summary": summary, "seed_count": len(rows)}


def paired_difference(candidate, baseline, key):
    a, b = candidate["episodes"], baseline["episodes"]
    if [x["seed"] for x in a] != [x["seed"] for x in b]:
        raise ValueError("Paired comparisons require the same ordered evaluation seeds")
    delta = np.array([x[key] - y[key] for x, y in zip(a, b, strict=True)])
    half = float(1.96 * delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else 0.0
    return {"mean_difference": float(delta.mean()), "ci95_low": float(delta.mean() - half),
            "ci95_high": float(delta.mean() + half), "interpretation": "candidate minus baseline; approximate normal CI over traffic seeds"}

