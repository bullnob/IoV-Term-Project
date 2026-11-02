import csv
import json
from pathlib import Path

from .evaluation import evaluate, TEST_SEEDS, paired_difference
from .learner import Learner


def evaluate_run(run, seeds=TEST_SEEDS):
    run = Path(run)
    learner = Learner.load(run / "best.pt")
    config = learner.config.environment
    results = {"trained": evaluate(config, "trained", seeds, learner)}
    for name, checkpoint in (("untrained", "untrained.pt"), ("warmstart", "warmstart.pt"), ("final", "final.pt")):
        model = Learner.load(run / checkpoint)
        results[name] = evaluate(config, "trained", seeds, model)
    for name in ("random", "local", "mec_first", "matching"):
        results[name] = evaluate(config, name, seeds)
    comparisons = {name: {metric: paired_difference(results["trained"], results[name], metric)
                          for metric in ("reward", "energy_j_per_completed_mbit", "deadline_success")}
                   for name in results if name != "trained"}
    payload = {"seed_split": "held-out test traffic; never used for checkpoint selection",
               "seeds": list(seeds), "results": results, "paired_comparisons": comparisons}
    (run / "evaluation.json").write_text(json.dumps(payload, indent=2))
    with (run / "evaluation.csv").open("w", newline="") as file:
        fields = ["policy", "reward", "energy_j_per_slot", "energy_j_per_completed_mbit", "deadline_success", "queue_mbit", "queue_growth_mbit", "latency_s"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for name, result in results.items():
            writer.writerow({"policy": name, **{key: result["summary"][key]["mean"] for key in fields[1:]}})
    print((run / "evaluation.csv").read_text(), flush=True)
    plot_run(run, payload)
    return payload


def plot_run(run, payload=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run = Path(run)
    payload = payload or json.loads((run / "evaluation.json").read_text())
    history = list(csv.DictReader((run / "history.csv").open()))
    validation = json.loads((run / "validation.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")
    axes[0].plot([int(x["episode"]) for x in history], [float(x["reward"]) for x in history], alpha=0.5, label="Exploratory training")
    vals = [x for x in validation if x["stage"] == "RL"]
    axes[0].plot([x["episode"] for x in vals], [x["result"]["summary"]["reward"]["mean"] for x in vals], "o-", label="Validation (no noise)")
    axes[0].set(xlabel="RL episode", ylabel="Reward", title="Learning progress")
    axes[0].legend(fontsize=8)
    names = ["untrained", "warmstart", "trained", "random", "local", "mec_first", "matching"]
    for ax, metric, title in [(axes[1], "energy_j_per_completed_mbit", "Energy / completed work (lower is better)"),
                              (axes[2], "deadline_success", "Within service budget / attempted packets")]:
        stats = [payload["results"][name]["summary"][metric] for name in names]
        ax.bar(names, [s["mean"] for s in stats], yerr=[s["ci95"] for s in stats], capsize=3,
               color=["#8b97a8", "#b299e5", "#15aa94", "#e6b35c", "#e18380", "#6794d5", "#334a67"])
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=45)
    axes[1].set_ylabel("J / Mbit")
    axes[2].set_ylabel("Fraction")
    fig.suptitle("Simulation prototype • held-out traffic • 95% approximate confidence intervals")
    fig.savefig(run / "results.png", dpi=170)
    fig.savefig(run / "results.svg")
    plt.close(fig)

