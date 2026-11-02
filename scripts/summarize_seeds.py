"""Aggregate separate training runs without treating repeated traffic as independent."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--output", type=Path, default=Path("artifacts/seed_summary.json"))
    args = p.parse_args()
    rows = []
    for run in args.runs:
        manifest = json.loads((run / "training_manifest.json").read_text())
        result = json.loads((run / "evaluation.json").read_text())
        rows.append({"training_seed": manifest["config"]["training"]["seed"],
                     "training_seconds": manifest["elapsed_seconds"],
                     "policies": {name: {metric: value["mean"] for metric, value in data["summary"].items()}
                                  for name, data in result["results"].items()}})
    means = {}
    for policy in rows[0]["policies"]:
        means[policy] = {}
        for metric in rows[0]["policies"][policy]:
            values = [r["policies"][policy][metric] for r in rows]
            means[policy][metric] = {"mean_over_training_seeds": statistics.mean(values),
                                    "min": min(values), "max": max(values)}
    payload = {"training_seeds": [r["training_seed"] for r in rows], "runs": rows,
               "summary": means, "note": "Each run uses the same 20 paired test traffic seeds. Ranges are over training seeds; these are not 60 independent traffic scenarios."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
