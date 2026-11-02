import argparse
from pathlib import Path

from .config import Config


def main():
    parser = argparse.ArgumentParser(description="Train, evaluate, and showcase graph-assisted offloading")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("train", help="Generate matching transitions, warm-start, then train with RL")
    p.add_argument("--config", type=Path)
    p.add_argument("--output", type=Path, default=Path("runs/demo"))
    p.add_argument("--seed", type=int)
    p.add_argument("--episodes", type=int)
    p.add_argument("--no-imitation", action="store_true", help="Disable the optional behavior-cloning warm-start")
    p = commands.add_parser("evaluate", help="Evaluate frozen checkpoints on held-out traffic")
    p.add_argument("--run", type=Path, default=Path("artifacts/demo"))
    p = commands.add_parser("demo", help="Start the interactive dashboard")
    p.add_argument("--run", type=Path, default=Path("artifacts/demo"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p = commands.add_parser("export", help="Export a self-contained, offline interactive HTML showcase")
    p.add_argument("--run", type=Path, default=Path("artifacts/demo"))
    p.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        from .train import train
        config = Config.load(args.config) if args.config else Config()
        if args.seed is not None:
            config.training.seed = args.seed
        if args.episodes is not None:
            config.training.episodes = args.episodes
        if args.no_imitation:
            config.training.imitation_steps = 0
        train(config, args.output)
    elif args.command == "evaluate":
        from .report import evaluate_run
        evaluate_run(args.run)
    else:
        from .demo import serve, export
        if args.command == "export":
            export(args.run, args.output)
        else:
            serve(args.run, args.host, args.port)


if __name__ == "__main__":
    main()
