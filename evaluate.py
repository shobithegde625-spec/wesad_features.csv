"""Run subject-held-out baselines, nested routing ablations and waveform failures."""
import argparse
from pathlib import Path
from pulsepoint.data import BASE_DIR, load_dataset
from pulsepoint.evaluation import evaluate
from pulsepoint.training import new_run
from pulsepoint.corruption import SCENARIOS


def main(default_protocol="loso", default_baseline_only=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=BASE_DIR / "wesad_features.csv")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--protocol", choices=["loso", "group5"], default=default_protocol)
    parser.add_argument("--baseline-only", action="store_true", default=default_baseline_only)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--families", nargs="+", choices=["logistic", "svm", "forest", "historical_forest"], default=["logistic", "svm", "forest"])
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--replay", type=Path, help="Optional chronological replay CSV; fit only on --data")
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=["clean"])
    args = parser.parse_args()
    frame = load_dataset(args.data)
    replay = load_dataset(args.replay) if args.replay else None
    if args.output and args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be empty or new")
    if args.raw_root and not args.raw_root.exists():
        parser.error("--raw-root does not exist")
    if tuple(args.scenarios) != ("clean",) and not args.raw_root:
        parser.error("Corruption scenarios require --raw-root")
    if args.baseline_only and tuple(args.scenarios) != ("clean",):
        parser.error("Corruption evaluation requires the routing experiment, not --baseline-only")
    output = args.output or new_run(BASE_DIR / "reports", args.protocol)
    evaluate(frame, args.data, output, protocol=args.protocol, seed=args.seed, families=tuple(args.families),
             baseline_only=args.baseline_only, raw_root=args.raw_root, scenarios=tuple(args.scenarios), replay_frame=replay,
             replay_path=args.replay)
    print(f"Saved report: {output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
