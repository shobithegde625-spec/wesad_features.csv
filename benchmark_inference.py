"""Benchmark real processing branches with reproducible representative windows."""
import argparse
from pathlib import Path
from pulsepoint.data import BASE_DIR
from pulsepoint.benchmark import benchmark
from pulsepoint.corruption import SCENARIOS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=BASE_DIR / "wesad_features.csv")
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New directory")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=["clean"])
    args = parser.parse_args()
    try:
        report = benchmark(args.bundle, args.data, args.output, args.raw_root, args.limit, args.repeats, tuple(args.scenarios))
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(f"Saved {args.output / 'benchmark.json'}; input: {report['input_kind']}")


if __name__ == "__main__":
    main()
