"""Extract multimodal classification and replay tables from trusted local WESAD files."""
import argparse
from pathlib import Path
from pulsepoint.prepare import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing data is never overwritten")
    parser.add_argument("--window-seconds", type=int, default=60)
    parser.add_argument("--stride-seconds", type=int)
    args = parser.parse_args()
    try:
        output = prepare(args.raw_root, args.output, args.window_seconds, args.stride_seconds)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))
    print(f"Saved classification.csv, replay.csv and dataset.json in {output}")


if __name__ == "__main__":
    main()
