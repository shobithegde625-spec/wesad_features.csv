"""Inspect a trusted local WESAD recording without assuming a personal folder."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from pulsepoint.data import RawSource


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--subject", default="S2")
    args = parser.parse_args()
    data = RawSource(args.raw_root, retain_all=True).subject(f"{args.subject}/{args.subject}.pkl")
    print("Keys:", list(data))
    print("Labels:", dict(Counter(np.asarray(data['label']).reshape(-1))))
    for group, signals in data["signal"].items():
        print(group, {name: np.shape(value) for name, value in signals.items()})


if __name__ == "__main__":
    main()
