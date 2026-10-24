"""Inspect condition transitions and source times in a local recording."""
import argparse
from pathlib import Path
import numpy as np
from pulsepoint.data import RawSource


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--subject", default="S2")
    args = parser.parse_args()
    data = RawSource(args.raw_root).subject(f"{args.subject}/{args.subject}.pkl")
    labels = np.asarray(data["label"]).reshape(-1)
    boundaries = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1])+1, len(labels)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        print(f"Condition {labels[start]}: [{start/700:.3f}, {end/700:.3f}) seconds; {end-start} samples")


if __name__ == "__main__":
    main()
