"""Regenerate plots from a versioned evaluation run."""
import argparse
from pathlib import Path
import pandas as pd
from pulsepoint.reporting import plot_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    plot_report(pd.read_csv(args.report / "predictions.csv"), args.report)
    print(args.report / "evaluation.png")


if __name__ == "__main__":
    main()
