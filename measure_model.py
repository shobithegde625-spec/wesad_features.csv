"""Inspect provenance and complete bundle footprint."""
import argparse
from pathlib import Path
from time import perf_counter
from pulsepoint.training import load_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    bundle = load_bundle(args.bundle)
    print(f"Artifact: {bundle['artifact_id']}\nSize: {args.bundle.stat().st_size/1024:.2f} KiB\nLoad: {(perf_counter()-started)*1000:.3f} ms")
    print("Experts:", ", ".join(bundle["experts"]))
    print("Subject roles:", bundle["manifest"])


if __name__ == "__main__":
    main()
