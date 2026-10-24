"""Train a versioned calibrated bundle without replacing the original model."""
import argparse
from pathlib import Path
from threadpoolctl import threadpool_limits
from pulsepoint.data import BASE_DIR, load_dataset, binary_rows, RawSource
from pulsepoint.config import LEGACY_FEATURES
from pulsepoint.training import build_bundle, new_run, save_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=BASE_DIR / "wesad_features.csv")
    parser.add_argument("--output", type=Path, help="New run directory; defaults to a unique artifacts/ run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-root", type=Path, help="Optional waveform access for development-only corruption tuning")
    parser.add_argument("--families", nargs="+", choices=["logistic", "svm", "forest"], default=["logistic", "svm", "forest"])
    args = parser.parse_args()
    if args.output and (args.output / "model.joblib").exists():
        parser.error("Refusing to replace an existing model bundle; choose a new --output")
    frame = load_dataset(args.data, min_subjects=6)
    if args.raw_root and not {"source_file", "start_sample", "end_sample", "sampling_rate"}.issubset(frame):
        parser.error("--raw-root requires a dataset produced by prepare_dataset.py")
    frame = binary_rows(frame, LEGACY_FEATURES if "heart_rate" not in frame.columns else [])
    with threadpool_limits(limits=1):
        bundle = build_bundle(frame, args.seed, tuple(args.families), raw_source=RawSource(args.raw_root) if args.raw_root else None)
    output = args.output or new_run(BASE_DIR / "artifacts", "train")
    path = save_bundle(bundle, output, args.data)
    print(f"Saved {path}\nArtifact ID: {bundle['artifact_id']}\nMultimodal: {bundle['multimodal']}")
    if not bundle["multimodal"]:
        print("ECG-summary mode: waveform quality, alternate experts and event timing require regenerated raw data.")


if __name__ == "__main__":
    main()
