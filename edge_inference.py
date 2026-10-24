"""Replay waveform windows or stored features through the shared local runtime."""
import argparse
import json
from pathlib import Path
from pulsepoint.data import BASE_DIR, RawSource, load_dataset
from pulsepoint.replay import Runtime, chronological, source_interval
from pulsepoint.training import load_bundle
from pulsepoint.corruption import SCENARIOS, corrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=BASE_DIR / "wesad_features.csv")
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--subject")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="clean")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.scenario != "clean" and not args.raw_root:
        parser.error("Waveform corruption requires --raw-root")
    if args.output and args.output.exists():
        parser.error("Refusing to overwrite --output")
    runtime = Runtime(load_bundle(args.bundle))
    frame = load_dataset(args.data, min_subjects=1)
    if args.subject:
        frame = frame[frame.subject == args.subject]
    if frame.empty:
        parser.error("No matching rows")
    source = RawSource(args.raw_root) if args.raw_root else None
    rows = []
    for row in chronological(frame).head(args.limit).to_dict("records"):
        if source:
            signals, rate = source.window(row)
            result = runtime.predict(signals=corrupt(signals, args.scenario, row["row_id"]), rate=rate, interval=source_interval(row))
        else:
            result = runtime.predict(features=row, interval=source_interval(row))
        rows.append({"row_id": row["row_id"], "subject": row["subject"], **result})
    text = "\n".join(json.dumps(row) for row in rows) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
