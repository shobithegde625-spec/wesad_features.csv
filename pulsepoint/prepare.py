"""Generate separate pure-condition classification and chronological replay tables."""
from pathlib import Path
import numpy as np
import pandas as pd
from .config import EXPERT_FEATURES
from .data import RawSource, file_hash, write_json, environment
from .features import RawFeatures


def prepare(raw_root, output, seconds=60, stride=None):
    if seconds < 10 or (stride is not None and stride != seconds):
        raise ValueError("Initial protocol requires nonoverlapping windows >=10 seconds (stride = window)")
    source = RawSource(raw_root)
    recordings = source.recordings()
    if not recordings:
        raise FileNotFoundError(f"No S*/S*.pkl recordings found under {source.root}")
    output = Path(output)
    # Validate raw availability before making any output directory.
    output.mkdir(parents=True, exist_ok=False)
    rate, size = 700, int(700 * seconds)
    classification, replay, provenance = [], [], []
    for relative in recordings:
        subject = Path(relative).parent.name
        print(f"Loading {subject}...", flush=True)
        data = source.subject(relative)
        labels = np.asarray(data["label"]).reshape(-1)
        boundaries = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1])+1, len(labels)]
        provenance.append({"subject": subject, "source_file": relative, **source.recording_provenance(relative),
                           "samples": len(labels)})
        print(f"Processing {subject}", flush=True)
        def extract(start, end, segment, kind):
            observed = np.unique(labels[start:end])
            original = int(observed[0]) if len(observed) == 1 else -1
            row = {"row_id": f"{subject}-{kind}-{start}", "subject": subject, "session": subject,
                   "segment_id": segment, "source_file": relative, "sampling_rate": rate,
                   "start_sample": start, "end_sample": end, "start_seconds": start/rate,
                   "end_seconds": end/rate, "window_seconds": seconds, "stride_seconds": seconds,
                   "original_condition": original, "label": 0 if original == 1 else 1 if original == 2 else np.nan}
            signals, fs = source.window(row)
            row.update(RawFeatures(signals, fs).all_features())
            return row
        for segment, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if labels[start] in (1, 2):
                for pos in range(int(start), int(end)-size+1, size):
                    classification.append(extract(pos, pos+size, segment, "classification"))
        for start in range(0, len(labels)-size+1, size):
            segment = int(np.searchsorted(boundaries, start, side="right")-1)
            replay.append(extract(start, start+size, segment, "replay"))
        del data, labels
    if not classification:
        raise ValueError("No complete pure-condition baseline/stress windows were found")
    all_features = EXPERT_FEATURES["fusion"]
    for name, rows in [("classification", classification), ("replay", replay)]:
        frame = pd.DataFrame(rows)
        for feature in all_features:
            if feature not in frame:
                frame[feature] = np.nan
        frame.to_csv(output / f"{name}.csv", index=False)
    write_json(output / "dataset.json", {"schema_version": 1, "sampling_rate": rate,
               "window_seconds": seconds, "stride_seconds": seconds, "raw_root": str(source.root),
               "sources": provenance, "environment": environment(),
               "raw_archive_sha256": file_hash(source.root) if source.archive else None,
               "classification_rows": len(classification), "replay_rows": len(replay),
               "unsupported_replay_label": None,
               "quality_note": "Fixed conservative heuristics, not clinically validated quality labels"})
    return output
