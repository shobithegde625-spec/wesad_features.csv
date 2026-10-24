"""Dataset validation, provenance, and read-only raw waveform adapters."""
from pathlib import Path, PurePosixPath
import hashlib
import importlib.metadata
import json
import pickle
import platform
import sys
import zipfile
import threading

import numpy as np
import pandas as pd
from .config import LEGACY_FEATURES

BASE_DIR = Path(__file__).resolve().parent.parent


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment():
    packages = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "streamlit", "threadpoolctl", "psutil"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(), "packages": packages}


def json_safe(value):
    """Use standard JSON null for undefined metrics and estimator NaN sentinels."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, data):
    Path(path).write_text(json.dumps(json_safe(data), indent=2, allow_nan=False, default=str) + "\n")


def load_dataset(path, features=None, min_subjects=2):
    frame = pd.read_csv(path)
    required = list(features or LEGACY_FEATURES) + ["label", "subject"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    if frame.empty or frame.subject.isna().any():
        raise ValueError("Dataset is empty or has missing subject identifiers")
    frame["subject"] = frame.subject.astype(str)
    frame["label"] = pd.to_numeric(frame.label, errors="raise")
    # Replay may include unsupported conditions, represented by a missing binary label.
    invalid = frame.label.notna() & ~frame.label.isin([0, 1])
    if invalid.any():
        raise ValueError("Binary targets must be 0 or 1; unsupported replay intervals use a missing label")
    for name in set(required) - {"subject", "label"}:
        frame[name] = pd.to_numeric(frame[name], errors="raise").replace([np.inf, -np.inf], np.nan)
    for name in ("start_seconds", "end_seconds", "start_sample", "end_sample", "sampling_rate"):
        if name in frame:
            frame[name] = pd.to_numeric(frame[name], errors="raise")
            if not np.isfinite(frame[name].to_numpy(dtype=float)).all():
                raise ValueError(f"Source metadata {name} must be finite")
    if {"start_seconds", "end_seconds"}.issubset(frame):
        if (frame.start_seconds < 0).any() or (frame.end_seconds <= frame.start_seconds).any():
            raise ValueError("Source intervals require 0 <= start_seconds < end_seconds")
    if "row_id" not in frame:
        frame["row_id"] = np.arange(len(frame))
    if frame.row_id.isna().any() or frame.row_id.duplicated().any():
        raise ValueError("row_id must be unique and nonmissing")
    if frame.subject.nunique() < min_subjects:
        raise ValueError(f"Need at least {min_subjects} distinct subjects")
    return frame


def binary_rows(frame, features):
    out = frame.loc[frame.label.isin([0, 1])].copy()
    if out.empty or set(out.label.unique()) != {0, 1}:
        raise ValueError("Both baseline and stress targets are required")
    # Legacy models cannot use missing summaries; multimodal rows retain modality failures.
    if features == LEGACY_FEATURES:
        valid = np.isfinite(out[features].to_numpy(dtype=float)).all(axis=1)
        if not valid.all():
            raise ValueError("Legacy feature dataset contains missing or nonfinite values")
    return out.reset_index(drop=True)


class RawSource:
    """Trusted WESAD pickles in a directory or ZIP; cache only required channels."""
    def __init__(self, root, retain_all=False):
        self.root = Path(root).expanduser().resolve()
        self.retain_all = retain_all
        self.archive = self.root.is_file() and zipfile.is_zipfile(self.root)
        if not self.root.is_dir() and not self.archive:
            raise FileNotFoundError(f"Raw WESAD directory or ZIP not found: {self.root}")
        self.members = {}
        self._lock = threading.RLock()
        self._cached_name = self._cached_data = None
        if self.archive:
            with zipfile.ZipFile(self.root) as archive:
                for member in archive.infolist():
                    parts = PurePosixPath(member.filename).parts
                    if member.is_dir() or "__MACOSX" in parts:
                        continue
                    if len(parts) >= 2 and parts[-1] == parts[-2] + ".pkl" and parts[-2].startswith("S"):
                        relative = "/".join(parts[-2:])
                        if relative in self.members:
                            raise ValueError(f"Duplicate synchronized recording in archive: {relative}")
                        self.members[relative] = member.filename

    def subject(self, relative_path):
        # Streamlit may share this source between browser sessions.
        with self._lock:
            return self._load_subject(relative_path)

    def _load_subject(self, relative_path):
        relative = PurePosixPath(relative_path)
        if relative.is_absolute() or ".." in relative.parts or "\\" in relative_path:
            raise ValueError("Raw source path escapes the configured data directory")
        if relative_path == self._cached_name:
            return self._cached_data
        # Evict before loading the next participant to avoid a two-recording cache peak.
        self._cached_name = self._cached_data = None
        if self.archive:
            if relative_path not in self.members:
                raise FileNotFoundError(f"Recording not found in archive: {relative_path}")
            with zipfile.ZipFile(self.root) as archive, archive.open(self.members[relative_path]) as handle:
                data = pickle.load(handle, encoding="latin1")
                # Finish the entry so ZipExtFile verifies its CRC, including any trailing bytes.
                while handle.read(1024*1024):
                    pass
        else:
            path = (self.root / relative_path).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("Raw source path escapes the configured data directory")
            with path.open("rb") as handle:
                data = pickle.load(handle, encoding="latin1")
        if not self.retain_all:
            chest = data["signal"]["chest"]
            data = {"label": data["label"], "signal": {"chest": {
                name: chest[name] for name in ("ECG", "EDA", "Resp", "ACC") if name in chest}}}
        self._cached_name, self._cached_data = relative_path, data
        return data

    def recordings(self):
        if self.archive:
            return sorted(self.members)
        return sorted(str(path.relative_to(self.root)) for path in self.root.glob("S*/S*.pkl")
                      if path.stem == path.parent.name)

    def recording_provenance(self, relative_path):
        if not self.archive:
            path = self.root / relative_path
            return {"sha256": file_hash(path), "bytes": path.stat().st_size}
        with zipfile.ZipFile(self.root) as archive:
            member = archive.getinfo(self.members[relative_path])
        # The verified parent archive hash plus entry CRC and byte count identify
        # each recording without a second 13 GiB decompression pass for hashing.
        return {"archive_member": member.filename, "archive_crc32": f"{member.CRC:08x}",
                "bytes": member.file_size, "compressed_bytes": member.compress_size}

    def window(self, row):
        data = self.subject(str(row["source_file"]))
        chest = data["signal"]["chest"]
        labels = np.asarray(data["label"]).reshape(-1)
        rate = float(row["sampling_rate"])
        if rate != 700:
            raise ValueError("This WESAD chest adapter supports the documented 700 Hz rate only")
        start, end = int(row["start_sample"]), int(row["end_sample"])
        if not 0 <= start < end <= len(labels):
            raise ValueError("Invalid waveform interval")
        signals = {}
        for name, key in [("ecg", "ECG"), ("eda", "EDA"), ("resp", "Resp"), ("acc", "ACC")]:
            if key in chest:
                values = np.asarray(chest[key])
                if len(values) != len(labels):
                    raise ValueError(f"{key} does not align with the 700 Hz label stream")
                signals[name] = np.asarray(values[start:end], dtype=float)
        return signals, rate
