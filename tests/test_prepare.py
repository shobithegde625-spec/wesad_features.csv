import pickle
import numpy as np
import pandas as pd
import pytest
from pulsepoint.prepare import prepare
from pulsepoint.data import RawSource


def test_raw_adapter_preserves_interval_provenance(synthetic_signals, tmp_path):
    # 30 s baseline, 5 s unsupported, then 30 s stress. Pure windows start at segment boundaries.
    root = tmp_path / "raw"
    subject = root / "S2"
    subject.mkdir(parents=True)
    rate = 700
    chest = {}
    for name, key in [("ecg", "ECG"), ("eda", "EDA"), ("resp", "Resp"), ("acc", "ACC")]:
        values = synthetic_signals[name]
        chest[key] = np.concatenate([values, values[:5*rate], values], axis=0)
    labels = np.r_[np.ones(30*rate), np.zeros(5*rate), np.full(30*rate, 2)]
    with (subject / "S2.pkl").open("wb") as handle:
        pickle.dump({"signal": {"chest": chest}, "label": labels}, handle)
    output = prepare(root, tmp_path / "prepared", seconds=30)
    classification = pd.read_csv(output / "classification.csv")
    replay = pd.read_csv(output / "replay.csv")
    assert classification.start_seconds.tolist() == [0, 35]
    assert classification.label.tolist() == [0, 1]
    assert replay.start_seconds.tolist() == [0, 30]
    assert pd.isna(replay.iloc[1].label)
    assert replay.iloc[1].original_condition == -1
    signals, fs = RawSource(root).window(classification.iloc[1].to_dict())
    assert fs == rate and len(signals["ecg"]) == 30*rate
    with pytest.raises(FileExistsError):
        prepare(root, output)


def test_missing_root_does_not_create_output(tmp_path):
    output = tmp_path / "output"
    with pytest.raises(FileNotFoundError):
        prepare(tmp_path / "absent", output)
    assert not output.exists()


def test_raw_source_cannot_escape_root(tmp_path):
    source = RawSource(tmp_path)
    with pytest.raises(ValueError):
        source.subject("../outside.pkl")


def test_zip_source_matches_directory_and_bounds_memory(synthetic_signals, tmp_path):
    import zipfile
    raw = tmp_path / "raw"
    subject = raw / "S2"
    subject.mkdir(parents=True)
    content = {"label": np.ones(30*700), "signal": {
        "chest": {"ECG": synthetic_signals["ecg"], "EDA": synthetic_signals["eda"],
                  "Resp": synthetic_signals["resp"], "ACC": synthetic_signals["acc"], "EMG": np.ones(30*700)},
        "wrist": {"BVP": np.ones(100)}}}
    payload = pickle.dumps(content)
    (subject / "S2.pkl").write_bytes(payload)
    zip_path = tmp_path / "WESAD.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("WESAD/S2/S2.pkl", payload)
    directory = RawSource(raw)
    compressed = RawSource(zip_path)
    assert directory.recordings() == compressed.recordings() == ["S2/S2.pkl"]
    row = {"source_file": "S2/S2.pkl", "sampling_rate": 700, "start_sample": 0, "end_sample": 30*700}
    a, rate = directory.window(row)
    b, _ = compressed.window(row)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
    assert "wrist" not in compressed.subject("S2/S2.pkl")["signal"]
    assert "EMG" not in compressed.subject("S2/S2.pkl")["signal"]["chest"]
    assert "wrist" in RawSource(zip_path, retain_all=True).subject("S2/S2.pkl")["signal"]
    assert compressed.recording_provenance("S2/S2.pkl")["bytes"] == len(payload)
    with pytest.raises(ValueError):
        compressed.subject("../outside.pkl")
    output = prepare(zip_path, tmp_path / "zip-prepared", seconds=30)
    assert len(pd.read_csv(output / "classification.csv")) == 1
