"""Exercise the complete waveform pipeline; these signals are NOT WESAD evidence."""
import pickle
import numpy as np
import pandas as pd
from pulsepoint.prepare import prepare
from pulsepoint.data import RawSource, load_dataset
from pulsepoint.training import build_bundle, save_bundle, load_bundle
from pulsepoint.replay import Runtime, source_interval
from pulsepoint.corruption import corrupt
from pulsepoint.evaluation import evaluate


def test_multimodal_training_raw_replay_and_failure_experiment(tmp_path):
    root = tmp_path / "synthetic_raw"
    rate = 700
    time = np.arange(60*rate) / rate
    for subject_id in range(7):
        rng = np.random.default_rng(subject_id)
        signals = {"ECG": [], "EDA": [], "Resp": [], "ACC": []}
        for label in (0, 1):
            rr = (.85 if label == 0 else .60) + subject_id*.002
            phase = np.minimum(time % rr, rr-time % rr)
            signals["ECG"].append(np.exp(-.5*(phase/.015)**2) + rng.normal(0, .002, len(time)))
            signals["EDA"].append(1.5+label + .002*time + .05*np.sin(2*np.pi*.12*time))
            signals["Resp"].append(np.sin(2*np.pi*(.20 if label == 0 else .32)*time))
            signals["ACC"].append(np.zeros((len(time), 3)))
        chest = {k: np.concatenate(v) for k, v in signals.items()}
        path = root / f"S{subject_id}" / f"S{subject_id}.pkl"
        path.parent.mkdir(parents=True)
        with path.open("wb") as handle:
            pickle.dump({"signal": {"chest": chest}, "label": np.r_[np.ones(len(time)), np.full(len(time), 2)]}, handle)
    output = prepare(root, tmp_path / "dataset", seconds=30)
    frame = load_dataset(output / "classification.csv")
    bundle = build_bundle(frame, families=("logistic",))
    assert bundle["multimodal"]
    assert bundle["event_selection"]["status"] == "selected_on_policy_subjects"
    path = save_bundle(bundle, tmp_path / "bundle", output / "classification.csv")
    runtime = Runtime(load_bundle(path))
    row = frame.iloc[0].to_dict()
    source = RawSource(root)
    signals, fs = source.window(row)
    result = runtime.predict(signals=signals, rate=fs, interval=source_interval(row))
    assert result["input_kind"] == "waveform"
    assert result["feature_ms"] > 0
    failed = runtime.predict(signals=corrupt(signals, "ecg_flatline", row["row_id"]), rate=fs)
    assert failed["route"] == "alternate"
    rejected = runtime.predict(signals=corrupt(signals, "total_dropout", row["row_id"]), rate=fs)
    assert not rejected["accepted"] and rejected["prediction"] is None
    report = evaluate(frame, output / "classification.csv", tmp_path / "evaluation", families=("logistic",),
                      raw_root=root, scenarios=("clean", "total_dropout"))
    assert report["waveform_processing_measured"] and report["multimodal"]
    records = pd.read_csv(tmp_path / "evaluation/predictions.csv")
    failed = records[(records.scenario == "total_dropout") & (records.method == "cascade")]
    assert not failed.accepted.any()
    assert failed.stress_probability.isna().all()
    assert set(records.loc[records.scenario == "clean", "method"]) >= {"fusion", "cascade", "confidence_only", "cascade_single_window"}
