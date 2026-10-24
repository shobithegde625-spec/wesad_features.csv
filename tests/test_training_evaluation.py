import importlib
import json
import numpy as np
import pandas as pd
import pytest
from pulsepoint.models import development_split, fit_experts
from pulsepoint.training import build_bundle, save_bundle, load_bundle
from pulsepoint.replay import Runtime
from pulsepoint.evaluation import evaluate
from pulsepoint.corruption import corrupt


def test_participant_roles_are_disjoint(grouped_frame):
    parts, manifest = development_split(grouped_frame, 42)
    roles = list(manifest)
    for i, a in enumerate(roles):
        for b in roles[i+1:]:
            assert set(manifest[a]).isdisjoint(manifest[b])
    assert set().union(*map(set, manifest.values())) == set(grouped_frame.subject)


def test_artifact_roundtrip_and_no_overwrite(grouped_frame, tmp_path):
    data = tmp_path / "data.csv"
    grouped_frame.to_csv(data, index=False)
    bundle = build_bundle(grouped_frame, families=("logistic",))
    path = save_bundle(bundle, tmp_path / "model", data)
    loaded = load_bundle(path)
    runtime = Runtime(loaded)
    row = grouped_frame.iloc[0].to_dict()
    a = runtime.predict(features=row)
    row["label"], row["subject"] = 1, "changed"
    b = runtime.predict(features=row)
    assert a["stress_probability"] == b["stress_probability"]
    assert a["event_state"] == "unavailable_without_source_time"
    with pytest.raises(FileExistsError):
        save_bundle(bundle, tmp_path / "model", data)


def test_nested_evaluation_writes_disjoint_manifests(grouped_frame, tmp_path):
    # Seven total subjects is the smallest supported nested outer experiment.
    frame = grouped_frame[grouped_frame.subject.isin([f"S{i}" for i in range(7)])]
    data = tmp_path / "data.csv"
    frame.to_csv(data, index=False)
    report = evaluate(frame, data, tmp_path / "report", families=("logistic",))
    manifests = json.loads((tmp_path / "report/split_manifest.json").read_text())
    assert len(manifests) == 7
    for fold in manifests:
        held_out = set(fold["outer_test_subjects"])
        assert held_out.isdisjoint(fold["outer_train_subjects"])
        for role in fold["roles"].values():
            assert held_out.isdisjoint(role)
    predictions = pd.read_csv(tmp_path / "report/predictions.csv")
    for _, method in predictions.groupby("method"):
        assert set(method.row_id) == set(frame.row_id)
    assert report["subjects"] == 7
    assert (tmp_path / "report/evaluation.png").exists()


def test_waveform_corruption_is_reproducible(synthetic_signals):
    a = corrupt(synthetic_signals, "ecg_noise_severe", "one")
    b = corrupt(synthetic_signals, "ecg_noise_severe", "one")
    assert np.array_equal(a["ecg"], b["ecg"])
    assert not np.array_equal(a["ecg"], synthetic_signals["ecg"])
    assert set(corrupt(synthetic_signals, "total_dropout", "one")) == {"acc"}


def test_scripts_are_safe_to_import():
    for name in ["prepare_dataset", "train_model", "validate_model", "compare_models", "edge_inference",
                 "benchmark_inference", "plot_results", "measure_model", "test_model", "inspect_wesad", "explore_labels"]:
        importlib.import_module(name)
