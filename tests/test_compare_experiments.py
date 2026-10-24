import json
import pandas as pd
import pytest
from compare_experiments import compare


def make_report(path, recall):
    path.mkdir()
    (path / "summary.json").write_text(json.dumps(dict(protocol="loso", dataset_sha256="a", replay_sha256="b", subjects=2, windows=2)))
    (path / "split_manifest.json").write_text(json.dumps([dict(outer_train_subjects=["a"], outer_test_subjects=["b"]),
                                                          dict(outer_train_subjects=["b"], outer_test_subjects=["a"])]))
    (path / "targets.json").write_text(json.dumps({"compute_and_recall": {"status": "not_met_on_this_host"}}))
    pd.DataFrame([dict(row_id="one", scenario="clean", method="cascade", label=0),
                  dict(row_id="two", scenario="clean", method="cascade", label=1)]).to_csv(path / "predictions.csv", index=False)
    pd.DataFrame([dict(method="cascade", scenario="clean", coverage=1., all_window_stress_recall=recall,
                       stress_f1=recall, mean_cpu_ms=10.)]).to_csv(path / "comparison.csv", index=False)
    pd.DataFrame([dict(method="cascade", scenario="clean", subject=s, coverage=1., all_window_stress_recall=recall,
                       stress_f1=recall, mean_cpu_ms=10.) for s in ("a", "b")]).to_csv(path / "per_subject.csv", index=False)


def test_revision_comparison_preserves_paired_participants(tmp_path):
    old, new, output = [tmp_path / p for p in ("old", "new", "comparison")]
    make_report(old, .5)
    make_report(new, .75)
    compare(old, new, output)
    differences = json.loads((output / "paired_revision_differences.json").read_text())
    recall = next(d for d in differences if d["metric"] == "all_window_stress_recall")
    assert recall["subjects"] == 2
    assert recall["mean_difference"] == recall["lower_95"] == recall["upper_95"] == .25
    assert "exploratory revision" in (output / "FINDINGS.md").read_text()


def test_revision_comparison_rejects_different_targets(tmp_path):
    old, new, output = [tmp_path / p for p in ("old", "new", "comparison")]
    make_report(old, .5)
    make_report(new, .75)
    rows = pd.read_csv(new / "predictions.csv")
    rows.loc[0, "label"] = 1
    rows.to_csv(new / "predictions.csv", index=False)
    with pytest.raises(ValueError, match="identical source intervals"):
        compare(old, new, output)
    assert not output.exists()
