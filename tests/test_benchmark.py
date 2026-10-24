import pandas as pd
from pulsepoint.training import build_bundle, save_bundle
from pulsepoint.benchmark import benchmark


def test_interleaved_repeats_keep_independent_chronological_state(grouped_frame, tmp_path):
    frame = grouped_frame.copy()
    # Synthetic source intervals exist only within this fixture.
    frame["session"] = frame.subject
    frame["start_seconds"] = frame.groupby("subject").cumcount()*60
    frame["end_seconds"] = frame.start_seconds+60
    data = tmp_path / "features.csv"
    frame.to_csv(data, index=False)
    bundle = build_bundle(frame, families=("logistic",))
    path = save_bundle(bundle, tmp_path / "model", data)
    output = tmp_path / "benchmark"
    report = benchmark(path, data, output, limit=len(frame), repeats=3)
    timings = pd.read_csv(output / "timings.csv")
    assert len(timings) == len(frame)*3*2
    assert not timings.duplicated(["mode", "scenario", "repeat", "row_id"]).any()
    for _, rows in timings.groupby(["mode", "row_id"]):
        assert rows.event_state.nunique() == 1
        assert rows.event_opened.nunique() == 1
        assert rows.stress_probability.nunique() <= 1
    assert report["input_kind"] == "stored_features"
    assert all(result["count"] == len(frame)*3 for result in report["results"])
