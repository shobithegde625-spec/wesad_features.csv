"""Pulsepoint: local prerecorded monitoring with shared quality-aware runtime."""
from datetime import datetime
from pathlib import Path
import json
import uuid

import numpy as np
import pandas as pd
import streamlit as st

from pulsepoint.data import BASE_DIR, RawSource, load_dataset, file_hash
from pulsepoint.replay import Runtime, chronological, source_interval
from pulsepoint.training import load_bundle
from pulsepoint.corruption import SCENARIOS, corrupt

st.set_page_config(page_title="Pulsepoint | Stress Monitor", page_icon="✳", layout="wide")
st.markdown("""<style>
.stApp {background:#f4f8f8;color:#183345;font-family:Arial,sans-serif}
[data-testid="stSidebar"] {background:#102d3b}
[data-testid="stSidebar"] * {color:#eaf4f3}
.hero {background:linear-gradient(115deg,#123745,#167873);padding:26px 30px;border-radius:18px;color:white;margin-bottom:20px}
.hero h1 {color:white;margin:0;font-size:30px}.hero p {color:#c8e3df;margin-bottom:0}
.stButton button,.stDownloadButton button {border-radius:10px}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def cached_bundle(path, mtime):
    return load_bundle(path)


@st.cache_data
def cached_data(path, mtime):
    return load_dataset(path, min_subjects=1)


@st.cache_resource
def raw_source(path):
    return RawSource(path)


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def show_result(entry):
    if not entry["accepted"]:
        st.warning("Insufficient evidence — " + entry["reason"].replace("_", " "))
    elif entry["prediction"] == 1:
        st.error("Stress pattern predicted")
    else:
        st.success("Baseline pattern predicted")
    columns = st.columns(3)
    p = entry["stress_probability"]
    columns[0].metric("Stress probability", f"{p:.1%}" if p is not None else "Unavailable")
    columns[1].metric("Processing time", f"{entry['total_ms']:.2f} ms")
    columns[2].metric("Event state", entry["event_state"].replace("_", " ").title())
    st.caption(f"Evidence: {entry['route']} · source row {entry['source_row']} · {entry['subject']} · {entry['timestamp']}")
    if entry["event_opened"]:
        st.warning("Sustained stress pattern detected in this prerecorded demonstration.")
    reasons = {name: q["reason"] for name, q in entry["quality"].items()}
    with st.expander("Signal quality and processing details"):
        st.json({"quality": reasons, "quality_ms": entry["quality_ms"], "feature_ms": entry["feature_ms"],
                 "inference_ms": entry["inference_ms"], "input_kind": entry["input_kind"],
                 "source_interval": entry.get("source_interval"), "artifact_id": entry["artifact_id"]})


for key, value in {"history": [], "cursor": 0, "running": False, "latest": None, "runtime": None,
                   "context": None, "replay_id": None}.items():
    if key not in st.session_state:
        st.session_state[key] = value

with st.sidebar:
    st.markdown("## ✳ Pulsepoint")
    st.caption("LOCAL STRESS MONITORING")
    page = st.radio("Workspace", ["Overview", "Prediction", "Monitoring", "History", "Analysis", "Performance"])
    artifacts = sorted((p for p in BASE_DIR.glob("artifacts/*/model.joblib") if p.with_name("metadata.json").exists()),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    if not artifacts:
        st.info("Train a local model to begin.")
        st.code("python train_model.py")
        st.stop()
    chosen_bundle = st.selectbox("Model", artifacts, format_func=lambda p: p.parent.name)
    try:
        bundle = cached_bundle(str(chosen_bundle), chosen_bundle.stat().st_mtime_ns)
    except Exception as exc:
        st.error(f"Could not load this model: {exc}")
        st.stop()
    default_data = bundle.get("dataset_path", str(BASE_DIR / "wesad_features.csv"))
    replay_path = Path(default_data).with_name("replay.csv")
    if bundle["multimodal"] and replay_path.exists():
        default_data = str(replay_path)
    default_raw = ""
    dataset_metadata = Path(default_data).parent / "dataset.json"
    if bundle["multimodal"] and dataset_metadata.exists():
        configured_raw = json.loads(dataset_metadata.read_text()).get("raw_root", "")
        if configured_raw and Path(configured_raw).exists():
            default_raw = configured_raw
    data_path_text = st.text_input("Feature or replay CSV", value=default_data)
    raw_root_text = st.text_input("Raw WESAD folder or ZIP (optional)", value=default_raw,
                                 help="Required for waveform replay and corruption demos. A ZIP avoids extracting the full recordings.")
    mode = st.selectbox("Inference policy", ["cascade", "ecg"] + (["fusion", "confidence_only", "no_abstention"] if bundle["multimodal"] else ["no_abstention"]))
    scenario = st.selectbox("Waveform scenario", list(SCENARIOS), disabled=not bool(raw_root_text.strip()))
    st.caption("Prerecorded research demonstration. Predictions are not medical diagnoses.")

try:
    data_path = Path(data_path_text).expanduser().resolve()
    samples = cached_data(str(data_path), data_path.stat().st_mtime_ns)
    expected = bundle["experts"]["ecg"].features
    missing = set(expected) - set(samples)
    if missing:
        raise ValueError("Dataset lacks inputs for this model: " + ", ".join(sorted(missing)))
    source = raw_source(raw_root_text.strip()) if raw_root_text.strip() else None
    if source and not {"source_file", "sampling_rate", "start_sample", "end_sample"}.issubset(samples):
        raise ValueError("Waveform replay requires metadata from prepare_dataset.py")
    if not source:
        scenario = "clean"
except Exception as exc:
    st.error(f"Could not load the selected data source: {exc}")
    st.stop()

with st.sidebar:
    subject = st.selectbox("Participant", sorted(samples.subject.unique()))
    selected = samples[samples.subject == subject]
    session = None
    if "session" in selected:
        session = st.selectbox("Recording", sorted(selected.session.astype(str).unique()))
        selected = selected[selected.session.astype(str) == session]
    selected = chronological(selected).copy()
    st.caption(f"{len(selected)} stored windows · processing stays on this computer")

context = (str(chosen_bundle), chosen_bundle.stat().st_mtime_ns, str(data_path), data_path.stat().st_mtime_ns,
           raw_root_text, subject, session, mode, scenario)
if st.session_state.context != context:
    st.session_state.update(context=context, cursor=0, running=False, latest=None,
                            runtime=Runtime(bundle, mode), replay_id=uuid.uuid4().hex[:8])


def process_row(row, source_kind, runtime):
    interval = source_interval(row) if source_kind == "Replay" else None
    if source:
        signals, rate = source.window(row)
        result = runtime.predict(signals=corrupt(signals, scenario, row["row_id"]), rate=rate, interval=interval)
    else:
        # Only the declared model features and quality fields are consumed by the runtime.
        result = runtime.predict(features=row, interval=interval)
    entry = {**result, "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
             "source_row": str(row["row_id"]), "subject": row["subject"], "session": row.get("session"),
             "source": source_kind, "scenario": scenario, "source_interval": interval,
             "replay_id": st.session_state.replay_id, "known_label": row.get("label"),
             "start_seconds": row.get("start_seconds"), "end_seconds": row.get("end_seconds")}
    st.session_state.history.append(entry)
    st.session_state.latest = entry
    return entry


if page == "Overview":
    hero("Stress detection workspace", "Quality-aware local inference and prerecorded monitoring")
    st.info("This workspace uses stored WESAD data. There is no live wearable connection.")
    cols = st.columns(3)
    cols[0].metric("Dataset windows", len(samples))
    cols[1].metric("Participants", samples.subject.nunique())
    cols[2].metric("Session predictions", len(st.session_state.history))
    if not bundle["multimodal"]:
        st.info("This model uses the original ECG summaries. Multimodal routing requires a model trained on regenerated waveform features.")
    if not source:
        st.caption("Stored-feature mode: processing time excludes waveform feature extraction. Original summary rows have no verified signal-quality annotations.")
    if "start_seconds" not in samples:
        st.caption("This CSV has no source timestamps. Sustained event detection is unavailable.")
    if st.session_state.latest:
        show_result(st.session_state.latest)

elif page == "Prediction":
    hero("Run a prediction", "Select a stored window; manual predictions do not accumulate event evidence")
    positions = list(range(len(selected)))
    position = st.selectbox("Window", positions, format_func=lambda i: f"{subject} · row {selected.iloc[i]['row_id']}")
    row = selected.iloc[position].to_dict()
    with st.expander("Inspect stored features"):
        st.dataframe(pd.Series({f: row.get(f) for f in expected}, name="Value"))
    if st.button("Run Prediction", type="primary"):
        try:
            process_row(row, "Manual", Runtime(bundle, mode))
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
    if st.session_state.latest:
        show_result(st.session_state.latest)

elif page == "Monitoring":
    hero("Monitoring replay", "Source-time event decisions from one participant and recording")
    st.info("Playback speed controls the display only. Physiological intervals come from recording metadata.")
    if not source:
        st.caption("Stored-feature replay; waveform processing costs are excluded.")
    interval = st.select_slider("Playback interval (seconds)", [1, 2, 3, 5, 10], value=2)
    columns = st.columns(3)
    if columns[0].button("Start / Resume", type="primary", disabled=st.session_state.running):
        if st.session_state.cursor >= len(selected):
            st.session_state.cursor = 0
            st.session_state.runtime.reset()
            st.session_state.replay_id = uuid.uuid4().hex[:8]
        st.session_state.running = True
        st.rerun()
    if columns[1].button("Stop", disabled=not st.session_state.running):
        st.session_state.running = False
        st.rerun()
    if columns[2].button("Restart recording"):
        st.session_state.update(cursor=0, running=False, latest=None, replay_id=uuid.uuid4().hex[:8])
        st.session_state.runtime.reset()
        st.rerun()

    @st.fragment(run_every=interval if st.session_state.running else None)
    def monitoring_panel():
        if st.session_state.running:
            if st.session_state.cursor < len(selected):
                row = selected.iloc[st.session_state.cursor].to_dict()
                try:
                    process_row(row, "Replay", st.session_state.runtime)
                    st.session_state.cursor += 1
                except Exception as exc:
                    st.session_state.running = False
                    st.error(f"Replay stopped: {exc}")
            if st.session_state.cursor >= len(selected):
                st.session_state.running = False
                # Refresh the surrounding controls as well as the fragment.
                st.rerun()
        st.progress(min(st.session_state.cursor/len(selected), 1.0))
        if st.session_state.cursor >= len(selected):
            st.success("Recording complete. Start again to begin a new replay.")
        else:
            st.caption("Running" if st.session_state.running else "Stopped")
        if st.session_state.latest:
            show_result(st.session_state.latest)
        history = [r for r in st.session_state.history if r["source"] == "Replay" and r["replay_id"] == st.session_state.replay_id]
        if history:
            chart = pd.DataFrame(history)
            index = "end_seconds" if chart.end_seconds.notna().all() else "source_row"
            st.line_chart(chart.set_index(index)[["stress_probability"]], height=260)
    monitoring_panel()

elif page == "History":
    hero("Prediction history", "Results from this browser session, including rejected decisions")
    if not st.session_state.history:
        st.info("Run a prediction or monitoring replay to add results.")
    else:
        history = pd.DataFrame(st.session_state.history)
        history["quality"] = history.quality.map(json.dumps)
        history["source_interval"] = history.source_interval.map(json.dumps)
        st.dataframe(history, hide_index=True)
        st.download_button("Download CSV", history.to_csv(index=False).encode(), "stress_prediction_history.csv", "text/csv")
        if st.button("Clear session history"):
            st.session_state.update(history=[], latest=None, running=False, cursor=0, replay_id=uuid.uuid4().hex[:8])
            st.session_state.runtime.reset()
            st.rerun()

elif page == "Analysis":
    hero("Evaluation analysis", "Participant-held-out estimates with coverage and operational recall")
    reports = sorted(BASE_DIR.glob("reports/*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    matching = [p for p in reports if json.loads(p.read_text()).get("dataset_sha256") == bundle.get("dataset_sha256")]
    if not matching:
        st.info("No evaluation report for this model's training dataset. Run evaluate.py to produce one.")
    else:
        report_path = st.selectbox("Evaluation run", matching, format_func=lambda p: p.parent.name)
        report = json.loads(report_path.read_text())
        st.caption(f"Protocol: {report['protocol']} · {report['subjects']} participants. These folds evaluate the training procedure, not this specific fitted deployment model.")
        comparison = pd.read_csv(report_path.parent / "comparison.csv")
        st.dataframe(comparison, hide_index=True)
        clean = comparison[comparison.scenario == "clean"].set_index("method")
        metric = st.selectbox("Metric", ["coverage", "all_window_stress_recall", "stress_f1", "mean_total_ms"])
        st.bar_chart(clean[[metric]])
        if (report_path.parent / "evaluation.png").exists():
            st.image(str(report_path.parent / "evaluation.png"))
    with st.expander("Historical comparison from the original project"):
        historical = BASE_DIR / "model_comparison.csv"
        if historical.exists():
            st.caption("Historical results use different estimator settings and do not describe the selected model.")
            st.dataframe(pd.read_csv(historical), hide_index=True)

elif page == "Performance":
    hero("Model performance", "Artifact provenance and measured processing footprint")
    a, b = st.columns(2)
    a.metric("Complete model bundle", f"{chosen_bundle.stat().st_size/1024:.1f} KiB")
    b.metric("Expert models", len(bundle["experts"]))
    st.caption(f"Artifact: {bundle['artifact_id']} · created {bundle['created_at']}")
    st.write("Confidence policy", bundle["policy"])
    st.write("Event policy", bundle["events"])
    if st.button("Measure this window locally"):
        row = selected.iloc[0].to_dict()
        result = process_row(row, "Benchmark", Runtime(bundle, mode))
        st.metric("Processing time", f"{result['total_ms']:.2f} ms")
        st.caption("One host measurement; acquisition time and device energy are excluded.")
    with st.expander("Training provenance"):
        st.json({"artifact_id": bundle["artifact_id"], "dataset_sha256": bundle.get("dataset_sha256"),
                 "roles": bundle["manifest"], "selection": bundle["selections"], "environment": bundle["environment"]})
