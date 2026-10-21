from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "stress_detection_model.joblib"
DATA_PATH = BASE_DIR / "wesad_features.csv"
COMPARISON_PATH = BASE_DIR / "model_comparison.csv"
EXPECTED_FEATURES = ["mean", "std", "minimum", "maximum", "median", "skewness", "kurtosis"]
LABEL_NAMES = {0: "Baseline", 1: "Stress"}
COMPARISON_METRICS = {
    "Accuracy": "Accuracy",
    "Balanced accuracy": "Balanced Accuracy",
    "Stress precision": "Stress Precision",
    "Stress recall": "Stress Recall",
    "Stress F1": "Stress F1",
}

st.set_page_config(
    page_title="Pulsepoint | Stress Monitor",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { --ink:#183345; --muted:#6d8290; --teal:#177b76; --line:#e5edef; }
.stApp { background:#f4f8f8; color:var(--ink); font-family:Arial,sans-serif; }
[data-testid="stSidebar"] { background:#102d3b; }
[data-testid="stSidebar"] * { color:#eaf4f3 !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding:8px 10px; border-radius:9px; }
h1,h2,h3 { color:var(--ink); font-family:Arial,sans-serif !important; }
.hero { background:linear-gradient(115deg,#123745,#167873); padding:28px 32px; border-radius:19px; color:white; margin:4px 0 20px; }
.hero h1 { color:white; font-size:30px; margin:0 0 6px; }
.hero p { color:#c8e3df; margin:0; font-size:14px; }
.eyebrow { text-transform:uppercase; letter-spacing:.15em; color:#b7ddd7; font-size:10px; font-weight:700; margin-bottom:9px; }
.metric-card { background:white; border:1px solid var(--line); border-radius:15px; padding:17px 19px; min-height:112px; box-shadow:0 4px 18px #163b4810; }
.metric-label { color:var(--muted); font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }
.metric-value { color:var(--ink); font-size:27px; font-weight:800; margin-top:12px; }
.metric-note { color:var(--muted); font-size:12px; margin-top:3px; }
.panel { background:white; padding:20px 22px; border:1px solid var(--line); border-radius:15px; margin:13px 0; }
.small-note { color:#718292; font-size:13px; }
.stButton button, .stDownloadButton button { border-radius:10px; font-weight:700; }
div[data-testid="stAlert"] { border-radius:12px; }
</style>
""", unsafe_allow_html=True)


def card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def show_hero(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{eyebrow}</div>'
        f'<h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH.name}")
    saved = joblib.load(MODEL_PATH)
    if isinstance(saved, dict):
        model = saved.get("model")
        features = saved.get("features", EXPECTED_FEATURES)
    else:
        model, features = saved, EXPECTED_FEATURES
    if model is None or not callable(getattr(model, "predict", None)):
        raise ValueError("The saved model bundle does not contain a usable estimator.")
    return model, list(features)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Feature dataset not found: {DATA_PATH.name}")
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_comparison() -> pd.DataFrame | None:
    if not COMPARISON_PATH.exists():
        return None
    return pd.read_csv(COMPARISON_PATH)


def prepare_samples(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    missing = [name for name in features if name not in data.columns]
    if missing:
        raise ValueError("Feature CSV is missing model inputs: " + ", ".join(missing))
    samples = data.copy()
    samples[features] = samples[features].apply(pd.to_numeric, errors="coerce")
    samples = samples.replace([np.inf, -np.inf], np.nan)
    samples = samples.dropna(subset=features).reset_index(names="source_row")
    if "label" in samples:
        samples["label"] = pd.to_numeric(samples["label"], errors="coerce")
    return samples


def predict(model, features: list[str], row: pd.DataFrame):
    sample = row[features]
    started = perf_counter()
    prediction = int(model.predict(sample)[0])
    probabilities: dict[int, float] = {}
    if callable(getattr(model, "predict_proba", None)):
        scores = model.predict_proba(sample)[0]
        for class_id, score in zip(model.classes_, scores):
            probabilities[int(class_id)] = float(score)
    elapsed_ms = (perf_counter() - started) * 1000
    return prediction, probabilities, elapsed_ms


def record_prediction(row: pd.DataFrame, prediction: int, probabilities: dict[int, float], latency_ms: float, source: str) -> dict:
    sample = row.iloc[0]
    timestamp = datetime.now().astimezone()
    entry = {
        "timestamp": timestamp.isoformat(timespec="seconds"),
        "source_row": int(sample["source_row"]) + 1,
        "source": source,
        "condition": LABEL_NAMES.get(prediction, f"Class {prediction}"),
        "baseline_probability": probabilities.get(0),
        "stress_probability": probabilities.get(1),
        "latency_ms": float(latency_ms),
    }
    if "label" in sample and pd.notna(sample["label"]):
        entry["known_label"] = LABEL_NAMES.get(int(sample["label"]), str(int(sample["label"])))
    else:
        entry["known_label"] = "Unavailable"
    st.session_state.prediction_history.append(entry)
    st.session_state.latest_prediction = entry
    return entry


def show_alert(probability: float | None, threshold: float) -> None:
    if probability is None:
        st.info("Class probabilities are unavailable for this saved estimator.")
    elif probability >= threshold:
        st.warning(
            f"Demonstration alert: stress probability ({probability:.1%}) meets or exceeds "
            f"your {threshold:.0%} threshold. This is a model demo alert, not a medical warning."
        )


def show_prediction(entry: dict, threshold: float) -> None:
    condition = entry["condition"]
    if condition == "Stress":
        st.error("Stress pattern predicted")
    elif condition == "Baseline":
        st.success("Baseline pattern predicted")
    else:
        st.info(f"Predicted condition: {condition}")
    c1, c2, c3 = st.columns(3)
    with c1:
        value = entry["baseline_probability"]
        card("Baseline probability", f"{value:.1%}" if value is not None else "Unavailable", "Model output for this sample")
    with c2:
        value = entry["stress_probability"]
        card("Stress probability", f"{value:.1%}" if value is not None else "Unavailable", "Model output for this sample")
    with c3:
        card("Inference latency", f"{entry['latency_ms']:.3f} ms", "Measured on this computer")
    show_alert(entry["stress_probability"], threshold)


def clear_session() -> None:
    st.session_state.monitor_running = False
    st.session_state.monitor_cursor = 0
    st.session_state.monitor_last = None
    st.session_state.prediction_history = []
    st.session_state.latest_prediction = None


for key, initial in {
    "prediction_history": [],
    "latest_prediction": None,
    "monitor_running": False,
    "monitor_cursor": 0,
    "monitor_last": None,
    "alert_threshold": 0.70,
}.items():
    if key not in st.session_state:
        st.session_state[key] = initial

with st.sidebar:
    st.markdown("## ✳ Pulsepoint")
    st.caption("EDGE HEALTH  /  WESAD")
    st.divider()
    page = st.radio(
        "Workspace",
        ["Overview", "Prediction", "Monitoring", "History", "Analysis", "Performance"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**DEMO ALERT THRESHOLD**")
    threshold_pct = st.slider(
        "Stress probability threshold",
        min_value=50,
        max_value=95,
        value=int(round(st.session_state.alert_threshold * 100)),
        step=5,
        format="%d%%",
        label_visibility="collapsed",
        key="alert_threshold_slider",
    )
    st.session_state.alert_threshold = threshold_pct / 100
    st.caption("Alerts are demonstrations, not medical warnings.")
    st.divider()
    st.markdown("**INFERENCE**")
    st.markdown("🟢 Local model")
    st.caption("Prediction runs on this computer. No sensor data is uploaded.")

try:
    model, features = load_model()
    raw_dataset = load_dataset()
    samples = prepare_samples(raw_dataset, features)
except Exception as exc:
    st.error(f"Dashboard could not load a required project artifact: {exc}")
    st.info("Keep app.py beside stress_detection_model.joblib and wesad_features.csv.")
    st.stop()

if samples.empty:
    st.error("The feature CSV has no complete, finite rows that can be sent to the saved model.")
    st.stop()

threshold = st.session_state.alert_threshold

if page == "Overview":
    show_hero("EDGE-BASED HEALTHCARE MONITORING", "Stress detection workspace", "Explore local model predictions and prerecorded WESAD demonstrations.")
    st.info("WESAD data is prerecorded. This dashboard has no live wearable connection.")
    c1, c2, c3 = st.columns(3)
    with c1:
        card("Usable feature windows", f"{len(samples):,}", "Complete rows from wesad_features.csv")
    with c2:
        card("Session predictions", f"{len(st.session_state.prediction_history):,}", "Stored in this browser session")
    with c3:
        state = "Monitoring" if st.session_state.monitor_running else "Standby"
        card("Simulation state", state, "Prerecorded sample playback")
    if st.session_state.latest_prediction:
        st.markdown("### Most recent prediction")
        st.caption(f"{st.session_state.latest_prediction['timestamp']} · source row {st.session_state.latest_prediction['source_row']}")
        show_prediction(st.session_state.latest_prediction, threshold)
    st.markdown('<div class="panel"><h3>Start here</h3><p class="small-note">Choose Prediction to run one selected feature row. Monitoring steps through stored WESAD rows on a timer. Every result is retained in History until you reset the session.</p></div>', unsafe_allow_html=True)

elif page == "Prediction":
    show_hero("LOCAL MODEL INFERENCE", "Run a prediction", "Choose an existing feature window, then explicitly run it through the saved model.")
    st.info("Dataset-based demonstration only. The selected row is not a live wearable measurement.")
    sample_options = samples["source_row"].astype(int).tolist()
    def sample_label(source_row: int) -> str:
        item = samples.loc[samples["source_row"] == source_row].iloc[0]
        details = [f"Row {int(source_row) + 1}"]
        if "subject" in samples.columns:
            details.append(str(item["subject"]))
        if "label" in samples.columns and pd.notna(item["label"]):
            details.append(f"known {LABEL_NAMES.get(int(item['label']), int(item['label']))}")
        return " · ".join(details)

    chosen_row = st.selectbox("WESAD sample", sample_options, format_func=sample_label)
    sample_row = samples[samples["source_row"] == chosen_row].iloc[[0]]
    with st.expander("Inspect stored model features", expanded=False):
        st.dataframe(sample_row[features].T.rename(columns={sample_row.index[0]: "Value"}), use_container_width=True)
    if st.button("Run Prediction", type="primary", use_container_width=True):
        try:
            prediction, probabilities, latency = predict(model, features, sample_row)
            record_prediction(sample_row, prediction, probabilities, latency, "Manual")
        except Exception as exc:
            st.error(f"Prediction failed for this row: {exc}")
    latest = st.session_state.latest_prediction
    if latest:
        st.markdown("### Latest result")
        st.caption(f"Sample row {latest['source_row']} · {latest['timestamp']} · {latest['source']} prediction")
        show_prediction(latest, threshold)

elif page == "Monitoring":
    show_hero("PRERECORDED-DATA SIMULATION", "Monitoring demo", "A timed replay of stored WESAD feature windows, processed by the local model.")
    st.warning("Simulation only: these are prerecorded dataset rows, not a live wearable connection or live physiological measurements.")
    interval = st.select_slider("Sample interval", options=[1, 2, 3, 5, 10], value=2, format_func=lambda value: f"{value} sec")
    start_col, stop_col, reset_col = st.columns([1, 1, 1])
    with start_col:
        if st.button("▶  Start Monitoring", type="primary", disabled=st.session_state.monitor_running, use_container_width=True):
            st.session_state.monitor_running = True
            st.session_state.monitor_cursor = 1
            row = samples.iloc[[0]]
            prediction, probabilities, latency = predict(model, features, row)
            st.session_state.monitor_last = record_prediction(row, prediction, probabilities, latency, "Simulation")
            st.rerun()
    with stop_col:
        if st.button("■  Stop Monitoring", disabled=not st.session_state.monitor_running, use_container_width=True):
            st.session_state.monitor_running = False
            st.rerun()
    with reset_col:
        if st.button("↺  Reset Session", use_container_width=True):
            clear_session()
            st.rerun()

    @st.fragment(run_every=interval if st.session_state.monitor_running else None)
    def monitoring_panel():
        if st.session_state.monitor_running:
            next_sample_index = st.session_state.monitor_cursor % len(samples)
            row = samples.iloc[[next_sample_index]]
            prediction, probabilities, latency = predict(model, features, row)
            st.session_state.monitor_last = record_prediction(row, prediction, probabilities, latency, "Simulation")
            st.session_state.monitor_cursor = next_sample_index + 1
        latest = st.session_state.monitor_last
        status = "RUNNING · prerecorded replay" if st.session_state.monitor_running else "STOPPED · prerecorded replay"
        st.markdown(f"**Status:** {status}")
        if latest is None:
            st.caption("Press Start Monitoring to begin processing stored samples.")
            return
        c1, c2, c3 = st.columns(3)
        with c1:
            card("Current prediction", latest["condition"], f"Source row {latest['source_row']}")
        with c2:
            score = latest["stress_probability"]
            card("Stress probability", f"{score:.1%}" if score is not None else "Unavailable", "Saved model output")
        with c3:
            card("Last processed", latest["timestamp"], f"Inference {latest['latency_ms']:.3f} ms")
        show_alert(latest["stress_probability"], threshold)
        chart_history = [item for item in st.session_state.prediction_history if item["source"] == "Simulation" and item["stress_probability"] is not None]
        if chart_history:
            trend = pd.DataFrame(chart_history)
            trend["timestamp"] = pd.to_datetime(trend["timestamp"])
            trend = trend.set_index("timestamp")[["stress_probability"]].rename(columns={"stress_probability": "Stress probability"})
            st.markdown("#### Stress score trend")
            st.line_chart(trend, height=260)
            st.caption("Each point represents one prediction from a prerecorded WESAD row.")

    monitoring_panel()

elif page == "History":
    show_hero("CURRENT SESSION", "Prediction history", "Review, filter, and export results created during this app session.")
    history = st.session_state.prediction_history
    st.caption("Rows are stored in Streamlit session state and are cleared with Reset Session or when the session ends.")
    if not history:
        st.info("No predictions yet. Run a selected sample or start the monitoring simulation to add history.")
    else:
        condition_filter = st.selectbox("Filter by predicted condition", ["All conditions", "Baseline", "Stress"])
        history_df = pd.DataFrame(history)
        if condition_filter != "All conditions":
            history_df = history_df[history_df["condition"] == condition_filter]
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        csv_bytes = history_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_bytes, file_name="stress_prediction_history.csv", mime="text/csv", type="primary")
        st.caption(f"Showing {len(history_df)} of {len(history)} session predictions.")
        if st.button("Reset Session", help="Stops monitoring and clears this session’s prediction history."):
            clear_session()
            st.rerun()

elif page == "Analysis":
    show_hero("SAVED EVALUATION RESULTS", "Interactive model analysis", "Choose a saved metric to compare the three models evaluated by your project.")
    try:
        comparison = load_comparison()
    except Exception as exc:
        comparison = None
        st.error(f"Could not read model_comparison.csv: {exc}")
    if comparison is None or comparison.empty:
        st.warning("model_comparison.csv is missing or empty. Run compare_models.py to create it.")
    elif "Model" not in comparison:
        st.error("Comparison results need a Model column.")
    else:
        valid_metrics = {label: column for label, column in COMPARISON_METRICS.items() if column in comparison.columns}
        if not valid_metrics:
            st.error("No supported saved metric columns were found in model_comparison.csv.")
        else:
            metric_label = st.selectbox("Metric to compare", list(valid_metrics))
            metric_column = valid_metrics[metric_label]
            chart_data = comparison[["Model", metric_column]].copy()
            chart_data[metric_column] = pd.to_numeric(chart_data[metric_column], errors="coerce")
            chart_data = chart_data.dropna(subset=[metric_column]).set_index("Model")
            if chart_data.empty:
                st.warning(f"The saved {metric_label.lower()} values are not numeric.")
            else:
                st.caption(f"Showing actual values saved in model_comparison.csv · {metric_label}")
                st.bar_chart(chart_data, y=metric_column, y_label="Score", x_label="Model", height=360)
                st.dataframe(chart_data.rename(columns={metric_column: metric_label}).style.format("{:.1%}"), use_container_width=True)

elif page == "Performance":
    show_hero("MODEL HEALTH", "Performance", "Saved subject-wise results and device-measured edge footprint.")
    try:
        comparison = load_comparison()
    except Exception as exc:
        comparison = None
        st.error(f"Could not read model_comparison.csv: {exc}")
    if comparison is None or comparison.empty:
        st.info("Saved evaluation metrics are unavailable because model_comparison.csv is missing or empty.")
    else:
        st.caption("Accuracy, balanced accuracy, stress precision, stress recall, and stress F1 are read from your saved subject-wise comparison results.")
        st.dataframe(comparison.style.format({column: "{:.1%}" for column in COMPARISON_METRICS.values() if column in comparison.columns}), use_container_width=True, hide_index=True)
    size_bytes = MODEL_PATH.stat().st_size if MODEL_PATH.exists() else None
    a, b = st.columns(2)
    with a:
        card("Saved model size", f"{size_bytes / 1024:.1f} KB" if size_bytes is not None else "Unavailable", "Measured from stress_detection_model.joblib")
    with b:
        if st.button("Measure local inference latency", type="primary", use_container_width=True):
            try:
                _, _, elapsed = predict(model, features, samples.iloc[[0]])
                st.metric("Single-row latency", f"{elapsed:.3f} ms")
                st.caption("Measured on this computer for one prediction and probability calculation.")
            except Exception as exc:
                st.error(f"Could not measure local inference: {exc}")
        else:
            card("Inference latency", "Not measured", "Run a device timing to record it")
