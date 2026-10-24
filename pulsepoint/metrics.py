"""Coverage-aware window and chronological event metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, brier_score_loss, log_loss


def window_metrics(frame):
    frame = frame[frame.label.isin([0, 1])].copy()
    if frame.empty:
        return {"windows": 0}
    y = frame.label.to_numpy(dtype=int)
    accepted = frame.accepted.to_numpy(dtype=bool)
    pred = frame.prediction.fillna(-1).to_numpy(dtype=int)
    available = frame.stress_probability.notna().to_numpy()
    result = {"windows": len(frame), "coverage": float(accepted.mean()),
              "baseline_coverage": float(accepted[y == 0].mean()) if (y == 0).any() else None,
              "stress_coverage": float(accepted[y == 1].mean()) if (y == 1).any() else None,
              "all_window_stress_recall": float(np.mean(pred[y == 1] == 1)) if (y == 1).any() else None,
              "probability_coverage": float(available.mean()),
              "escalation_fraction": float(frame.escalated.mean()) if "escalated" in frame else 0.0}
    if accepted.any():
        ya, pa = y[accepted], pred[accepted]
        result.update(accuracy=float(accuracy_score(ya, pa)),
                      balanced_accuracy=float(balanced_accuracy_score(ya, pa)) if len(set(ya)) == 2 else None,
                      stress_precision=float(precision_score(ya, pa, zero_division=0)),
                      stress_recall=float(recall_score(ya, pa, zero_division=0)),
                      stress_f1=float(f1_score(ya, pa, zero_division=0)),
                      selective_error=float(np.mean(ya != pa)))
    else:
        result.update({k: None for k in ("accuracy", "balanced_accuracy", "stress_precision", "stress_recall", "stress_f1", "selective_error")})
    for subset, mask in [("", available), ("accepted_", available & accepted)]:
        if mask.any():
            p = frame.stress_probability.to_numpy(dtype=float)[mask]
            result[subset + "brier"] = float(brier_score_loss(y[mask], p))
            result[subset + "log_loss"] = float(log_loss(y[mask], np.column_stack([1-p, p]), labels=[0, 1]))
        else:
            result[subset + "brier"] = result[subset + "log_loss"] = None
    for field in ("quality_ms", "feature_ms", "inference_ms", "total_ms", "cpu_ms"):
        if field in frame and frame[field].notna().any():
            result[f"mean_{field}"] = float(frame[field].mean())
            result[f"p95_{field}"] = float(frame[field].quantile(.95))
    return result


def risk_coverage(frame):
    valid = frame[frame.label.isin([0, 1]) & frame.stress_probability.notna()].copy()
    if valid.empty:
        return pd.DataFrame(columns=["coverage", "risk", "stress_coverage", "threshold"])
    valid["confidence"] = np.maximum(valid.stress_probability, 1 - valid.stress_probability)
    valid = valid.sort_values("confidence", ascending=False, kind="stable")
    error = (valid.stress_probability.to_numpy() >= .5) != valid.label.to_numpy()
    counts = np.arange(1, len(valid)+1)
    stress = valid.label.to_numpy() == 1
    return pd.DataFrame({"coverage": counts / max(1, len(frame[frame.label.isin([0, 1])])),
                         "risk": np.cumsum(error) / counts,
                         "stress_coverage": np.cumsum(stress) / max(1, stress.sum()),
                         "threshold": valid.confidence.to_numpy()})


def reliability(frame, bins=8):
    rows = frame[frame.label.isin([0, 1]) & frame.stress_probability.notna()].copy()
    output = []
    for i in range(bins):
        low, high = i / bins, (i+1) / bins
        group = rows[(rows.stress_probability >= low) & (rows.stress_probability <= high if i == bins-1 else rows.stress_probability < high)]
        output.append({"lower": low, "upper": high, "count": len(group),
                       "mean_probability": float(group.stress_probability.mean()) if len(group) else None,
                       "stress_frequency": float(group.label.mean()) if len(group) else None})
    return pd.DataFrame(output)


def event_metrics(frame):
    required = {"start_seconds", "end_seconds", "session", "event_opened", "subject"}
    if not required.issubset(frame):
        return {"status": "unavailable_without_source_timestamps"}
    if frame.event_opened.isna().any():
        return {"status": "unavailable_without_event_decisions"}
    baseline_seconds, false_events, episodes, detected, delays = 0.0, 0, 0, 0, []
    for _, group in frame.groupby(["subject", "session"], sort=False):
        group = group.sort_values("start_seconds")
        in_episode = False
        episode_start = None
        episode_detected = False
        last_end = None
        for row in group.to_dict("records"):
            contiguous = last_end is not None and abs(row["start_seconds"] - last_end) < 1e-6
            if row["label"] == 1:
                if not in_episode or not contiguous:
                    episodes += 1
                    episode_start, episode_detected = row["start_seconds"], False
                in_episode = True
                if row["event_opened"] and not episode_detected:
                    detected += 1
                    delays.append(row["end_seconds"] - episode_start)
                    episode_detected = True
            else:
                in_episode = False
            if row["label"] == 0:
                baseline_seconds += row["end_seconds"] - row["start_seconds"]
                false_events += int(row["event_opened"])
            last_end = row["end_seconds"]
    return {"status": "laboratory_replay", "eligible_baseline_hours": baseline_seconds/3600,
            "false_event_starts": false_events,
            "false_starts_per_baseline_hour": false_events/(baseline_seconds/3600) if baseline_seconds else None,
            "stress_episodes": episodes, "detected_episodes": detected,
            "event_recall": detected/episodes if episodes else None,
            "median_onset_delay_seconds": float(np.median(delays)) if delays else None}


def subject_intervals(table, seed=42, draws=1000):
    """Bootstrap participant metrics, not individual correlated windows."""
    rng = np.random.default_rng(seed)
    results = {}
    for name in ("balanced_accuracy", "all_window_stress_recall", "coverage", "stress_f1", "selective_error"):
        values = pd.to_numeric(table[name], errors="coerce").dropna().to_numpy() if name in table else np.array([])
        if len(values):
            means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
            results[name] = {"mean": float(values.mean()), "lower_95": float(np.quantile(means, .025)),
                             "upper_95": float(np.quantile(means, .975)), "subjects": len(values)}
    return results
