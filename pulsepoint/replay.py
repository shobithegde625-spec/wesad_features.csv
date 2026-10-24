"""One runtime shared by CLI, dashboard, experiments and benchmarks."""
from time import perf_counter, process_time
import numpy as np
from .config import EventConfig, PolicyConfig, QualityConfig
from .features import RawFeatures, StoredFeatures
from .routing import route
from .temporal import EventTracker


class Runtime:
    def __init__(self, bundle, mode="cascade", event_config=None):
        self.bundle, self.mode = bundle, mode
        self.policy = PolicyConfig(**bundle["policy"])
        self.events = EventTracker(event_config or EventConfig(**bundle["events"]))
        self.quality_config = QualityConfig(**bundle.get("quality_config", {}))

    def reset(self):
        self.events.reset()

    def predict(self, *, signals=None, rate=700, features=None, interval=None):
        if (signals is None) == (features is None):
            raise ValueError("Provide either waveform signals or stored features")
        started, cpu_started = perf_counter(), process_time()
        provider = RawFeatures(signals, rate, self.quality_config) if signals is not None else StoredFeatures(features)
        result = route(provider, self.bundle["experts"], self.policy, self.bundle["artifact_id"], self.mode).to_dict()
        if interval is not None:
            result.update(self.events.update(**interval, accepted=result["accepted"], probability=result["stress_probability"]))
        else:
            self.events.reset()
            result.update(event_state="unavailable_without_source_time", event_opened=False, event_closed=False)
        result["total_ms"] = (perf_counter() - started) * 1000
        result["cpu_ms"] = (process_time() - cpu_started) * 1000
        result["input_kind"] = "waveform" if signals is not None else "stored_features"
        return result


def source_interval(row):
    if not all(k in row and row[k] is not None for k in ("subject", "session", "start_seconds", "end_seconds")):
        return None
    if not np.isfinite([row["start_seconds"], row["end_seconds"]]).all():
        return None
    return {"subject": str(row["subject"]), "session": str(row["session"]),
            "start": float(row["start_seconds"]), "end": float(row["end_seconds"])}


def chronological(frame):
    keys = [k for k in ("subject", "session", "start_seconds", "row_id") if k in frame]
    return frame.sort_values(keys, kind="stable")


def tune_events(frame, experts, policy):
    from .metrics import event_metrics
    import pandas as pd
    if not {"start_seconds", "end_seconds", "session"}.issubset(frame):
        return EventConfig(), {"status": "not_tuned_no_source_timestamps"}
    # Cache model outputs once; only the causal state machine varies in this sweep.
    decisions = []
    for row in chronological(frame).to_dict("records"):
        d = route(StoredFeatures(row), experts, policy).to_dict()
        decisions.append({**row, **d})
    candidates = []
    from itertools import product
    for confirm, open_threshold, recovery, cooldown in product((60., 120., 180.), (.6, .7, .8), (60., 120.), (0., 120.)):
        config = EventConfig(open_threshold=open_threshold, confirm_seconds=confirm,
                             recovery_seconds=recovery, cooldown_seconds=cooldown)
        tracker = EventTracker(config)
        rows = []
        for row in decisions:
            event = tracker.update(**source_interval(row), accepted=row["accepted"], probability=row["stress_probability"])
            rows.append({**row, **event})
        metrics = event_metrics(pd.DataFrame(rows))
        candidates.append({"config": config.to_dict(), "metrics": metrics})
    # Maximize event recall, then minimize false starts and delay. Test labels never enter.
    winner = min(candidates, key=lambda c: (-(c["metrics"]["event_recall"] or 0),
                                            c["metrics"]["false_event_starts"],
                                            c["metrics"]["median_onset_delay_seconds"] or float("inf")))
    return EventConfig(**winner["config"]), {"status": "selected_on_policy_subjects", "candidates": candidates}
