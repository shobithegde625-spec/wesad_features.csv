import pandas as pd
import pytest
from pulsepoint.temporal import EventTracker
from pulsepoint.config import EventConfig
from pulsepoint.metrics import window_metrics, event_metrics, subject_intervals


def update(tracker, start, p=.9, accepted=True, subject="S1"):
    return tracker.update(subject=subject, session="one", start=start, end=start+60, accepted=accepted, probability=p)


def test_duration_confirmation_recovery_and_cooldown():
    t = EventTracker()
    assert not update(t, 0)["event_opened"]
    assert update(t, 60)["event_opened"]
    assert not update(t, 120, .1)["event_closed"]
    assert update(t, 180, .1)["event_closed"]
    assert not update(t, 240)["event_opened"]
    assert update(t, 300)["event_opened"]


def test_gaps_participant_changes_and_unknowns_clear_evidence():
    for second_start, second_subject in [(120, "S1"), (60, "S2"), (0, "S1")]:
        t = EventTracker()
        update(t, 0)
        assert not update(t, second_start, subject=second_subject)["event_opened"]
    t = EventTracker()
    update(t, 0)
    update(t, 60)
    assert update(t, 120, None, False)["event_state"] == "unknown"
    assert not update(t, 180)["event_opened"]


def test_abstention_counts_as_undetected_stress():
    frame = pd.DataFrame({"label": [0, 1, 1], "accepted": [True, True, False], "prediction": [0, 1, None],
                          "stress_probability": [.1, .9, .5]})
    metrics = window_metrics(frame)
    assert metrics["all_window_stress_recall"] == .5
    assert metrics["stress_recall"] == 1
    assert metrics["stress_coverage"] == .5


def test_event_delay_uses_recording_time():
    frame = pd.DataFrame({"subject": ["S1"]*4, "session": ["one"]*4, "label": [0, 0, 1, 1],
                          "start_seconds": [0, 60, 120, 180], "end_seconds": [60, 120, 180, 240],
                          "event_opened": [False, False, False, True]})
    metrics = event_metrics(frame)
    assert metrics["event_recall"] == 1
    assert metrics["median_onset_delay_seconds"] == 120
    assert metrics["false_starts_per_baseline_hour"] == 0
