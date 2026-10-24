"""Causal duration-based events; unknown observations never count as baseline."""
from dataclasses import dataclass
import math
from .config import EventConfig


class EventTracker:
    def __init__(self, config=EventConfig()):
        self.config = config
        self.reset()

    def reset(self):
        self.key = None
        self.last_end = None
        self.stress_seconds = self.baseline_seconds = 0.0
        self.active = False
        self.cooldown_until = float("-inf")

    def update(self, *, subject, session, start, end, accepted, probability):
        if not all(math.isfinite(v) for v in (start, end)) or not end > start:
            raise ValueError("Source interval must have end > start")
        if accepted and (probability is None or not math.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError("Accepted observations require a finite probability in [0, 1]")
        key = (str(subject), str(session))
        if key != self.key or (self.last_end is not None and abs(start - self.last_end) > 1e-6):
            # Gaps, overlaps, backwards playback, and participant changes all break evidence.
            self.reset()
        self.key, self.last_end = key, end
        opened = closed = False
        if not accepted or probability is None:
            self.stress_seconds = self.baseline_seconds = 0.0
            self.active = False
            return {"event_state": "unknown", "event_opened": False, "event_closed": False}
        duration = end - start
        if probability >= self.config.open_threshold:
            self.stress_seconds += duration
            self.baseline_seconds = 0.0
            if not self.active and self.stress_seconds >= self.config.confirm_seconds and end >= self.cooldown_until:
                self.active, opened = True, True
        elif probability <= self.config.close_threshold:
            self.baseline_seconds += duration
            self.stress_seconds = 0.0
            if self.active and self.baseline_seconds >= self.config.recovery_seconds:
                self.active, closed = False, True
                self.cooldown_until = end + self.config.cooldown_seconds
        else:
            self.stress_seconds = self.baseline_seconds = 0.0
        state = ("stress_event" if self.active else "pending" if self.stress_seconds else
                 "baseline" if probability <= self.config.close_threshold else "watching")
        return {"event_state": state, "event_opened": opened, "event_closed": closed}
