"""Shared schemas. Features never include target or participant identifiers."""
from dataclasses import asdict, dataclass

LEGACY_FEATURES = ["mean", "std", "minimum", "maximum", "median", "skewness", "kurtosis"]
ECG_FEATURES = LEGACY_FEATURES + ["heart_rate", "beat_count", "median_rr", "sdnn", "rmssd"]
EDA_FEATURES = ["eda_mean", "eda_std", "eda_slope", "eda_response_rate", "eda_phasic_std"]
RESP_FEATURES = ["resp_rate", "resp_interval_std", "resp_amplitude_std"]
ALTERNATE_FEATURES = EDA_FEATURES + RESP_FEATURES
EXPERT_FEATURES = {"ecg": ECG_FEATURES, "alternate": ALTERNATE_FEATURES,
                   "fusion": ECG_FEATURES + ALTERNATE_FEATURES}
EXPERT_MODALITIES = {"ecg": ("ecg",), "alternate": ("eda", "resp"), "fusion": ("ecg", "eda", "resp")}
LABEL_NAMES = {0: "Baseline", 1: "Stress"}
SCHEMA_VERSION = 1

@dataclass(frozen=True)
class QualityConfig:
    max_missing: float = 0.01
    max_flatline: float = 0.95
    max_rail_fraction: float = 0.20
    min_seconds: float = 10.0
    min_valid_rr_fraction: float = 0.80
    max_ecg_roughness: float | None = None

    def __post_init__(self):
        if not all(0 <= value <= 1 for value in (self.max_missing, self.max_flatline,
                                                self.max_rail_fraction, self.min_valid_rr_fraction)) or self.min_seconds <= 0:
            raise ValueError("Quality fractions must lie in [0, 1] and minimum duration must be positive")
        if self.max_ecg_roughness is not None and not 0 <= self.max_ecg_roughness <= 1:
            raise ValueError("Maximum ECG roughness must lie in [0, 1]")

@dataclass(frozen=True)
class PolicyConfig:
    baseline_confidence: float = 0.70
    stress_confidence: float = 0.70
    min_coverage: float = 0.80
    min_stress_coverage: float = 0.80
    max_recall_loss: float = 0.03
    # None preserves the original bundle's shared early-exit/acceptance rule.
    early_exit_baseline: float | None = None
    early_exit_stress: float | None = None

    def __post_init__(self):
        if not all(.5 <= value <= 1 for value in (self.baseline_confidence, self.stress_confidence)):
            raise ValueError("Confidence thresholds must lie in [0.5, 1]")
        if any(value is not None and not .5 <= value <= 1
               for value in (self.early_exit_baseline, self.early_exit_stress)):
            raise ValueError("Early-exit thresholds must lie in [0.5, 1]")
        if any(exit_value is not None and exit_value < final_value for exit_value, final_value in
               ((self.early_exit_baseline, self.baseline_confidence),
                (self.early_exit_stress, self.stress_confidence))):
            raise ValueError("Early-exit confidence cannot be lower than final acceptance")
        if not all(0 <= value <= 1 for value in (self.min_coverage, self.min_stress_coverage, self.max_recall_loss)):
            raise ValueError("Coverage and recall constraints must lie in [0, 1]")

@dataclass(frozen=True)
class EventConfig:
    open_threshold: float = 0.70
    close_threshold: float = 0.40
    confirm_seconds: float = 120.0
    recovery_seconds: float = 120.0
    cooldown_seconds: float = 120.0

    def __post_init__(self):
        if not 0 <= self.close_threshold < self.open_threshold <= 1:
            raise ValueError("Event thresholds must satisfy 0 <= close < open <= 1")
        if min(self.confirm_seconds, self.recovery_seconds) <= 0 or self.cooldown_seconds < 0:
            raise ValueError("Event durations must be positive; cooldown cannot be negative")

    def to_dict(self):
        return asdict(self)
