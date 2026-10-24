"""Shared causal, within-window signal features and conservative quality heuristics.

All samples in the completed trailing window are available at decision time.
No filtering uses samples after that window. These heuristics are not a validated
clinical ECG quality index; waveform overlays and real-data validation are required.
"""
from dataclasses import asdict
from time import perf_counter
import numpy as np
from scipy.signal import butter, find_peaks, sosfilt
from scipy.stats import kurtosis, skew
from .config import QualityConfig, LEGACY_FEATURES


def inspect_quality(values, rate, config=QualityConfig()):
    x = np.asarray(values, dtype=float).reshape(-1) if values is not None else np.array([])
    if rate <= 0:
        raise ValueError("Sampling rate must be positive")
    metrics = {"missing_fraction": 1.0, "flatline_fraction": 1.0, "rail_fraction": 0.0}
    if len(x) < rate * config.min_seconds:
        return {"usable": False, "reason": "missing_or_short", **metrics}
    valid = np.isfinite(x)
    metrics["missing_fraction"] = float(1 - valid.mean())
    finite = x[valid]
    if not len(finite):
        return {"usable": False, "reason": "missing", **metrics}
    scale = max(float(np.ptp(finite)), 1e-12)
    # Adjacent missing samples do not get bridged for flatline counting.
    adjacent = valid[1:] & valid[:-1]
    difference = np.abs(np.diff(x))
    metrics["flatline_fraction"] = float(np.mean(difference[adjacent] <= scale * 1e-8)) if adjacent.any() else 1.0
    # Repeated extrema approximate clipping, but do not assert ADC rail knowledge.
    metrics["rail_fraction"] = float(max(np.mean(finite == finite.min()), np.mean(finite == finite.max())))
    reason = "ok"
    if metrics["missing_fraction"] > config.max_missing:
        reason = "missing"
    elif metrics["flatline_fraction"] > config.max_flatline or np.std(finite) < 1e-12:
        reason = "flatline"
    elif metrics["rail_fraction"] > config.max_rail_fraction:
        reason = "possible_clipping"
    return {"usable": reason == "ok", "reason": reason, **metrics}


def _clean(values):
    x = np.asarray(values, dtype=float).reshape(-1)
    valid = np.isfinite(x)
    if not valid.any():
        raise ValueError("No finite signal samples")
    return np.interp(np.arange(len(x)), np.flatnonzero(valid), x[valid])


def _filter(x, rate, low, high):
    if high >= rate / 2:
        raise ValueError("Sampling rate is too low for the configured signal filter")
    return sosfilt(butter(3, [low, high], fs=rate, btype="bandpass", output="sos"), x)


def legacy_features(values):
    x = _clean(values)
    summaries = [np.mean(x), np.std(x), np.min(x), np.max(x), np.median(x), skew(x), kurtosis(x)]
    return dict(zip(LEGACY_FEATURES, map(float, summaries)))


def ecg_peak_positions(values, rate):
    x = _clean(values)
    filtered = _filter(x, rate, 5, 25)
    # Squared filtered energy handles positive and inverted QRS polarity.
    energy = filtered ** 2
    smooth = np.convolve(energy, np.ones(max(1, int(0.08 * rate))) / max(1, int(0.08 * rate)), mode="same")
    peaks, _ = find_peaks(smooth, distance=int(0.28 * rate), prominence=max(float(np.std(smooth)) * 0.5, 1e-12))
    # Ignore filter startup and end effects for beat interval estimates.
    peaks = peaks[(peaks >= rate) & (peaks < len(x) - rate)]
    return peaks


def ecg_features(values, rate, config=QualityConfig()):
    x = _clean(values)
    result = legacy_features(x)
    peaks = ecg_peak_positions(x, rate)
    rr = np.diff(peaks) / rate
    valid = (rr >= 0.30) & (rr <= 2.0)
    valid_fraction = float(valid.mean()) if len(rr) else 0.0
    if valid.sum() < 3 or valid_fraction < config.min_valid_rr_fraction:
        raise ValueError("unreliable_beats")
    accepted = rr[valid]
    # Successive differences only use adjacent valid RR intervals.
    successive = np.diff(rr)[valid[1:] & valid[:-1]]
    if not len(successive):
        raise ValueError("insufficient_adjacent_beats")
    result.update(heart_rate=float(60 / np.median(accepted)), beat_count=float(len(peaks)),
                  median_rr=float(np.median(accepted)), sdnn=float(np.std(accepted, ddof=1)),
                  rmssd=float(np.sqrt(np.mean(successive ** 2))))
    return result


def eda_features(values, rate):
    x = _clean(values)
    # Downsample to ~10 Hz after a causal anti-aliasing filter.
    low = sosfilt(butter(3, 2.0, fs=rate, output="sos"), x)
    step = max(1, int(rate / 10))
    x = low[::step]
    effective_rate = rate / step
    tonic = sosfilt(butter(2, 0.05, fs=effective_rate, output="sos"), x)
    phasic = x - tonic
    # Relative-prominence peaks are a response proxy, not clinical SCR annotation.
    peaks, _ = find_peaks(phasic, distance=max(1, int(effective_rate)), prominence=max(float(np.std(phasic)) * 0.5, 1e-8))
    duration = len(x) / effective_rate
    return {"eda_mean": float(np.mean(x)), "eda_std": float(np.std(x)),
            "eda_slope": float(np.polyfit(np.arange(len(x)) / effective_rate, x, 1)[0]),
            "eda_response_rate": float(len(peaks) * 60 / duration), "eda_phasic_std": float(np.std(phasic))}


def resp_features(values, rate):
    x = _filter(_clean(values), rate, 0.08, 0.70)
    peaks, _ = find_peaks(x, distance=int(rate * 1.4), prominence=max(float(np.std(x)) * 0.3, 1e-12))
    peaks = peaks[peaks >= rate * 3]
    intervals = np.diff(peaks) / rate
    if len(intervals) < 2 or np.mean((intervals >= 1.4) & (intervals <= 12.5)) < 0.8:
        raise ValueError("unreliable_respiration")
    intervals = intervals[(intervals >= 1.4) & (intervals <= 12.5)]
    return {"resp_rate": float(60 / np.median(intervals)), "resp_interval_std": float(np.std(intervals)),
            "resp_amplitude_std": float(np.std(x))}


EXTRACTORS = {"ecg": ecg_features, "eda": eda_features, "resp": resp_features}


class RawFeatures:
    """Lazy extraction: each modality is computed at most once per window."""
    def __init__(self, signals, rate=700, quality_config=QualityConfig()):
        self.signals, self.rate, self.config = signals, rate, quality_config
        started = perf_counter()
        self.quality = {m: inspect_quality(signals.get(m), rate, quality_config) for m in EXTRACTORS}
        ecg = signals.get("ecg")
        if ecg is not None and self.quality["ecg"]["usable"] and quality_config.max_ecg_roughness is not None:
            x = _clean(ecg)
            # Scale-invariant lag-one roughness: white noise approaches one.
            # A fixed engineering heuristic, not a clinical signal-quality score.
            roughness = float(np.mean(np.diff(x) ** 2) / max(2 * np.var(x), 1e-24))
            self.quality["ecg"]["roughness"] = roughness
            if roughness > quality_config.max_ecg_roughness:
                self.quality["ecg"].update(usable=False, reason="high_frequency_noise")
        self.quality_ms = (perf_counter() - started) * 1000
        self.feature_ms = 0.0
        self.cache = {}
        self.attempted = set()

    def get(self, modalities, enforce_quality=True):
        result = {}
        for modality in modalities:
            if enforce_quality and not self.quality[modality]["usable"]:
                return None
            if modality not in self.attempted:
                self.attempted.add(modality)
                started = perf_counter()
                try:
                    if modality == "ecg":
                        values = ecg_features(self.signals.get(modality), self.rate, self.config)
                    else:
                        values = EXTRACTORS[modality](self.signals.get(modality), self.rate)
                    if not all(np.isfinite(v) for v in values.values()):
                        raise ValueError("nonfinite_features")
                    self.cache[modality] = values
                except (ValueError, TypeError, IndexError) as exc:
                    self.quality[modality].update(usable=False, reason=str(exc))
                    self.cache[modality] = None
                self.feature_ms += (perf_counter() - started) * 1000
            if self.cache[modality] is None:
                return None
            result.update(self.cache[modality])
        return result

    def all_features(self):
        result = {}
        # Preserve the original amplitude-only experiment even when beat extraction
        # fails. Physiological experts still require the independent quality flag.
        ecg = self.signals.get("ecg")
        if ecg is not None and np.isfinite(ecg).all() and np.std(ecg) > 1e-12:
            result.update(legacy_features(ecg))
        for modality in EXTRACTORS:
            values = self.get((modality,))
            if values:
                result.update(values)
        for modality, quality in self.quality.items():
            for name, value in quality.items():
                result[f"{modality}_{name}"] = value
        acc = self.signals.get("acc")
        if acc is not None:
            result["movement_std"] = float(np.nanstd(np.linalg.norm(np.asarray(acc), axis=-1)))
        return result


class StoredFeatures:
    """Stored rows cannot provide waveform-quality evidence if annotations are absent."""
    def __init__(self, row):
        self.row = row
        self.quality_ms = self.feature_ms = 0.0
        self.quality = {}
        for modality in EXTRACTORS:
            value = row.get(f"{modality}_usable")
            if value is None or (isinstance(value, float) and np.isnan(value)):
                usable, reason = True, "unverified_stored_features"
            else:
                usable = value is True or str(value).lower() in ("true", "1", "1.0")
                reason = str(row.get(f"{modality}_reason", "ok" if usable else "quality_failed"))
            self.quality[modality] = {"usable": usable, "reason": reason}

    def get(self, modalities, enforce_quality=True):
        if enforce_quality and any(not self.quality[m]["usable"] for m in modalities):
            return None
        return self.row
