"""Deterministic held-out waveform perturbations, never feature-number noise."""
import hashlib
import numpy as np

SCENARIOS = {
    "clean": (None, None, 0.),
    "ecg_missing_burst": ("ecg", "missing", .25),
    "ecg_flatline": ("ecg", "flatline", 1.),
    "ecg_clipping": ("ecg", "clipping", .5),
    "ecg_noise_moderate": ("ecg", "noise", .5),
    "ecg_noise_severe": ("ecg", "noise", 2.),
    "ecg_amplitude_scale": ("ecg", "scale", 3.),
    "eda_dropout": ("eda", "dropout", 1.),
    "resp_dropout": ("resp", "dropout", 1.),
    "alternate_dropout": ("alternate", "dropout", 1.),
    "total_dropout": ("all", "dropout", 1.),
}


def corrupt(signals, scenario, row_id, seed=42):
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown corruption scenario: {scenario}")
    channel, kind, severity = SCENARIOS[scenario]
    # Independent stable seeds avoid Python's randomized hash and method-order effects.
    digest = hashlib.sha256(f"{seed}:{row_id}:{scenario}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    out = {name: np.array(values, dtype=float, copy=True) for name, values in signals.items()}
    channels = ["ecg", "eda", "resp"] if channel == "all" else ["eda", "resp"] if channel == "alternate" else [channel]
    for name in channels:
        if name not in out:
            continue
        values = out[name]
        if kind == "dropout":
            out.pop(name)
        elif kind == "flatline":
            out[name] = np.zeros_like(values)
        elif kind == "missing":
            length = max(1, int(len(values) * severity))
            start = int(rng.integers(0, len(values)-length+1))
            values[start:start+length] = np.nan
        elif kind == "noise":
            values += rng.normal(0, max(float(np.nanstd(values)), 1e-8) * severity, values.shape)
        elif kind == "clipping":
            center = np.nanmedian(values)
            width = max(float(np.nanstd(values)), 1e-8) * severity
            out[name] = np.clip(values, center-width, center+width)
        elif kind == "scale":
            out[name] = values * severity
    return out
