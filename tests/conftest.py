import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
from pulsepoint.config import LEGACY_FEATURES


@pytest.fixture
def synthetic_signals():
    rate = 700
    t = np.arange(30*rate) / rate
    rng = np.random.default_rng(123)
    phase = np.minimum(t % .8, .8 - t % .8)
    ecg = np.exp(-.5*(phase/.015)**2) + rng.normal(0, .002, len(t))
    return {"ecg": ecg, "eda": 1.5 + .002*t + .05*np.sin(2*np.pi*.12*t),
            "resp": np.sin(2*np.pi*.25*t), "acc": np.zeros((len(t), 3))}


@pytest.fixture
def grouped_frame():
    rng = np.random.default_rng(12)
    rows = []
    for subject in range(9):
        for label in (0, 1):
            for j in range(5):
                values = rng.normal(label*.7, .6, len(LEGACY_FEATURES))
                rows.append({**dict(zip(LEGACY_FEATURES, values)), "label": label,
                             "subject": f"S{subject}", "row_id": f"S{subject}-{label}-{j}"})
    return pd.DataFrame(rows)
