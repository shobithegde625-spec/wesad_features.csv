import numpy as np
import pytest
from pulsepoint.config import ECG_FEATURES
from pulsepoint.features import inspect_quality, RawFeatures, ecg_features, resp_features


def test_known_heart_and_respiration_rate(synthetic_signals):
    ecg = ecg_features(synthetic_signals["ecg"], 700)
    respiration = resp_features(synthetic_signals["resp"], 700)
    assert ecg["heart_rate"] == pytest.approx(75, abs=3)
    assert respiration["resp_rate"] == pytest.approx(15, abs=2)
    assert all(np.isfinite(list(ecg.values())))


def test_waveform_failure_is_not_imputed_as_normal():
    for values in (None, np.full(21000, np.nan), np.zeros(21000)):
        quality = inspect_quality(values, 700)
        assert not quality["usable"]
    assert not inspect_quality(np.tile([-1., 1.], 10500), 700)["usable"]


def test_lazy_extraction_reuses_features(synthetic_signals):
    provider = RawFeatures(synthetic_signals)
    ecg = provider.get(("ecg",))
    assert set(ecg) == set(ECG_FEATURES)
    assert provider.attempted == {"ecg"}
    before = provider.feature_ms
    again = provider.get(("ecg",))
    assert ecg == again and provider.feature_ms == before
    assert provider.get(("eda", "resp")) is not None
    assert provider.attempted == {"ecg", "eda", "resp"}


def test_raw_and_offline_features_agree(synthetic_signals):
    offline = RawFeatures(synthetic_signals).all_features()
    online = RawFeatures(synthetic_signals).get(("ecg", "eda", "resp"))
    assert online is not None
    for feature, value in online.items():
        assert value == offline[feature]


def test_noise_gate_is_scale_invariant_and_routes_to_alternate(synthetic_signals):
    from pulsepoint.config import QualityConfig
    from pulsepoint.corruption import corrupt
    config = QualityConfig(max_ecg_roughness=.5)
    clean = RawFeatures(synthetic_signals, quality_config=config)
    scaled = RawFeatures(corrupt(synthetic_signals, "ecg_amplitude_scale", "test"), quality_config=config)
    noisy = RawFeatures(corrupt(synthetic_signals, "ecg_noise_severe", "test"), quality_config=config)
    assert clean.quality["ecg"]["usable"] and scaled.quality["ecg"]["usable"]
    assert clean.quality["ecg"]["roughness"] == pytest.approx(scaled.quality["ecg"]["roughness"])
    assert noisy.quality["ecg"]["reason"] == "high_frequency_noise"
    assert noisy.get(("ecg",)) is None
    assert noisy.get(("eda", "resp")) is not None
