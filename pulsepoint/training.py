"""Versioned bundles; the shipped model keeps fit/calibration/policy roles disjoint."""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import uuid
import joblib
from .config import SCHEMA_VERSION, EventConfig, QualityConfig
from .data import file_hash, environment, write_json, code_hash
from .models import fit_experts
from .routing import tune_policy


def build_bundle(frame, seed=42, families=("logistic", "svm", "forest"), artifact_id=None, raw_source=None):
    experts, policy_data, manifest, selections = fit_experts(frame, seed, families)
    quality_config = QualityConfig(max_ecg_roughness=.5)
    from .corruption import corrupt
    from .features import RawFeatures
    from .config import EXPERT_FEATURES
    import pandas as pd
    import numpy as np
    tuning_data = policy_data.copy()
    tuning_data["development_scenario"] = "clean"
    development_scenarios = []
    if raw_source is not None:
        development_scenarios = ["ecg_missing_burst", "ecg_noise_moderate", "eda_dropout"]
        augmented = []
        for row in policy_data.to_dict("records"):
            signals, rate = raw_source.window(row)
            for scenario in development_scenarios:
                features = RawFeatures(corrupt(signals, scenario, row["row_id"], seed), rate, quality_config).all_features()
                # Never retain clean features for a modality that failed under corruption.
                blank = {name: np.nan for name in EXPERT_FEATURES["fusion"]}
                augmented.append({**row, **blank, **features, "development_scenario": scenario,
                                  "row_id": f"{row['row_id']}-development-{scenario}"})
        tuning_data = pd.concat([tuning_data, pd.DataFrame(augmented)], ignore_index=True)
    policy, policy_report = tune_policy(tuning_data, experts)
    policy_report["development_corruptions"] = development_scenarios
    # Event tuning is possible only with actual temporal metadata, never CSV row numbers.
    from .replay import tune_events
    events, event_report = tune_events(policy_data, experts, policy)
    return {"schema_version": SCHEMA_VERSION, "artifact_id": artifact_id or uuid.uuid4().hex[:12],
            "experts": experts, "policy": asdict(policy), "events": asdict(events),
            "quality_config": asdict(quality_config), "multimodal": "fusion" in experts,
            "training_revision": "v2-separated-exit-acceptance-grouped-search",
            "manifest": manifest, "selections": selections,
            "policy_selection": policy_report, "event_selection": event_report,
            "seed": seed, "environment": environment(),
            "source_code_sha256": code_hash(),
            "created_at": datetime.now(timezone.utc).isoformat()}


def new_run(parent, prefix):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = Path(parent) / f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"
    target.mkdir(parents=True, exist_ok=False)
    return target


def save_bundle(bundle, output, data_path):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "model.joblib"
    if path.exists():
        raise FileExistsError(f"Refusing to replace {path}")
    bundle["dataset_sha256"] = file_hash(data_path)
    bundle["dataset_path"] = str(Path(data_path).resolve())
    joblib.dump(bundle, path, compress=3)
    metadata = {k: v for k, v in bundle.items() if k != "experts"}
    metadata["expert_features"] = {k: v.features for k, v in bundle["experts"].items()}
    metadata["estimator_parameters"] = {k: v.estimator.get_params(deep=True) for k, v in bundle["experts"].items()}
    metadata["bundle_bytes"] = path.stat().st_size
    write_json(output / "metadata.json", metadata)
    return path


def load_bundle(path):
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("schema_version") != SCHEMA_VERSION or "experts" not in bundle:
        raise ValueError("Expected a versioned Pulsepoint bundle. Train one with train_model.py; the legacy model is preserved separately.")
    return bundle
