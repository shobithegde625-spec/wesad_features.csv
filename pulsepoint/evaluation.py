"""Nested subject-held-out experiments with auditable splits and predictions."""
from pathlib import Path
import json
import uuid
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from threadpoolctl import threadpool_limits
from .config import LEGACY_FEATURES, EventConfig
from .data import RawSource, environment, file_hash, write_json, code_hash
from .models import model_factory
from .metrics import window_metrics, event_metrics, reliability, risk_coverage, subject_intervals
from .training import build_bundle, save_bundle
from .replay import Runtime, chronological, source_interval
from .corruption import SCENARIOS, corrupt


def _baseline(train, test, family, seed, fold, source=None, scenarios=("clean",), output=None, evaluation_id="evaluation"):
    from time import perf_counter
    from .features import legacy_features
    finite = np.isfinite(train[LEGACY_FEATURES].to_numpy(dtype=float)).all(axis=1)
    fit = train.loc[finite]
    if set(fit.label.unique()) != {0, 1}:
        raise ValueError("Baseline training needs both classes after excluding invalid summaries")
    estimator = model_factory(family, seed).fit(fit[LEGACY_FEATURES], fit.label.astype(int))
    artifact_id = f"{evaluation_id}-fold-{fold}-legacy-{family}"
    if output is not None:
        target = Path(output) / "models" / f"fold-{fold}-legacy-{family}"
        target.mkdir(parents=True, exist_ok=False)
        joblib.dump({"model": estimator, "features": LEGACY_FEATURES, "artifact_id": artifact_id}, target / "model.joblib", compress=3)
        write_json(target / "metadata.json", {"artifact_id": artifact_id, "parameters": estimator.get_params(deep=True),
                   "training_subjects": sorted(fit.subject.unique()), "seed": seed, "source_code_sha256": code_hash()})
    rows = []
    for scenario in scenarios:
        for row in test.to_dict("records"):
            signals, rate = source.window(row) if source else (None, 700)
            if source:
                signals = corrupt(signals, scenario, row["row_id"], seed)
            started = perf_counter()
            values = row
            if source:
                ecg = signals.get("ecg")
                values = legacy_features(ecg) if ecg is not None and np.isfinite(ecg).all() and np.std(ecg) > 1e-12 else {}
            sample = pd.DataFrame([{f: values.get(f, np.nan) for f in LEGACY_FEATURES}])
            accepted = bool(np.isfinite(sample.to_numpy(dtype=float)).all())
            prediction, probability = None, None
            if accepted:
                prediction = int(estimator.predict(sample)[0])
                if hasattr(estimator, "predict_proba"):
                    probability = float(estimator.predict_proba(sample)[0, list(estimator.classes_).index(1)])
            rows.append({**row, "fold": fold, "method": f"legacy_{family}", "scenario": scenario,
                         "artifact_id": artifact_id, "accepted": accepted,
                         "prediction": prediction, "stress_probability": probability,
                         "route": "legacy_ecg", "reason": "accepted" if accepted else "nonfinite_features",
                         "escalated": False, "total_ms": (perf_counter()-started)*1000})
    return rows


def evaluate(frame, data_path, output, *, protocol="loso", seed=42, families=("logistic", "svm", "forest"),
             baseline_only=False, raw_root=None, scenarios=("clean",), replay_frame=None, replay_path=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError("Evaluation output already contains a report")
    if any(s not in SCENARIOS for s in scenarios):
        raise ValueError("Unknown corruption scenario")
    if scenarios != ("clean",) and not raw_root:
        raise ValueError("Waveform corruption evaluation requires --raw-root; stored feature noise is not a substitute")
    source = RawSource(raw_root) if raw_root else None
    evaluation_id = uuid.uuid4().hex[:12]
    labeled = frame[frame.label.isin([0, 1])].reset_index(drop=True)
    if protocol not in ("loso", "group5"):
        raise ValueError("Protocol must be loso or group5")
    if set(labeled.label.unique()) != {0, 1} or labeled.subject.nunique() < 2:
        raise ValueError("Evaluation needs both classes and at least two subjects")
    if replay_frame is not None and set(replay_frame.subject) != set(labeled.subject):
        raise ValueError("Replay and classification datasets must contain the same subjects")
    if source:
        required = {"source_file", "sampling_rate", "start_sample", "end_sample"}
        for dataset in (frame, replay_frame if replay_frame is not None else frame):
            if not required.issubset(dataset):
                raise ValueError("Waveform evaluation requires metadata from prepare_dataset.py")
    if not baseline_only and labeled.subject.nunique() < 7:
        raise ValueError("Nested subject evaluation requires at least seven total subjects")
    cv = LeaveOneGroupOut() if protocol == "loso" else GroupKFold(n_splits=min(5, labeled.subject.nunique()))
    predictions, manifests = [], []
    for fold, (fit_idx, test_idx) in enumerate(cv.split(labeled, labeled.label, labeled.subject), 1):
        development, binary_test = labeled.iloc[fit_idx].copy(), labeled.iloc[test_idx].copy()
        test_subjects = sorted(binary_test.subject.unique())
        all_test = replay_frame if replay_frame is not None else frame
        test = chronological(all_test[all_test.subject.isin(test_subjects)])
        assert not set(development.subject) & set(test.subject)
        print(f"Fold {fold}: held out {', '.join(test_subjects)}", flush=True)
        manifest = {"fold": fold, "outer_train_subjects": sorted(development.subject.unique()), "outer_test_subjects": test_subjects}
        with threadpool_limits(limits=1):
            for family in families:
                predictions.extend(_baseline(development, test, family, seed, fold, source, scenarios, output, evaluation_id))
            if not baseline_only:
                bundle = build_bundle(development, seed, families, f"{evaluation_id}-fold-{fold}-cascade", raw_source=source)
                save_bundle(bundle, output / "models" / f"fold-{fold}-cascade", data_path)
                manifest.update(roles=bundle["manifest"], selections=bundle["selections"], policy=bundle["policy"],
                                policy_selection=bundle["policy_selection"], events=bundle["events"], event_selection=bundle["event_selection"],
                                artifact_id=bundle["artifact_id"], artifact_path=f"models/fold-{fold}-cascade/model.joblib")
                assert all(not set(ids) & set(test_subjects) for ids in bundle["manifest"].values())
                modes = ["ecg", "confidence_only", "cascade", "no_abstention"]
                if bundle["multimodal"]:
                    modes.append("fusion")
                for scenario in scenarios:
                    runtimes = {mode: Runtime(bundle, mode) for mode in modes}
                    # Same decision rule with a one-window event policy isolates temporal confirmation.
                    runtimes["cascade_single_window"] = Runtime(bundle, "cascade", EventConfig(
                        open_threshold=bundle["events"]["open_threshold"], confirm_seconds=1., recovery_seconds=1., cooldown_seconds=0.))
                    for row in test.to_dict("records"):
                        signals, rate = (source.window(row) if source else (None, 700))
                        if source:
                            signals = corrupt(signals, scenario, row["row_id"], seed)
                        for method, runtime in runtimes.items():
                            result = runtime.predict(signals=signals, rate=rate, features=None if source else row, interval=source_interval(row))
                            # Flatten quality to JSON for portable CSV export.
                            result["quality"] = json.dumps(result["quality"], sort_keys=True)
                            predictions.append({**row, **result, "fold": fold, "method": method, "scenario": scenario})
        manifests.append(manifest)
        # Checkpoint after each complete fold to make long raw runs inspectable.
        write_json(output / "split_manifest.json", manifests)
    result = pd.DataFrame(predictions)
    result.to_csv(output / "predictions.csv", index=False)
    summaries, participant_rows = [], []
    for (method, scenario), group in result.groupby(["method", "scenario"], sort=False):
        per_subject = []
        for subject, subject_frame in group.groupby("subject"):
            row = {"method": method, "scenario": scenario, "subject": subject, **window_metrics(subject_frame)}
            per_subject.append(row)
            participant_rows.append(row)
        summary = {"method": method, "scenario": scenario, "pooled": window_metrics(group),
                   "participant_macro_intervals": subject_intervals(pd.DataFrame(per_subject), seed),
                   "events": event_metrics(group)}
        summaries.append(summary)
        prefix = output / f"{method}-{scenario}"
        risk_coverage(group).to_csv(f"{prefix}-risk-coverage.csv", index=False)
        reliability(group).to_csv(f"{prefix}-reliability.csv", index=False)
        for expert, routed in group.groupby("route"):
            reliability(routed).to_csv(f"{prefix}-{expert}-routed-reliability.csv", index=False)
    pd.DataFrame(participant_rows).to_csv(output / "per_subject.csv", index=False)
    comparison = pd.DataFrame({"method": r["method"], "scenario": r["scenario"], **r["pooled"]} for r in summaries)
    comparison.to_csv(output / "comparison.csv", index=False)
    # Paired differences use one observation per participant and method.
    from .reporting import paired_comparisons, plot_report, write_report, tradeoff_diagnostics
    pairs = paired_comparisons(pd.DataFrame(participant_rows), seed)
    write_json(output / "paired_differences.json", pairs)
    report = {"evaluation_id": evaluation_id, "protocol": protocol, "seed": seed, "dataset_sha256": file_hash(data_path),
              "source_code_sha256": code_hash(),
              "subjects": int(labeled.subject.nunique()),
              "training_labeled_windows": len(labeled),
              "windows": int(result.drop_duplicates("row_id").label.isin([0, 1]).sum()),
              "replay_sha256": file_hash(replay_path) if replay_path else None,
              "waveform_processing_measured": bool(source), "multimodal": "fusion" in result.method.values,
              "environment": environment(), "results": summaries,
              "selection_rule": "inner grouped model selection, disjoint sigmoid calibration, policy-subject threshold search",
              "limitations": ["15-person laboratory data does not establish real-world alert rates",
                              "Stored-feature timings exclude feature extraction" if not source else "Energy and sensor acquisition savings are not measured",
                              "Outer-fold estimates assess the training procedure, not the final deployment artifact"]}
    write_json(output / "summary.json", report)
    from .targets import assess_targets
    write_json(output / "targets.json", assess_targets(report))
    tradeoff_diagnostics(result, output)
    plot_report(result, output)
    write_report(report, comparison, output, pairs)
    return report
