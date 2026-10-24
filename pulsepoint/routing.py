"""Label-free routing with explicit missing-modality handling."""
from dataclasses import asdict, dataclass
from time import perf_counter
import numpy as np
from .config import PolicyConfig, EXPERT_MODALITIES
from .features import StoredFeatures


@dataclass
class Decision:
    accepted: bool
    prediction: int | None
    stress_probability: float | None
    route: str
    reason: str
    quality: dict
    quality_ms: float
    feature_ms: float
    inference_ms: float
    total_ms: float
    artifact_id: str
    escalated: bool = False

    def to_dict(self):
        return asdict(self)


def accepted_probability(probability, policy, early_exit=False):
    if probability is None or not np.isfinite(probability):
        return False
    low, high = policy.baseline_confidence, policy.stress_confidence
    if early_exit:
        low = policy.early_exit_baseline if policy.early_exit_baseline is not None else low
        high = policy.early_exit_stress if policy.early_exit_stress is not None else high
    return probability >= high if probability >= 0.5 else (1 - probability) >= low


def route(provider, experts, policy=PolicyConfig(), artifact_id="unversioned", mode="cascade"):
    if mode not in ("cascade", "confidence_only", "no_abstention", "fusion", "ecg"):
        raise ValueError(f"Unknown routing mode: {mode}")
    started = perf_counter()
    inference_ms = 0.0
    last_probability, last_route, reason = None, "none", "no_usable_expert"
    escalated = False
    enforce_quality = mode != "confidence_only"

    def run(name):
        nonlocal inference_ms, last_probability, last_route, reason
        if name not in experts:
            return None
        values = provider.get(EXPERT_MODALITIES[name], enforce_quality=enforce_quality)
        if values is None:
            reason = "quality_failed_or_missing_modality"
            return None
        before = perf_counter()
        probability = experts[name].probability(values)
        inference_ms += (perf_counter() - before) * 1000
        if probability is not None and np.isfinite(probability) and 0 <= probability <= 1:
            last_probability, last_route, reason = probability, name, "uncertain"
        else:
            reason = "missing_or_nonfinite_features"
            probability = None
        return probability

    def finish(accepted):
        return Decision(accepted=accepted, prediction=int(last_probability >= .5) if accepted else None,
                        stress_probability=last_probability, route=last_route, reason="accepted" if accepted else reason,
                        quality=provider.quality, quality_ms=provider.quality_ms, feature_ms=provider.feature_ms,
                        inference_ms=inference_ms, total_ms=(perf_counter() - started)*1000 + provider.quality_ms,
                        artifact_id=artifact_id, escalated=escalated)

    if mode == "fusion":
        p = run("fusion")
        # Static baseline emits a class for every usable full-modality window.
        return finish(p is not None)
    p = run("ecg")
    if p is not None and (mode == "ecg" or accepted_probability(p, policy, early_exit=True)):
        return finish(True)
    if mode == "ecg":
        return finish(False)
    # Uncertain ECG uses fusion; failed ECG uses an independently trained alternate.
    if p is not None:
        escalated = True
        candidate = run("fusion")
    else:
        escalated = True
        candidate = run("alternate")
    if candidate is not None and (mode == "no_abstention" or accepted_probability(candidate, policy)):
        return finish(True)
    # A request for more evidence is separate from the final acceptance rule.
    if (candidate is None and p is not None and policy.early_exit_baseline is not None
            and accepted_probability(p, policy)):
        decision = finish(True)
        decision.reason = "accepted_ecg_fusion_unavailable"
        return decision
    if mode == "no_abstention" and last_probability is not None:
        return finish(True)
    return finish(False)


def _operating_metrics(y, decisions):
    accepted = np.array([d.accepted for d in decisions])
    predictions = np.array([d.prediction if d.accepted else -1 for d in decisions])
    y = np.asarray(y, dtype=int)
    return {"coverage": float(accepted.mean()),
            "stress_coverage": float(accepted[y == 1].mean()),
            "stress_recall": float(np.mean(predictions[y == 1] == 1)),
            "errors": float(np.mean((predictions != y) & accepted)),
            "escalation": float(np.mean([d.escalated for d in decisions]))}


def tune_policy_legacy(frame, experts, config=PolicyConfig()):
    y = frame.label.to_numpy(dtype=int)
    if set(y) != {0, 1}:
        raise ValueError("Policy development subjects need both target classes")
    rows = frame.to_dict("records")
    # Cache expert outputs across threshold sweeps; no estimator fitting here.
    class CachedExpert:
        def __init__(self, expert):
            self.expert, self.cache = expert, {}
        def probability(self, values):
            key = values["row_id"]
            if key not in self.cache:
                self.cache[key] = self.expert.probability(values)
            return self.cache[key]
    cached = {k: CachedExpert(v) for k, v in experts.items()}
    baseline = [route(StoredFeatures(row), cached, PolicyConfig(.5, .5), mode="no_abstention") for row in rows]
    reference = _operating_metrics(y, baseline)
    clean_mask = frame.development_scenario.eq("clean").to_numpy() if "development_scenario" in frame else np.ones(len(frame), dtype=bool)
    clean_reference = _operating_metrics(y[clean_mask], [d for d, clean in zip(baseline, clean_mask) if clean])
    records = []
    for low in (.5, .6, .7, .8, .9):
        for high in (.5, .6, .7, .8, .9):
            policy = PolicyConfig(low, high, config.min_coverage, config.min_stress_coverage, config.max_recall_loss)
            decisions = [route(StoredFeatures(row), cached, policy) for row in rows]
            metrics = _operating_metrics(y, decisions)
            clean_metrics = _operating_metrics(y[clean_mask], [d for d, clean in zip(decisions, clean_mask) if clean])
            feasible = (metrics["coverage"] >= config.min_coverage and metrics["stress_coverage"] >= config.min_stress_coverage
                        and metrics["stress_recall"] >= reference["stress_recall"] - config.max_recall_loss
                        and clean_metrics["coverage"] >= config.min_coverage
                        and clean_metrics["stress_coverage"] >= config.min_stress_coverage
                        and clean_metrics["stress_recall"] >= clean_reference["stress_recall"] - config.max_recall_loss)
            records.append({"baseline_confidence": low, "stress_confidence": high, "feasible": feasible,
                            "clean": clean_metrics, **metrics})
    eligible = [r for r in records if r["feasible"]]
    if not eligible:
        chosen = PolicyConfig(.5, .5, config.min_coverage, config.min_stress_coverage, config.max_recall_loss)
        status = "coverage_constraints_infeasible_due_to_quality_or_missing_data"
    else:
        winner = min(eligible, key=lambda r: (r["errors"], r["escalation"], -r["coverage"]))
        chosen = PolicyConfig(winner["baseline_confidence"], winner["stress_confidence"],
                              config.min_coverage, config.min_stress_coverage, config.max_recall_loss)
        status = "selected_on_development_subjects"
    return chosen, {"status": status, "reference": reference, "candidates": records}


def tune_policy(frame, experts, config=PolicyConfig()):
    """Minimize escalation subject to development-only fusion-relative constraints.

    Early exit and final acceptance are independent. Probabilities are computed
    once; all threshold candidates use exactly the runtime's quality contract.
    No timing from stored features is used as an edge-compute measurement.
    """
    if "fusion" not in experts:
        return tune_policy_legacy(frame, experts, config)
    from itertools import product
    y = frame.label.to_numpy(dtype=int)
    if set(y) != {0, 1}:
        raise ValueError("Policy development subjects need both target classes")
    rows = frame.to_dict("records")
    probabilities = {}
    for name, expert in experts.items():
        outputs = []
        for row in rows:
            values = StoredFeatures(row).get(EXPERT_MODALITIES[name])
            p = expert.probability(values) if values is not None else None
            outputs.append(p if p is not None and np.isfinite(p) and 0 <= p <= 1 else np.nan)
        probabilities[name] = np.asarray(outputs, dtype=float)
    ecg = probabilities["ecg"]
    fallback = np.where(np.isfinite(ecg), probabilities["fusion"], probabilities["alternate"])
    fallback = np.where(np.isfinite(fallback), fallback, ecg)
    # Failure-resilient full-evidence reference; clean full-modality rows use fusion.
    reference_p = np.where(np.isfinite(fallback), fallback, ecg)
    clean = frame.development_scenario.eq("clean").to_numpy() if "development_scenario" in frame else np.ones(len(frame), dtype=bool)
    subjects = frame.subject.to_numpy()

    def metrics(p, accepted, mask):
        prediction = np.where(accepted, p >= .5, -1)
        stress = mask & (y == 1)
        participant_recalls = [float(np.mean(prediction[s] == 1)) for subject in np.unique(subjects[mask])
                               for s in [stress & (subjects == subject)] if s.any()]
        return {"coverage": float(accepted[mask].mean()),
                "stress_coverage": float(accepted[stress].mean()),
                "stress_recall": float(np.mean(prediction[stress] == 1)),
                "macro_stress_recall": float(np.mean(participant_recalls)),
                # Abstention is an operational error here, so rejection cannot hide mistakes.
                "operational_error": float(np.mean(prediction[mask] != y[mask]))}

    all_rows = np.ones(len(frame), dtype=bool)
    references = {"all": metrics(reference_p, np.isfinite(reference_p), all_rows),
                  "clean": metrics(reference_p, np.isfinite(reference_p), clean)}
    records = []
    for low, high, acceptance in product((.5, .6, .7, .8, .9, .95, .99, 1.),
                                          (.5, .6, .7, .8, .9, .95, .99, 1.), (.5, .6, .7)):
        if min(low, high) < acceptance:
            continue
        early = np.isfinite(ecg) & (((ecg < .5) & (1-ecg >= low)) | ((ecg >= .5) & (ecg >= high)))
        p = np.where(early, ecg, fallback)
        accepted = np.isfinite(p) & (np.maximum(p, 1-p) >= acceptance)
        overall, clean_metrics = metrics(p, accepted, all_rows), metrics(p, accepted, clean)
        feasible = True
        for current, reference in ((overall, references["all"]), (clean_metrics, references["clean"])):
            feasible &= (current["coverage"] >= config.min_coverage
                         and current["stress_coverage"] >= config.min_stress_coverage
                         and current["stress_recall"] >= reference["stress_recall"] - config.max_recall_loss
                         and current["macro_stress_recall"] >= reference["macro_stress_recall"] - config.max_recall_loss
                         and current["operational_error"] <= reference["operational_error"] + config.max_recall_loss)
        records.append({"early_exit_baseline": low, "early_exit_stress": high,
                        "final_acceptance": acceptance, "feasible": bool(feasible),
                        "escalation": float((~early).mean()), "clean": clean_metrics, **overall})
    eligible = [r for r in records if r["feasible"]]
    if eligible:
        winner = min(eligible, key=lambda r: (r["escalation"], r["operational_error"], -r["coverage"]))
        chosen = PolicyConfig(winner["final_acceptance"], winner["final_acceptance"],
                              config.min_coverage, config.min_stress_coverage, config.max_recall_loss,
                              winner["early_exit_baseline"], winner["early_exit_stress"])
        status = "selected_on_development_subjects_fusion_relative_constraints"
    else:
        chosen = PolicyConfig(.5, .5, config.min_coverage, config.min_stress_coverage, config.max_recall_loss, 1., 1.)
        status = "constraints_infeasible_full_evidence_fallback"
    return chosen, {"status": status, "reference": references, "candidates": records,
                    "objective": "minimum escalation with pooled/macro recall and operational-error constraints",
                    "cost_proxy": "escalation fraction; CPU saving requires separate waveform measurement"}
