import pytest
from pulsepoint.features import StoredFeatures, RawFeatures
from pulsepoint.routing import route
from pulsepoint.config import PolicyConfig


class ConstantExpert:
    def __init__(self, probability):
        self.p = probability
        self.calls = 0
    def probability(self, row):
        self.calls += 1
        return self.p


def experts(ecg=.9, alternate=.9, fusion=.9):
    return {name: ConstantExpert(value) for name, value in [("ecg", ecg), ("alternate", alternate), ("fusion", fusion)]}


def test_confident_ecg_skips_expensive_experts():
    models = experts()
    decision = route(StoredFeatures({}), models)
    assert decision.accepted and decision.route == "ecg"
    assert models["fusion"].calls == models["alternate"].calls == 0


def test_uncertain_ecg_uses_fusion():
    decision = route(StoredFeatures({}), experts(ecg=.55, fusion=.8))
    assert decision.accepted and decision.route == "fusion" and decision.escalated


def test_ecg_failure_uses_independent_alternate():
    decision = route(StoredFeatures({"ecg_usable": False}), experts())
    assert decision.accepted and decision.route == "alternate"


def test_total_dropout_abstains():
    row = {f"{m}_usable": False for m in ("ecg", "eda", "resp")}
    decision = route(StoredFeatures(row), experts())
    assert not decision.accepted and decision.prediction is None
    assert decision.stress_probability is None


def test_uncertain_alternate_abstains():
    decision = route(StoredFeatures({"ecg_usable": False}), experts(alternate=.55))
    assert not decision.accepted and decision.route == "alternate"


def test_invalid_model_probabilities_are_not_accepted():
    decision = route(StoredFeatures({}), experts(ecg=float("nan"), alternate=float("nan")), mode="no_abstention")
    assert not decision.accepted and decision.stress_probability is None


def test_confidence_only_and_no_abstention_are_distinct_ablations():
    row = {"ecg_usable": False}
    assert route(StoredFeatures(row), experts(), mode="confidence_only").route == "ecg"
    assert route(StoredFeatures({}), experts(ecg=.55, fusion=.55), mode="no_abstention").accepted
    assert not route(StoredFeatures({}), experts(ecg=.55, fusion=.55)).accepted


def test_labels_and_subjects_cannot_control_routing():
    a = route(StoredFeatures({"label": 0, "subject": "S1"}), experts(ecg=.55))
    b = route(StoredFeatures({"label": 1, "subject": "unseen"}), experts(ecg=.55))
    assert (a.accepted, a.route, a.stress_probability) == (b.accepted, b.route, b.stress_probability)


def test_real_feature_gate_can_fall_back(synthetic_signals):
    synthetic_signals["ecg"][:] = 0
    provider = RawFeatures(synthetic_signals)
    decision = route(provider, experts())
    assert decision.accepted and decision.route == "alternate"
    assert provider.attempted == {"eda", "resp"}


def test_early_exit_is_separate_from_final_acceptance():
    policy = PolicyConfig(.5, .5, early_exit_baseline=.95, early_exit_stress=.95)
    decision = route(StoredFeatures({}), experts(ecg=.8, fusion=.55), policy)
    assert decision.accepted and decision.route == "fusion"
    assert decision.stress_probability == .55
    # Old serialized policy fields retain the previous behavior.
    assert route(StoredFeatures({}), experts(ecg=.8), PolicyConfig(.5, .5)).route == "ecg"


def test_policy_search_matches_runtime_and_protects_fusion_recall():
    import numpy as np
    import pandas as pd
    from pulsepoint.routing import tune_policy, _operating_metrics
    class ColumnExpert:
        def __init__(self, name):
            self.name = name
        def probability(self, row):
            return row[self.name]
    rows = []
    for subject in ("a", "b", "c"):
        for i in range(20):
            label = i % 2
            # Confident ECG misses cannot pass an exit rule merely because
            # the reference cascade also makes those misses.
            rows.append(dict(subject=subject, label=label, ecg=.1 if i < 10 else .9,
                             fusion=.9 if label else .1, alternate=.9 if label else .1,
                             row_id=f"{subject}-{i}", development_scenario="clean"))
    frame = pd.DataFrame(rows)
    models = {name: ColumnExpert(name) for name in ("ecg", "alternate", "fusion")}
    policy, report = tune_policy(frame, models)
    decisions = [route(StoredFeatures(row), models, policy) for row in rows]
    actual = _operating_metrics(frame.label, decisions)
    assert actual["stress_recall"] >= .97
    assert actual["coverage"] >= .8
    chosen = next(r for r in report["candidates"] if
                  r["early_exit_baseline"] == policy.early_exit_baseline and
                  r["early_exit_stress"] == policy.early_exit_stress and
                  r["final_acceptance"] == policy.stress_confidence)
    assert actual["escalation"] == pytest.approx(chosen["escalation"])
    assert actual["stress_recall"] == pytest.approx(chosen["stress_recall"])
    assert np.mean([not d.accepted or d.prediction != row["label"] for d, row in zip(decisions, rows)]) == pytest.approx(chosen["operational_error"])


def test_early_exit_cannot_bypass_final_confidence():
    with pytest.raises(ValueError, match="cannot be lower"):
        PolicyConfig(.8, .8, early_exit_stress=.6)


def test_unavailable_fusion_uses_final_ecg_acceptance_rule():
    row = {"eda_usable": False}
    policy = PolicyConfig(.7, .7, early_exit_baseline=.95, early_exit_stress=.95)
    decision = route(StoredFeatures(row), experts(ecg=.8), policy)
    assert decision.accepted and decision.route == "ecg" and decision.escalated
    assert decision.reason == "accepted_ecg_fusion_unavailable"
    assert not route(StoredFeatures(row), experts(ecg=.6), policy).accepted
    assert not route(StoredFeatures({"eda_usable": False, "ecg_usable": False}), experts(), policy).accepted
