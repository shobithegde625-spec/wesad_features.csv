"""Small estimator factories, explicit participant-disjoint calibration, selection."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from .config import EXPERT_FEATURES, LEGACY_FEATURES, EXPERT_MODALITIES


def model_factory(name, seed=42):
    if name == "logistic":
        estimator = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
    elif name in ("svm", "svm_c10", "svm_smooth"):
        estimator = SVC(C=1.0 if name == "svm" else 10.0,
                        gamma=.01 if name == "svm_smooth" else "scale",
                        class_weight="balanced", random_state=seed)
    elif name in ("forest", "historical_forest", "forest_compact"):
        # Historical reproduction is explicit; all current workflows use the same factory.
        estimator = RandomForestClassifier(n_estimators=200 if name == "historical_forest" else 64 if name == "forest_compact" else 100,
                                            max_depth=6 if name == "forest_compact" else None,
                                            min_samples_leaf=3 if name == "forest_compact" else 1,
                                            class_weight="balanced", n_jobs=1, random_state=seed)
    else:
        raise ValueError(f"Unknown model: {name}")
    steps = [SimpleImputer(strategy="median")]
    if name not in ("forest", "historical_forest", "forest_compact"):
        steps.append(StandardScaler())
    return make_pipeline(*steps, estimator)


def raw_score(estimator, X):
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X)).reshape(-1)
    p = estimator.predict_proba(X)[:, list(estimator.classes_).index(1)]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dataclass
class Expert:
    name: str
    features: list
    estimator: object
    calibrator: object
    family: str

    def probabilities(self, X):
        scores = raw_score(self.estimator, X[self.features])
        return self.calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]

    def probability(self, values):
        X = pd.DataFrame([{f: values.get(f, np.nan) for f in self.features}])
        if not np.isfinite(X.to_numpy(dtype=float)).all():
            return None
        return float(self.probabilities(X)[0])


def expert_specs(frame):
    if all(c in frame for c in EXPERT_FEATURES["fusion"]):
        return EXPERT_FEATURES.copy()
    return {"ecg": list(LEGACY_FEATURES)}


def valid_for(frame, name, features):
    mask = np.isfinite(frame[features].to_numpy(dtype=float)).all(axis=1)
    for modality in EXPERT_MODALITIES[name]:
        key = f"{modality}_usable"
        if key in frame:
            mask &= frame[key].astype(str).str.lower().isin(["true", "1", "1.0"]).to_numpy()
    return frame.loc[mask].copy()


def ensure_trainable(frame, role, min_subjects=1):
    if len(frame) < 4 or frame.subject.nunique() < min_subjects or set(frame.label.unique()) != {0, 1}:
        raise ValueError(f"{role}: need both classes and at least {min_subjects} usable subjects")


def development_split(frame, seed):
    subjects = np.array(sorted(frame.subject.unique()))
    if len(subjects) < 6:
        raise ValueError("Nested training requires at least six development subjects")
    subjects = np.random.default_rng(seed).permutation(subjects)
    n = max(1, len(subjects) // 5)
    groups = {"fit": subjects[2*n:].tolist(), "calibration": subjects[n:2*n].tolist(), "policy": subjects[:n].tolist()}
    assert not (set(groups["fit"]) & set(groups["calibration"]))
    assert not (set(groups["policy"]) & (set(groups["fit"]) | set(groups["calibration"])))
    return {name: frame[frame.subject.isin(ids)].copy() for name, ids in groups.items()}, groups


def fit_experts(frame, seed=42, families=("logistic", "svm", "forest")):
    partitions, manifest = development_split(frame, seed)
    experts, selections = {}, {}
    for name, features in expert_specs(frame).items():
        fit = valid_for(partitions["fit"], name, features)
        calibration = valid_for(partitions["calibration"], name, features)
        ensure_trainable(fit, f"{name} fit", 2)
        ensure_trainable(calibration, f"{name} calibration")
        cv = GroupKFold(n_splits=min(3, fit.subject.nunique()))
        candidates = {}
        search_families = list(families)
        if "svm" in families:
            search_families.extend(["svm_c10", "svm_smooth"])
        if "forest" in families:
            search_families.append("forest_compact")
        for family in dict.fromkeys(search_families):
            scores = []
            for train, test in cv.split(fit, fit.label, fit.subject):
                train_frame = fit.iloc[train]
                ensure_trainable(train_frame, "inner fit")
                estimator = model_factory(family, seed)
                estimator.fit(train_frame[features], train_frame.label.astype(int))
                predictions = estimator.predict(fit.iloc[test][features])
                held_out = fit.iloc[test]
                # Each participant gets equal weight inside model selection.
                scores.extend(balanced_accuracy_score(held_out.label.to_numpy()[mask], predictions[mask])
                              for subject in held_out.subject.unique()
                              for mask in [held_out.subject.to_numpy() == subject])
            candidates[family] = float(np.mean(scores))
        winner = max(candidates, key=candidates.get)
        estimator = model_factory(winner, seed).fit(fit[features], fit.label.astype(int))
        scores = raw_score(estimator, calibration[features])
        calibrator = LogisticRegression(C=1.0, random_state=seed).fit(scores.reshape(-1, 1), calibration.label.astype(int))
        experts[name] = Expert(name, features, estimator, calibrator, winner)
        selections[name] = {"selected_family": winner, "inner_balanced_accuracy": candidates,
                            "fit_rows": len(fit), "calibration_rows": len(calibration)}
    return experts, partitions["policy"], manifest, selections
