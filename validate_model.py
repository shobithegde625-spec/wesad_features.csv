
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "wesad_features.csv"
MODEL_PATH = BASE_DIR / "stress_detection_model.joblib"

# --------------------------------------------------
# 2. Load dataset and saved model
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)
saved_object = joblib.load(MODEL_PATH)

print("Saved object type:", type(saved_object).__name__)

# Support either a direct model or a dictionary wrapper
if isinstance(saved_object, dict):
    print("Saved object keys:", list(saved_object.keys()))

    # Find the actual estimator inside the dictionary
    model = next(
        (
            value for value in saved_object.values()
            if hasattr(value, "fit")
            and hasattr(value, "predict")
            and hasattr(value, "get_params")
        ),
        None
    )

    if model is None:
        raise ValueError(
            "No scikit-learn estimator found in the saved dictionary. "
            "Check the saved object's keys and structure."
        )
else:
    model = saved_object

# Confirm this is a scikit-learn-compatible estimator
if not (
    hasattr(model, "fit")
    and hasattr(model, "predict")
    and hasattr(model, "get_params")
):
    raise TypeError(
        "The loaded model is not a cloneable scikit-learn estimator."
    )

# --------------------------------------------------
# 3. Prepare features, labels, and subject groups
# --------------------------------------------------
feature_names = [
    "mean",
    "std",
    "minimum",
    "maximum",
    "median",
    "skewness",
    "kurtosis"
]

required_columns = feature_names + ["label", "subject"]
missing = [column for column in required_columns if column not in df.columns]

if missing:
    raise ValueError(f"Missing CSV columns: {missing}")

X = df[feature_names]
y = df["label"]
groups = df["subject"]

print("\nDataset windows:", len(df))
print("Unique subjects:", groups.nunique())
print("\nClass distribution:")
print(y.value_counts().sort_index())

# --------------------------------------------------
# 4. Subject-wise cross-validation
# --------------------------------------------------
number_of_folds = min(5, groups.nunique())

if number_of_folds < 2:
    raise ValueError("At least two subjects are required.")

cv = GroupKFold(n_splits=number_of_folds)

all_actual = []
all_predicted = []
fold_accuracies = []
fold_balanced_accuracies = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups=groups), start=1
):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # Train a fresh model for this fold
    fold_model = clone(model)
    fold_model.fit(X_train, y_train)

    predictions = fold_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)

    fold_accuracies.append(accuracy)
    fold_balanced_accuracies.append(balanced_accuracy)

    all_actual.extend(y_test.tolist())
    all_predicted.extend(predictions.tolist())

    held_out_subjects = sorted(groups.iloc[test_idx].unique())

    print(f"\n--- Fold {fold} ---")
    print("Held-out subjects:", held_out_subjects)
    print("Training windows:", len(train_idx))
    print("Testing windows:", len(test_idx))
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Balanced accuracy: {balanced_accuracy:.3f}")

# --------------------------------------------------
# 5. Overall evaluation
# --------------------------------------------------
print("\n========== SUBJECT-WISE VALIDATION ==========")

print(f"Mean fold accuracy: {np.mean(fold_accuracies):.3f}")
print(f"Accuracy standard deviation: {np.std(fold_accuracies):.3f}")
print(
    "Mean fold balanced accuracy:",
    f"{np.mean(fold_balanced_accuracies):.3f}"
)
print(
    "Overall accuracy:",
    f"{accuracy_score(all_actual, all_predicted):.3f}"
)
print(
    "Overall balanced accuracy:",
    f"{balanced_accuracy_score(all_actual, all_predicted):.3f}"
)

print("\nClassification report:")
print(
    classification_report(
        all_actual,
        all_predicted,
        zero_division=0
    )
)

print("Confusion matrix:")
print(confusion_matrix(all_actual, all_predicted))

# If stress is encoded as 1, report its F1 score explicitly
if 1 in set(all_actual):
    stress_f1 = f1_score(
        all_actual,
        all_predicted,
        pos_label=1,
        zero_division=0
    )
    print(f"Stress-class F1 score: {stress_f1:.3f}")
