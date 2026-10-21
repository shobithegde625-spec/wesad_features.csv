
import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load the prepared dataset
df = pd.read_csv("wesad_features.csv")

feature_columns = [
    "mean",
    "std",
    "minimum",
    "maximum",
    "median",
    "skewness",
    "kurtosis"
]

X = df[feature_columns]
y = df["label"].astype(int)
groups = df["subject"]

print("Dataset windows:", len(df))
print("Unique subjects:", groups.nunique())
print("Class distribution:")
print(y.value_counts().sort_index())

# Define models
models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "SVM": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVC(
            class_weight="balanced",
            random_state=42
        ))
    ])
}

# Five-fold subject-wise validation
cv = GroupKFold(n_splits=5)
results = []

for model_name, model in models.items():
    print(f"\nEvaluating {model_name}...")

    all_actual = []
    all_predicted = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups), start=1
    ):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        current_model = clone(model)
        current_model.fit(X_train, y_train)
        predictions = current_model.predict(X_test)

        all_actual.extend(y_test.tolist())
        all_predicted.extend(predictions.tolist())

        print(
            f"Fold {fold}: "
            f"Accuracy = {accuracy_score(y_test, predictions):.3f}"
        )

    # Combine held-out predictions from all folds
    accuracy = accuracy_score(all_actual, all_predicted)
    balanced_accuracy = balanced_accuracy_score(
        all_actual, all_predicted
    )
    precision = precision_score(
        all_actual, all_predicted, pos_label=1, zero_division=0
    )
    recall = recall_score(
        all_actual, all_predicted, pos_label=1, zero_division=0
    )
    f1 = f1_score(
        all_actual, all_predicted, pos_label=1, zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "Stress Precision": precision,
        "Stress Recall": recall,
        "Stress F1": f1
    })

# Display and save comparison
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Stress F1", ascending=False
)

print("\n===== MODEL COMPARISON =====")
print(results_df.round(3).to_string(index=False))

results_df.to_csv("model_comparison.csv", index=False)
print("\nSaved results to model_comparison.csv")
