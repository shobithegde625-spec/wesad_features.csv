
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)

# 1. Load the prepared dataset
df = pd.read_csv("wesad_features.csv")

# 2. Select the features used by the model
features = [
    "mean",
    "std",
    "minimum",
    "maximum",
    "median",
    "skewness",
    "kurtosis"
]

# Check that the required columns exist
required_columns = features + ["label", "subject"]
missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns in wesad_features.csv: {missing}")

# Keep only valid baseline and stress labels
df = df[df["label"].isin([0, 1])].copy()

# Remove rows with missing or infinite feature values
df = df.replace([float("inf"), float("-inf")], float("nan"))
df = df.dropna(subset=features + ["label", "subject"])

X = df[features]
y = df["label"].astype(int)
groups = df["subject"]

# 3. Split by subject, not by individual window
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)

train_idx, test_idx = next(splitter.split(X, y, groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("Training subjects:", sorted(groups.iloc[train_idx].unique()))
print("Testing subjects:", sorted(groups.iloc[test_idx].unique()))
print("Training windows:", len(X_train))
print("Testing windows:", len(X_test))

# 4. Create and train the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

# 5. Evaluate the model on held-out subjects
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
balanced_accuracy = balanced_accuracy_score(y_test, predictions)
stress_f1 = f1_score(
    y_test,
    predictions,
    pos_label=1,
    zero_division=0
)

print("\n--- Random Forest Evaluation ---")
print(f"Test accuracy: {accuracy:.3f}")
print(f"Balanced accuracy: {balanced_accuracy:.3f}")
print(f"Stress-class F1 score: {stress_f1:.3f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Baseline", "Stress"],
        zero_division=0
    )
)

cm = confusion_matrix(y_test, predictions, labels=[0, 1])
print("Confusion matrix:")
print(cm)

# 6. Save the trained model and feature order
joblib.dump(
    {
        "model": model,
        "features": features
    },
    "stress_detection_model.joblib"
)

# 7. Save the confusion matrix image
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Baseline", "Stress"]
)

display.plot()
plt.title(f"Random Forest (Accuracy: {accuracy:.1%})")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
plt.close()

print("\nSaved stress_detection_model.joblib")
print("Saved confusion_matrix.png")

