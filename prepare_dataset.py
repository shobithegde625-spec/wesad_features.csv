
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew, kurtosis

WESAD_FOLDER = Path.home() / "OneDrive" / "Documents" / "WESAD"

SAMPLING_RATE = 700
WINDOW_SIZE = SAMPLING_RATE * 60  # 60 seconds

all_features = []
all_labels = []
all_subjects = []

for subject_folder in sorted(WESAD_FOLDER.glob("S*")):
    subject_file = subject_folder / f"{subject_folder.name}.pkl"

    if not subject_file.exists():
        continue

    print("Processing", subject_folder.name)

    with open(subject_file, "rb") as file:
        data = pickle.load(file, encoding="latin1")

    ecg = np.asarray(data["signal"]["chest"]["ECG"]).reshape(-1)
    labels = np.asarray(data["label"]).reshape(-1)

    if len(ecg) != len(labels):
        print("Skipping: ECG and labels do not align")
        continue

    # Identify continuous periods with identical labels
    changes = np.where(labels[1:] != labels[:-1])[0] + 1
    boundaries = np.r_[0, changes, len(labels)]

    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        original_label = labels[start]

        if original_label not in (1, 2):
            continue

        model_label = 0 if original_label == 1 else 1

        for pos in range(start, end - WINDOW_SIZE + 1, WINDOW_SIZE):
            window = ecg[pos:pos + WINDOW_SIZE]

            features = [
                np.mean(window),
                np.std(window),
                np.min(window),
                np.max(window),
                np.median(window),
                skew(window),
                kurtosis(window)
            ]

            all_features.append(features)
            all_labels.append(model_label)
            all_subjects.append(subject_folder.name)

# Save the combined dataset
feature_names = [
    "mean", "std", "minimum", "maximum",
    "median", "skewness", "kurtosis"
]

df = pd.DataFrame(all_features, columns=feature_names)
df["label"] = all_labels
df["subject"] = all_subjects

output_file = Path("wesad_features.csv")
df.to_csv(output_file, index=False)

print("\n--- Dataset Summary ---")
print("Total windows:", len(df))
print("Baseline windows:", (df["label"] == 0).sum())
print("Stress windows:", (df["label"] == 1).sum())
print("Subjects processed:", df["subject"].nunique())
print("Saved to:", output_file.resolve())


