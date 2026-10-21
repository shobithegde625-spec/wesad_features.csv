
from pathlib import Path
import pickle
import numpy as np

WESAD_FOLDER = Path.home() / "OneDrive" / "Documents" / "WESAD"
data_file = WESAD_FOLDER / "S2" / "S2.pkl"

with open(data_file, "rb") as file:
    data = pickle.load(file, encoding="latin1")

labels = data["label"]

print("Unique label values:", np.unique(labels))
print("Total labels:", len(labels))

for label in np.unique(labels):
    print(f"Label {label}: {(labels == label).sum()} samples")

print("\nSamples selected for baseline and stress:")

baseline_count = (labels == 1).sum()
stress_count = (labels == 2).sum()

print("Baseline samples:", baseline_count)
print("Stress samples:", stress_count)
print("Total selected:", baseline_count + stress_count)

# Inspect the ECG signal and its sampling rate

ecg = data["signal"]["chest"]["ECG"]

print("\nECG information:")
print("ECG shape:", ecg.shape)
print("Number of ECG measurements:", len(ecg))
print("First 10 ECG values:")
print(ecg[:10])

# Show the first 30 labels
print("\nFirst 30 labels:")
print(labels[:30])

# Show how many times the label changes
changes = np.sum(labels[1:] != labels[:-1])
print("\nNumber of label changes:", changes)
print("\nFirst 30 labels:")
print(labels[:30])

changes = np.sum(labels[1:] != labels[:-1])
print("Number of label changes:", changes)

# Select only baseline and stress samples
selected_mask = (labels == 1) | (labels == 2)

# Extract the original ECG measurements
ecg = data["signal"]["chest"]["ECG"].flatten()
selected_ecg = ecg[selected_mask]

# Convert original labels into binary labels
selected_labels = (labels[selected_mask] == 2).astype(int)

print("\n--- Selected Data ---")
print("Selected ECG shape:", selected_ecg.shape)
print("Selected labels shape:", selected_labels.shape)
print("Baseline samples:", np.sum(selected_labels == 0))
print("Stress samples:", np.sum(selected_labels == 1))
print("First 10 model labels:", selected_labels[:10])

# Step 6: Create windows while preserving ECG-label alignment

# Use the ORIGINAL ECG and ORIGINAL labels
ecg = np.asarray(data["signal"]["chest"]["ECG"]).reshape(-1)
original_labels = np.asarray(data["label"]).reshape(-1)

# Check that each ECG sample has a corresponding label
assert len(ecg) == len(original_labels), "ECG and labels do not align!"

# Window settings
sampling_rate = 700
window_seconds = 60
window_size = sampling_rate * window_seconds  # 42,000 samples

windows = []
window_labels = []
rejected_windows = 0

# Create non-overlapping windows
for start in range(0, len(ecg) - window_size + 1, window_size):
    end = start + window_size

    # Keep ECG and labels from exactly the same time interval
    ecg_window = ecg[start:end]
    label_window = original_labels[start:end]

    # Accept only windows containing one target condition
    if np.all(label_window == 1):
        windows.append(ecg_window)
        window_labels.append(0)  # Baseline

    elif np.all(label_window == 2):
        windows.append(ecg_window)
        window_labels.append(1)  # Stress

    else:
        rejected_windows += 1

# Convert lists into NumPy arrays
X = np.asarray(windows)
y = np.asarray(window_labels)

print("\n--- Windowing Results ---")
print("Window size:", window_size, "samples")
print("Number of accepted windows:", len(X))
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Baseline windows:", np.sum(y == 0))
print("Stress windows:", np.sum(y == 1))
print("Rejected windows:", rejected_windows)

# Step 7: Inspect original label transitions

print("\n--- Original Label Transitions ---")

# Find sample positions where the label changes
change_positions = np.where(
    original_labels[1:] != original_labels[:-1]
)[0] + 1

print("Total label transitions:", len(change_positions))

# Show the first 15 transitions
for pos in change_positions[:15]:
    old_label = original_labels[pos - 1]
    new_label = original_labels[pos]

    print(
        f"Sample {pos}: label {old_label} -> label {new_label} "
        f"(time = {pos / sampling_rate:.2f} seconds)"
    )

# Create windows from continuous baseline/stress periods
ecg = np.asarray(data["signal"]["chest"]["ECG"]).reshape(-1)
original_labels = np.asarray(data["label"]).reshape(-1)

assert len(ecg) == len(original_labels)

window_size = 700 * 60  # 42,000 samples = 60 seconds
windows, window_labels = [], []

# Find continuous periods of identical labels
changes = np.where(original_labels[1:] != original_labels[:-1])[0] + 1
boundaries = np.r_[0, changes, len(original_labels)]

for i in range(len(boundaries) - 1):
    start, end = boundaries[i], boundaries[i + 1]
    label = original_labels[start]

    # Keep baseline (1) and stress (2) periods only
    if label not in (1, 2):
        continue

    # Split each continuous period into complete windows
    for pos in range(start, end - window_size + 1, window_size):
        windows.append(ecg[pos:pos + window_size])
        window_labels.append(0 if label == 1 else 1)

X = np.asarray(windows)
y = np.asarray(window_labels)

print("\n--- Corrected Windowing ---")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Baseline windows:", np.sum(y == 0))
print("Stress windows:", np.sum(y == 1))

# Step 8: Extract features from each ECG window

from scipy.stats import skew, kurtosis

features = []

for window in X:
    features.append([
        np.mean(window),          # Average ECG amplitude
        np.std(window),           # Variation in ECG amplitude
        np.min(window),           # Minimum amplitude
        np.max(window),           # Maximum amplitude
        np.median(window),        # Middle amplitude
        skew(window),             # Asymmetry of the signal
        kurtosis(window)          # Shape of the signal distribution
    ])

X_features = np.asarray(features)

print("\n--- Feature Extraction ---")
print("Feature matrix shape:", X_features.shape)
print("Features per window:", X_features.shape[1])
print("First window features:", X_features[0])
