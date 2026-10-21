
import pickle
from pathlib import Path

# Change this if your WESAD folder is in a different location
WESAD_FOLDER = Path.home() / "OneDrive" / "Documents" / "WESAD"

# Find the data file for participant S2
data_file = WESAD_FOLDER / "S2" / "S2.pkl"

print("Looking for:", data_file)

if not data_file.exists():
    print("File not found! Check the WESAD folder location.")
else:
    with open(data_file, "rb") as file:
        data = pickle.load(file, encoding="latin1")

    print("\nDataset loaded successfully!")
    print("Top-level keys:", list(data.keys()))

    print("\nAvailable sensor groups:")
    for group, signals in data["signal"].items():
        print(f"{group}: {list(signals.keys())}")

    print("\nLabel array shape:", data["label"].shape)
    print("Number of labels:", len(data["label"]))
    
from collections import Counter

print("\nLabel counts:")
for label, count in sorted(Counter(data["label"]).items()):
    print(f"Label {label}: {count:,} samples")

print("\nChest signal shapes:")
chest = data["signal"]["chest"]

for name, values in chest.items():
    print(name, values.shape)

print("\nWrist signal shapes:")
wrist = data["signal"]["wrist"]

for name, values in wrist.items():
    print(name, values.shape)

