
import joblib
import pandas as pd
from time import perf_counter

# 1. Load the trained model once
saved = joblib.load("stress_detection_model.joblib")
model = saved["model"]
features = saved["features"]

# 2. Load an example window from the dataset
df = pd.read_csv("wesad_features.csv")

# Select a stress example (label 1)
stress_samples = df[df["label"] == 1]

if stress_samples.empty:
    raise ValueError("No stress samples found in wesad_features.csv")

sample = stress_samples[features].iloc[[0]]

# 3. Run prediction locally and measure inference time
start = perf_counter()

prediction = model.predict(sample)[0]
probabilities = model.predict_proba(sample)[0]

elapsed_ms = (perf_counter() - start) * 1000

# 4. Display results
print("--- Local Edge Inference ---")
print("Actual label: Stress")
print("Predicted condition:", "Stress" if prediction == 1 else "Baseline")
print(f"Baseline score: {probabilities[0]:.3f}")
print(f"Stress score: {probabilities[1]:.3f}")
print(f"Inference time: {elapsed_ms:.3f} ms")
print("Processing location: Local computer")

