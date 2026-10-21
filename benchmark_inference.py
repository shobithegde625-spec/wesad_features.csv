
import time
import joblib
import pandas as pd
import numpy as np

saved = joblib.load("stress_detection_model.joblib")
model = saved["model"]
features = saved["features"]

df = pd.read_csv("wesad_features.csv")
sample = df[features].iloc[[0]]

# Warm-up predictions (not included in timing)
for _ in range(5):
    model.predict(sample)

# Measure 100 individual predictions
times_ms = []

for _ in range(100):
    start = time.perf_counter()
    model.predict(sample)
    times_ms.append((time.perf_counter() - start) * 1000)

print("--- Inference Benchmark ---")
print("Number of predictions:", len(times_ms))
print(f"Mean latency: {np.mean(times_ms):.3f} ms")
print(f"Median latency: {np.median(times_ms):.3f} ms")
print(f"95th percentile: {np.percentile(times_ms, 95):.3f} ms")
print(f"Maximum latency: {np.max(times_ms):.3f} ms")
