
from pathlib import Path
import joblib
import time

model_path = Path("stress_detection_model.joblib")

# Measure saved model size
size_bytes = model_path.stat().st_size
size_kb = size_bytes / 1024

# Load the model and measure loading time
start = time.perf_counter()
saved = joblib.load(model_path)
loading_ms = (time.perf_counter() - start) * 1000

print("--- Edge Model Measurements ---")
print(f"Saved model size: {size_kb:.2f} KB")
print(f"Model loading time: {loading_ms:.3f} ms")
print("Inference runs locally: Yes")
