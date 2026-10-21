
import joblib
import pandas as pd

# Load the saved model
saved = joblib.load("stress_detection_model.joblib")
model = saved["model"]
features = saved["features"]

# Load one example from our dataset
df = pd.read_csv("wesad_features.csv")
sample = df[features].iloc[[0]]

# Predict the condition
prediction = model.predict(sample)[0]
probabilities = model.predict_proba(sample)[0]

print("Prediction:", "Stress" if prediction == 1 else "Baseline")
print("Baseline probability:", round(probabilities[0], 3))
print("Stress probability:", round(probabilities[1], 3))
