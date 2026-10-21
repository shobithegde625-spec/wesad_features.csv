
import pandas as pd
import matplotlib.pyplot as plt

# Load the model comparison results
df = pd.read_csv("model_comparison.csv")

# Show the results in the terminal
print("Model comparison results:")
print(df.round(3).to_string(index=False))

# Create a graph comparing model performance
metrics = [
    "Accuracy",
    "Balanced Accuracy",
    "Stress F1"
]

ax = df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(10, 6),
    ylim=(0, 1),
    width=0.75
)

ax.set_title("Stress Detection: Model Performance Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("Machine Learning Model")
ax.legend(title="Evaluation Metric")
ax.tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300)
plt.show()

print("\nSaved graph as model_comparison.png")
