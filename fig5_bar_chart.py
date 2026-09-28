import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---- Load the Table 3 results we already saved ----
df = pd.read_csv("results/table3_accuracy_precision_recall.csv")
print(df)

# Convert percentage strings like "91.24%" back into plain numbers
for col in ["Accuracy", "Precision", "Recall"]:
    df[col] = df[col].str.replace("%", "").astype(float)

classifiers = df["Classifier"].tolist()
metrics = ["Accuracy", "Precision", "Recall"]

x = np.arange(len(metrics))       # positions for each metric group
width = 0.25                       # width of each bar

fig, ax = plt.subplots(figsize=(9, 6))

for i, clf in enumerate(classifiers):
    values = df[df["Classifier"] == clf][metrics].values.flatten()
    ax.bar(x + i*width, values, width, label=clf)

ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.set_ylabel("Percentage (%)")
ax.set_ylim(0, 100)
ax.set_title("Fig 5 (Abstracts Dataset): Accuracy, Precision, Recall by Classifier")
ax.legend()
ax.grid(axis="y", alpha=0.3)

# Add value labels on top of each bar for readability
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f", padding=2)

plt.tight_layout()
plt.savefig("results/fig5_bar_chart.png", dpi=150, bbox_inches="tight")
print("\nSaved bar chart to results/fig5_bar_chart.png")
plt.show()