import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_PATH = Path(
    "models/evaluation/test_results.json"
)

OUTPUT_DIR = Path(
    "models/evaluation/plots"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD TEST RESULTS
# ============================================================

print("=" * 70)
print("RIVA TEST EVALUATION PLOTS")
print("=" * 70)

print("\nLoading test results from:")
print(RESULTS_PATH)

with open(RESULTS_PATH, "r") as f:
    results = json.load(f)

print("Successfully loaded test results.")


# ============================================================
# CLASS INFORMATION
# ============================================================

CLASSES = [
    "NILM",
    "INFL",
    "LSIL",
    "HSIL",
    "SCC",
    "ENDO",
    "ASCH",
    "ASCUS",
]


# ============================================================
# 1. CONFUSION MATRIX
# ============================================================

print("\nGenerating confusion matrix...")

# The evaluation script should save this as:
# results["confusion_matrix"]

confusion_matrix = np.array(
    results["confusion_matrix"]
)

fig, ax = plt.subplots(
    figsize=(10, 8)
)

image = ax.imshow(
    confusion_matrix
)

# Color bar
fig.colorbar(
    image,
    ax=ax
)

# Axis labels
ax.set_xticks(
    np.arange(len(CLASSES))
)

ax.set_yticks(
    np.arange(len(CLASSES))
)

ax.set_xticklabels(
    CLASSES
)

ax.set_yticklabels(
    CLASSES
)

ax.set_xlabel(
    "Predicted Label"
)

ax.set_ylabel(
    "Actual Label"
)

ax.set_title(
    "RIVA Test Set Confusion Matrix"
)

# Write values inside cells
for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[1]):

        ax.text(
            j,
            i,
            str(confusion_matrix[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

confusion_path = (
    OUTPUT_DIR /
    "confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Confusion matrix saved to:\n"
    f"{confusion_path}"
)


# ============================================================
# 2. CLASS-WISE PERFORMANCE
# ============================================================

print("\nGenerating class-wise performance plot...")

# Expected structure:
#
# results["classification_report"] = {
#     "NILM": {
#         "precision": ...,
#         "recall": ...,
#         "f1-score": ...
#     },
#     ...
# }

report = results["classification_report"]


precision = []
recall = []
f1_score = []

valid_classes = []

for class_name in CLASSES:

    # Skip classes that don't exist
    if class_name not in report:
        continue

    metrics = report[class_name]

    valid_classes.append(
        class_name
    )

    precision.append(
        metrics["precision"]
    )

    recall.append(
        metrics["recall"]
    )

    f1_score.append(
        metrics["f1-score"]
    )


# ============================================================
# BAR CHART
# ============================================================

x = np.arange(
    len(valid_classes)
)

width = 0.25

fig, ax = plt.subplots(
    figsize=(12, 7)
)

ax.bar(
    x - width,
    precision,
    width,
    label="Precision"
)

ax.bar(
    x,
    recall,
    width,
    label="Recall"
)

ax.bar(
    x + width,
    f1_score,
    width,
    label="F1 Score"
)

ax.set_xlabel(
    "Cervical Cell Class"
)

ax.set_ylabel(
    "Score"
)

ax.set_title(
    "Class-wise Performance on RIVA Test Set"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    valid_classes
)

ax.set_ylim(
    0,
    1
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

performance_path = (
    OUTPUT_DIR /
    "classwise_performance.png"
)

plt.savefig(
    performance_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Class-wise performance saved to:\n"
    f"{performance_path}"
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PLOT GENERATION COMPLETE")
print("=" * 70)

print(
    f"\nConfusion Matrix:\n"
    f"{confusion_path}"
)

print(
    f"\nClass-wise Performance:\n"
    f"{performance_path}"
)