import json
from pathlib import Path

import matplotlib.pyplot as plt


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

HISTORY_PATH = Path("models/training_history.json")
OUTPUT_DIR = Path("models/evaluation")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# LOAD TRAINING HISTORY
# ---------------------------------------------------------

print("=" * 70)
print("RIVA TRAINING METRICS PLOT")
print("=" * 70)

print(f"\nLoading history from:\n{HISTORY_PATH}")

with open(HISTORY_PATH, "r") as f:
    history = json.load(f)

print("Successfully loaded training history.")

train_accuracy = history["train_accuracy"]
val_accuracy = history["val_accuracy"]

train_macro_f1 = history["train_macro_f1"]
val_macro_f1 = history["val_macro_f1"]

epochs = range(1, len(train_accuracy) + 1)

best_epoch = history.get("best_epoch")
best_val_f1 = history.get("best_val_macro_f1")


# ---------------------------------------------------------
# 1. ACCURACY PLOT
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    train_accuracy,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    val_accuracy,
    marker="o",
    label="Validation Accuracy"
)

if best_epoch is not None:
    plt.axvline(
        best_epoch,
        linestyle="--",
        label=f"Best Epoch ({best_epoch})"
    )

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xticks(list(epochs))
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

accuracy_path = OUTPUT_DIR / "training_vs_validation_accuracy.png"

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# 2. MACRO F1 PLOT
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    train_macro_f1,
    marker="o",
    label="Training Macro F1"
)

plt.plot(
    epochs,
    val_macro_f1,
    marker="o",
    label="Validation Macro F1"
)

if best_epoch is not None:
    plt.axvline(
        best_epoch,
        linestyle="--",
        label=f"Best Epoch ({best_epoch})"
    )

if best_epoch is not None and best_val_f1 is not None:
    plt.scatter(
        [best_epoch],
        [best_val_f1],
        s=80,
        zorder=5,
        label=f"Best Val F1 = {best_val_f1:.4f}"
    )

plt.xlabel("Epoch")
plt.ylabel("Macro F1")
plt.title("Training vs Validation Macro F1")
plt.xticks(list(epochs))
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

f1_path = OUTPUT_DIR / "training_vs_validation_macro_f1.png"

plt.savefig(
    f1_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("PLOTS GENERATED")
print("=" * 70)

print(f"\nAccuracy plot:")
print(accuracy_path)

print(f"\nMacro F1 plot:")
print(f1_path)

if best_epoch is not None:
    print(f"\nBest epoch      : {best_epoch}")

if best_val_f1 is not None:
    print(f"Best Val Macro F1: {best_val_f1:.4f}")

print("\nPlot generation complete.")