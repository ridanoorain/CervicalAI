import json
import matplotlib.pyplot as plt
from pathlib import Path

HISTORY_PATH = Path(
    "C:/CervicalAI/models/training_history.json"
)

with open(HISTORY_PATH, "r") as f:
    history = json.load(f)

epochs = history["epoch"]
train_loss = history["train_loss"]
val_loss = history["val_loss"]

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    train_loss,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    val_loss,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.xticks(epochs)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "C:/CervicalAI/models/evaluation/training_vs_validation_loss.png",
    dpi=300
)

plt.show()