import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from torchvision import transforms

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from dataset.riva_dataset import RIVADataset
from models.efficientnet import create_model
from training.loss import SoftLabelCrossEntropy


# ============================================================
# CONFIGURATION
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

TEST_SPLIT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "test.json"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "best_model.pth"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "models"
    / "evaluation"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BATCH_SIZE = 16

NUM_CLASSES = 8

LABELS = [
    "NILM",
    "INFL",
    "LSIL",
    "HSIL",
    "SCC",
    "ENDO",
    "ASCH",
    "ASCUS",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# TEST DATASET
# ============================================================

def create_test_dataset():

    print("\n" + "=" * 70)
    print("LOADING TEST DATASET")
    print("=" * 70)

    print(f"\nTest split:")
    print(TEST_SPLIT)

    dataset = RIVADataset(
    split_file=TEST_SPLIT,
    transform=test_transform
)

    print(
        f"\nTest samples: "
        f"{len(dataset)}"
    )

    return dataset


# ============================================================
# MODEL
# ============================================================

def load_model():

    print("\n" + "=" * 70)
    print("LOADING BEST MODEL")
    print("=" * 70)

    print(f"\nCheckpoint:")
    print(CHECKPOINT_PATH)

    model = create_model(
        num_classes=NUM_CLASSES
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    # Handle both checkpoint formats:
    # 1. Direct state_dict
    # 2. Dictionary containing model_state_dict

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if "epoch" in checkpoint:
            print(
                f"Checkpoint epoch: "
                f"{checkpoint['epoch']}"
            )

        if "val_macro_f1" in checkpoint:
            print(
                f"Checkpoint Val Macro F1: "
                f"{checkpoint['val_macro_f1']:.4f}"
            )

    else:

        model.load_state_dict(
            checkpoint
        )

    model = model.to(DEVICE)

    model.eval()

    print("\nModel loaded successfully.")

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader):
    device = next(model.parameters()).device
    model.eval()

    print("\n" + "=" * 70)
    print("TEST EVALUATION")
    print("=" * 70)

    criterion = SoftLabelCrossEntropy()

    total_loss = 0.0
    total_samples = 0

    all_predictions = []
    all_majority_labels = []
    all_soft_labels = []

    with torch.no_grad():
        for images, soft_labels, majority_labels in loader:
            images = images.to(device)
            soft_labels = soft_labels.to(device)
            majority_labels = majority_labels.to(device)

            logits = model(images)
            loss = criterion(logits, soft_labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_majority_labels.extend(majority_labels.cpu().numpy())
            all_soft_labels.extend(soft_labels.cpu().numpy())

    test_loss = total_loss / total_samples

    return test_loss, all_predictions, all_majority_labels, all_soft_labels

# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("RIVA TEST / EVALUATION")
    print("=" * 70)

    print(f"\nDevice: {DEVICE}")

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    test_dataset = create_test_dataset()

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------
    test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)
    (
        test_loss,
        targets,
        predictions,
        soft_labels
    ) = evaluate(
        model,
        test_loader
    )

    test_accuracy = accuracy_score(targets, predictions)
    test_macro_f1 = f1_score(targets, predictions, average="macro", zero_division=0)
    test_weighted_f1 = f1_score(targets, predictions, average="weighted", zero_division=0)


    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(
        f"\nTest Loss       : "
        f"{test_loss:.4f}"
    )

    print(
        f"Test Accuracy   : "
        f"{test_accuracy:.4f}"
        f" ({test_accuracy * 100:.2f}%)"
    )

    print(
        f"Test Macro F1   : "
        f"{test_macro_f1:.4f}"
    )

    print(
        f"Test Weighted F1: "
        f"{test_weighted_f1:.4f}"
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    report = classification_report(
        targets,
        predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=LABELS,
        digits=4,
        zero_division=0
    )

    print("\n")
    print(report)

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    cm = confusion_matrix(
        targets,
        predictions,
        labels=list(range(NUM_CLASSES))
    )

    print("\nRows = Actual")
    print("Columns = Predicted\n")

    print(
        f"{'':>10}",
        end=""
    )

    for label in LABELS:
        print(
            f"{label:>8}",
            end=""
        )

    print()

    for i, label in enumerate(LABELS):

        print(
            f"{label:>10}",
            end=""
        )

        for j in range(NUM_CLASSES):

            print(
                f"{cm[i][j]:>8}",
                end=""
            )

        print()

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results = {

        "test_loss": float(test_loss),

        "test_accuracy": float(
            test_accuracy
        ),

        "test_macro_f1": float(
            test_macro_f1
        ),

        "test_weighted_f1": float(
            test_weighted_f1
        ),

        "classification_report":
            classification_report(
                targets,
                predictions,
                labels=list(range(NUM_CLASSES)),
                target_names=LABELS,
                output_dict=True,
                zero_division=0
            ),

        "confusion_matrix":
            cm.tolist(),

        "labels": LABELS,

        "num_test_samples":
            len(test_dataset),

        "checkpoint":
            str(CHECKPOINT_PATH),
    }

    results_path = (
        RESULTS_DIR
        / "test_results.json"
    )

    with open(
        results_path,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print("\n" + "=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(
        f"\nTest results saved to:"
        f"\n{results_path}"
    )

    print("\n" + "=" * 70)
    print("TEST EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()