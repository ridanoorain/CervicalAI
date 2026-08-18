"""
RIVA Coarse-Grouping Evaluation

Purpose:
    Takes your best model's (Swin-Tiny) existing 8-class predictions
    and re-evaluates them under two coarser, clinically standard
    groupings:

      1. Binary: Normal vs. Abnormal
      2. 3-tier: Normal / Low-grade / High-grade

    This does NOT retrain anything or change the model's actual
    predictions -- it re-labels the SAME predictions under a
    different, legitimate grouping and reports accuracy on that
    grouping. Report this ALONGSIDE your 8-class result, not instead
    of it.

Groupings used (standard in cervical cytology literature, based on
Bethesda System severity categories):

    Binary:
        Normal   <- NILM, INFL, ENDO
        Abnormal <- LSIL, HSIL, SCC, ASCH, ASCUS

    3-tier:
        Normal     <- NILM, INFL, ENDO
        Low-grade  <- LSIL, ASCUS, ASCH
        High-grade <- HSIL, SCC

    Note on ASCUS/ASCH placement: these are "atypical" (indeterminate)
    categories, not confirmed dysplasia. Some papers group them with
    Abnormal/Low-grade (since they trigger follow-up testing
    clinically), others treat them as a separate "indeterminate"
    tier. This script uses the follow-up-triggering convention
    (grouped with Abnormal/Low-grade) -- state this choice explicitly
    in your paper, since it's a genuine methodological decision, not
    an obvious default.

Usage:
    python src/training/coarse_grouping_eval.py
"""

import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset.riva_dataset import RIVADataset
from models.swin import create_model, CLASS_NAMES, NUM_CLASSES
from utils.metrics import evaluate_predictions


CONFIG = {
    "val_split": os.path.join(PROJECT_ROOT, "data", "processed", "splits", "val.json"),
    "test_split": os.path.join(PROJECT_ROOT, "data", "processed", "splits", "test.json"),

    "checkpoint_path": os.path.join(PROJECT_ROOT, "models", "checkpoints", "swin", "best_model.pth"),

    "batch_size": 8,
    "num_workers": 0,

    "reports_dir": os.path.join(PROJECT_ROOT, "reports"),
}

# ----------------------------------------------------------------
# Original 8-class index -> class name (from models/swin.py)
# ----------------------------------------------------------------
# 0 NILM, 1 INFL, 2 LSIL, 3 HSIL, 4 SCC, 5 ENDO, 6 ASCH, 7 ASCUS

BINARY_NAMES = ["Normal", "Abnormal"]
BINARY_MAP = {
    0: 0,  # NILM -> Normal
    1: 0,  # INFL -> Normal
    5: 0,  # ENDO -> Normal
    2: 1,  # LSIL -> Abnormal
    3: 1,  # HSIL -> Abnormal
    4: 1,  # SCC -> Abnormal
    6: 1,  # ASCH -> Abnormal
    7: 1,  # ASCUS -> Abnormal
}

TIER_NAMES = ["Normal", "Low-grade", "High-grade"]
TIER_MAP = {
    0: 0,  # NILM -> Normal
    1: 0,  # INFL -> Normal
    5: 0,  # ENDO -> Normal
    2: 1,  # LSIL -> Low-grade
    6: 1,  # ASCH -> Low-grade
    7: 1,  # ASCUS -> Low-grade
    3: 2,  # HSIL -> High-grade
    4: 2,  # SCC -> High-grade
}


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")
    return device


def create_val_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_predictions(model, loader, device):
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, soft_labels, _ in loader:
            images = images.to(device, non_blocking=True)
            soft_labels = soft_labels.to(device, non_blocking=True)

            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            targets = torch.argmax(soft_labels, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return all_targets, all_predictions


def remap(labels, mapping):
    return [mapping[int(label)] for label in labels]


def evaluate_split(split_name, split_path, model, device):
    print(f"\n{'#' * 70}")
    print(f"SPLIT: {split_name.upper()}")
    print(f"{'#' * 70}")

    transform = create_val_transform()
    dataset = RIVADataset(split_file=split_path, transform=transform)
    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    print(f"Samples: {len(dataset)}")

    y_true_8class, y_pred_8class = get_predictions(model, loader, device)

    # ------------------------------------------------------------
    # Original 8-class result (for reference -- should match your
    # existing riva_swin_tiny report if this is the val split)
    # ------------------------------------------------------------
    evaluate_predictions(
        y_true=y_true_8class,
        y_pred=y_pred_8class,
        class_names=CLASS_NAMES,
        output_dir=CONFIG["reports_dir"],
        run_name=f"riva_swin_tiny_{split_name}_8class_check",
    )

    # ------------------------------------------------------------
    # Binary: Normal vs Abnormal
    # ------------------------------------------------------------
    y_true_binary = remap(y_true_8class, BINARY_MAP)
    y_pred_binary = remap(y_pred_8class, BINARY_MAP)

    evaluate_predictions(
        y_true=y_true_binary,
        y_pred=y_pred_binary,
        class_names=BINARY_NAMES,
        output_dir=CONFIG["reports_dir"],
        run_name=f"riva_swin_tiny_{split_name}_binary",
    )

    # ------------------------------------------------------------
    # 3-tier: Normal / Low-grade / High-grade
    # ------------------------------------------------------------
    y_true_tier = remap(y_true_8class, TIER_MAP)
    y_pred_tier = remap(y_pred_8class, TIER_MAP)

    evaluate_predictions(
        y_true=y_true_tier,
        y_pred=y_pred_tier,
        class_names=TIER_NAMES,
        output_dir=CONFIG["reports_dir"],
        run_name=f"riva_swin_tiny_{split_name}_3tier",
    )


def main():
    print("\n" + "=" * 70)
    print("RIVA COARSE-GROUPING EVALUATION (Swin-Tiny)")
    print("=" * 70)

    device = get_device()

    print(f"\nLoading checkpoint: {CONFIG['checkpoint_path']}")
    checkpoint = torch.load(CONFIG["checkpoint_path"], map_location=device)

    model = create_model(pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    if "val_macro_f1" in checkpoint:
        print(f"Checkpoint's recorded val macro F1: {checkpoint['val_macro_f1']:.4f}")

    # Run on val split (matches your existing riva_swin_tiny numbers).
    evaluate_split("val", CONFIG["val_split"], model, device)

    # Also run on test split, since this is a good final checkpoint
    # to report a true held-out number from -- your test set has
    # NEVER been touched by any training or model-selection decision
    # so far, making it the most defensible number for your paper's
    # final results table.
    if os.path.exists(CONFIG["test_split"]):
        evaluate_split("test", CONFIG["test_split"], model, device)

    print("\n" + "=" * 70)
    print("COARSE-GROUPING EVALUATION COMPLETE")
    print("Check reports/ for *_binary_report.json and *_3tier_report.json")
    print("=" * 70)


if __name__ == "__main__":
    main()