"""
RIVA Swin-Tiny Training

Purpose:
    Trains Swin-Tiny on RIVA as a second architecture, using the
    same class-weighted loss approach that worked best so far
    (riva_sipakmed_transfer style: weighting on, no freezing).

    This is a standalone script (not requiring edits to train.py)
    so it can run independently and be compared directly against
    your EfficientNet results.

Usage:
    python src/training/train_swin.py

Memory:
    batch_size=8 is deliberately conservative for a 6GB laptop GPU.
    If you see a CUDA out-of-memory error, drop to batch_size=4.
"""

import os
import sys
import json

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import f1_score, accuracy_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset.riva_dataset import RIVADataset
from models.swin import create_model, CLASS_NAMES, NUM_CLASSES
from training.loss import SoftLabelKLDivergence
from training.loss_weighted import compute_class_weights, WeightedSoftLabelKLDivergence
from utils.metrics import evaluate_predictions


CONFIG = {
    "train_split": os.path.join(PROJECT_ROOT, "data", "processed", "splits", "train.json"),
    "val_split": os.path.join(PROJECT_ROOT, "data", "processed", "splits", "val.json"),

    "num_classes": 8,
    "pretrained": True,
    "dropout": 0.3,

    "batch_size": 8,          # conservative for 6GB VRAM
    "num_epochs": 20,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,

    "num_workers": 0,
    "patience": 5,
    "seed": 42,

    "use_class_weighting": True,   # matches your best pipeline so far

    "checkpoint_dir": os.path.join(PROJECT_ROOT, "models", "checkpoints", "swin"),
    "history_path": os.path.join(PROJECT_ROOT, "models", "swin_training_history.json"),
    "reports_dir": os.path.join(PROJECT_ROOT, "reports"),
    "run_name": "riva_swin_tiny",
}


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")
    return device


def create_transforms():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()

    running_loss = 0.0
    total_samples = 0
    all_preds, all_targets = [], []

    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, soft_labels, _ in loader:
            images = images.to(device, non_blocking=True)
            soft_labels = soft_labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, soft_labels)

            if train:
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(logits, dim=1)
            targets = torch.argmax(soft_labels, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())

    epoch_loss = running_loss / total_samples
    accuracy = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(
        all_targets, all_preds,
        labels=list(range(NUM_CLASSES)),
        average="macro", zero_division=0,
    )

    return epoch_loss, accuracy, macro_f1, all_preds, all_targets


def train():
    print("\n" + "=" * 70)
    print("RIVA SWIN-TINY TRAINING")
    print("=" * 70)

    torch.manual_seed(CONFIG["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG["seed"])

    device = get_device()

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["reports_dir"], exist_ok=True)

    train_transform, val_transform = create_transforms()

    print("\nLoading datasets...")
    train_dataset = RIVADataset(split_file=CONFIG["train_split"], transform=train_transform)
    val_dataset = RIVADataset(split_file=CONFIG["val_split"], transform=val_transform)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=CONFIG["num_workers"], pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG["num_workers"], pin_memory=torch.cuda.is_available(),
    )

    print("\nCreating Swin-Tiny model...")
    model = create_model(pretrained=CONFIG["pretrained"], num_classes=CONFIG["num_classes"], dropout=CONFIG["dropout"])
    model = model.to(device)
    print("Swin-Tiny created.")

    if CONFIG["use_class_weighting"]:
        class_weights = compute_class_weights(train_dataset, method="inverse_freq", cap=10.0).to(device)
        criterion = WeightedSoftLabelKLDivergence(class_weights)
        print("\nLoss: Weighted KL Divergence (class-weighted)")
    else:
        criterion = SoftLabelKLDivergence()
        print("\nLoss: KL Divergence (unweighted)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history = {"train_loss": [], "train_accuracy": [], "train_macro_f1": [],
               "val_loss": [], "val_accuracy": [], "val_macro_f1": [], "learning_rate": []}

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_preds, best_targets = None, None

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        print(f"\n{'-' * 70}")
        print(f"Epoch {epoch}/{CONFIG['num_epochs']}")

        train_loss, train_acc, train_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_f1, val_preds, val_targets = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["train_macro_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_macro_f1"].append(val_f1)
        history["learning_rate"].append(current_lr)

        print(f"\nTrain Loss      : {train_loss:.4f}")
        print(f"Train Accuracy  : {train_acc:.4f}")
        print(f"Train Macro F1  : {train_f1:.4f}")
        print(f"Val Loss        : {val_loss:.4f}")
        print(f"Val Accuracy    : {val_acc:.4f}")
        print(f"Val Macro F1    : {val_f1:.4f}")
        print(f"Learning Rate   : {current_lr:.6f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            best_preds, best_targets = val_preds, val_targets

            checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_f1,
                "class_names": CLASS_NAMES,
                "config": CONFIG,
            }, checkpoint_path)

            print(f"\n✓ Best model saved. Validation Macro F1: {val_f1:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"\nNo improvement. ({epochs_without_improvement}/{CONFIG['patience']})")

        if epochs_without_improvement >= CONFIG["patience"]:
            print("\nEarly stopping triggered.")
            break

    history["best_epoch"] = best_epoch
    history["best_val_macro_f1"] = best_val_f1

    with open(CONFIG["history_path"], "w") as f:
        json.dump(history, f, indent=4)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation Macro F1: {best_val_f1:.4f}")
    print("=" * 70)

    # Final per-class evaluation using the best epoch's predictions
    evaluate_predictions(
        y_true=best_targets,
        y_pred=best_preds,
        class_names=CLASS_NAMES,
        output_dir=CONFIG["reports_dir"],
        run_name=CONFIG["run_name"],
    )


if __name__ == "__main__":
    train()