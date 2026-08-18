"""
Pretrain EfficientNet-B3 Backbone on SIPaKMeD

Purpose:
    Trains the same RIVAEfficientNet architecture on SIPaKMeD
    (class-mapped to RIVA's scheme), then saves ONLY the backbone
    weights (not the classifier head) for later loading into a
    fresh RIVA fine-tuning run.

Why backbone-only:
    SIPaKMeD's mapped labels are a simplification (5 classes ->
    3 RIVA classes used). We don't want to keep SIPaKMeD's
    classifier head, since that head has learned to separate a
    different, coarser label set. What we DO want is the feature
    extractor (backbone) to have learned useful general cervical-
    cell visual features before it ever sees the real RIVA data.

Usage:
    python src/training/pretrain_sipakmed.py
"""

import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from sklearn.metrics import accuracy_score, f1_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset.sipakmed_dataset import SIPaKMeDDataset, CLASS_NAMES, NUM_CLASSES
from models.efficientnet import create_model
from training.loss import SoftLabelCrossEntropy


CONFIG = {
    "sipakmed_root": os.path.join(PROJECT_ROOT, "data", "raw", "sipakmed"),

    "val_fraction": 0.15,  # held-out fraction of SIPaKMeD for monitoring only

    "batch_size": 16,
    "num_epochs": 15,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,

    "num_workers": 0,
    "seed": 42,

    "output_path": os.path.join(
        PROJECT_ROOT, "models", "checkpoints", "sipakmed_backbone.pth"
    ),
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
    # Same normalization as RIVA training, so features transfer
    # cleanly. Lighter geometric augmentation is fine here since
    # this is just pretraining, not the final result.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return transform


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

    return epoch_loss, accuracy, macro_f1


def main():
    torch.manual_seed(CONFIG["seed"])

    device = get_device()

    os.makedirs(os.path.dirname(CONFIG["output_path"]), exist_ok=True)

    transform = create_transforms()

    print("\n" + "=" * 70)
    print("LOADING SIPAKMED")
    print("=" * 70)

    full_dataset = SIPaKMeDDataset(
        root_dir=CONFIG["sipakmed_root"],
        transform=transform,
    )

    val_size = int(len(full_dataset) * CONFIG["val_fraction"])
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG["seed"]),
    )

    print(f"\nTrain samples: {train_size}")
    print(f"Val samples (monitoring only): {val_size}")

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"],
        shuffle=True, num_workers=CONFIG["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"],
        shuffle=False, num_workers=CONFIG["num_workers"],
    )

    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)

    # Full 8-class model, ImageNet-pretrained backbone as starting
    # point -- we're pretraining the backbone further on cervical
    # cytology data before it ever sees RIVA.
    model = create_model(pretrained=True, num_classes=NUM_CLASSES, dropout=0.3)
    model = model.to(device)

    criterion = SoftLabelCrossEntropy()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    print("\n" + "=" * 70)
    print("PRETRAINING ON SIPAKMED")
    print("=" * 70)

    best_val_f1 = -1.0

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )

        print(
            f"\nEpoch {epoch}/{CONFIG['num_epochs']}  "
            f"Train: loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f}  "
            f"Val: loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

            # ------------------------------------------------------
            # Save BACKBONE ONLY -- strip the classifier head.
            # This is the key step: we only want the feature
            # extractor's learned weights, not SIPaKMeD's classifier.
            # ------------------------------------------------------

            backbone_state_dict = {
                k: v for k, v in model.state_dict().items()
                if not k.startswith("backbone.classifier")
            }

            torch.save(
                {
                    "backbone_state_dict": backbone_state_dict,
                    "source_dataset": "sipakmed",
                    "val_macro_f1": val_f1,
                    "epoch": epoch,
                    "class_mapping_note": (
                        "Superficial/Parabasal->NILM, Koilocytotic->LSIL, "
                        "Dyskeratotic->HSIL. Backbone weights only; "
                        "classifier head excluded."
                    ),
                },
                CONFIG["output_path"],
            )

            print(f"  Saved backbone checkpoint (val macro F1={val_f1:.4f})")

    print("\n" + "=" * 70)
    print("SIPAKMED PRETRAINING COMPLETE")
    print(f"Best val macro F1: {best_val_f1:.4f}")
    print(f"Backbone checkpoint saved to: {CONFIG['output_path']}")
    print("=" * 70)


if __name__ == "__main__":
    main()