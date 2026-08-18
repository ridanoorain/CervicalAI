
"""
RIVA EfficientNet-B3 Training

Purpose:
    Train EfficientNet-B3 for cervical cytology classification
    using multi-expert soft-label distributions.

Pipeline:

    RIVADataset
        ↓
    EfficientNet-B3
        ↓
    Soft-label Cross Entropy
        ↓
    Validation
        ↓
    Best model checkpoint

The TEST set is NOT used during training.

Classes:
    0 -> NILM
    1 -> INFL
    2 -> LSIL
    3 -> HSIL
    4 -> SCC
    5 -> ENDO
    6 -> ASCH
    7 -> ASCUS
"""

import os
import sys
import json
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import f1_score, accuracy_score
from training.loss_weighted import (
    compute_class_weights,
    WeightedSoftLabelKLDivergence,
    build_weighted_sampler,
)
from utils.metrics import evaluate_predictions
from training.finetune_utils import (
    freeze_backbone,
    unfreeze_backbone,
    create_differential_optimizer,
)

# ============================================================
# PROJECT PATH
# ============================================================

# Add src directory to Python path so that this file works
# when executed from the project root.

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

SRC_DIR = os.path.dirname(CURRENT_DIR)

PROJECT_ROOT = os.path.dirname(SRC_DIR)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from dataset.riva_dataset import RIVADataset
from models.efficientnet import create_model, CLASS_NAMES
from training.loss import SoftLabelKLDivergence


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    "train_split": os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "splits",
        "train.json"
    ),

    "val_split": os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "splits",
        "val.json"
    ),

    "test_split": os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "splits",
        "test.json"
    ),

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    "num_classes": 8,

    "pretrained": True,

    "dropout": 0.3,

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    "batch_size": 16,

    "num_epochs": 20,

    "learning_rate": 1e-4,

    "weight_decay": 1e-4,

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    "num_workers": 0,

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    "patience": 5,

    # --------------------------------------------------------
    # Random seed
    # --------------------------------------------------------

    "seed": 42,
    
    # --------------------------------------------------------
    # Class imbalance handling 
    # --------------------------------------------------------
    "freeze_backbone_epochs": 0,  
    "backbone_lr": 1e-4,           
    "use_class_weighting": True, 
    "pretrained_backbone_path": os.path.join(
        PROJECT_ROOT, "models", "checkpoints", "sipakmed_backbone.pth"),     
    "use_weighted_sampler": False,    
                                   
    "reports_dir": os.path.join(PROJECT_ROOT, "reports"),
    "run_name": "riva_sipakmed_transfer_v2", 
    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    "checkpoint_dir": os.path.join(
        PROJECT_ROOT,
        "models",
        "checkpoints"
    ),

    "history_path": os.path.join(
        PROJECT_ROOT,
        "models",
        "training_history.json"
    ),
}


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed=42):
    """
    Make training as reproducible as possible.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    # These settings improve reproducibility.

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    else:

        device = torch.device("cpu")

        print("Using device: CPU")

    return device


# ============================================================
# TRANSFORMS
# ============================================================

def create_transforms():

    """
    Create image transformations.

    RIVA cell crops are already 224x224.

    Training:
        - horizontal flip
        - vertical flip
        - small rotation
        - ImageNet normalization

    Validation:
        - ImageNet normalization only
    """

    train_transform = transforms.Compose([

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomVerticalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=15
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])

    val_transform = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])

    return train_transform, val_transform


# ============================================================
# DATASET CREATION
# ============================================================

def create_datasets(
    train_transform,
    val_transform
):

    print("\n" + "=" * 70)
    print("LOADING DATASETS")
    print("=" * 70)

    print(
        f"\nTrain split:\n"
        f"{CONFIG['train_split']}"
    )

    train_dataset = RIVADataset(
        split_file=CONFIG["train_split"],
        transform=train_transform
    )

    print(
        f"Train samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"\nValidation split:\n"
        f"{CONFIG['val_split']}"
    )

    val_dataset = RIVADataset(
        split_file=CONFIG["val_split"],
        transform=val_transform
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    return train_dataset, val_dataset


# ============================================================
# DATALOADERS
# ============================================================

def create_dataloaders(
    train_dataset,
    val_dataset
):

    train_loader = DataLoader(

        train_dataset,

        batch_size=CONFIG["batch_size"],

        shuffle=True,

        num_workers=CONFIG["num_workers"],

        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=CONFIG["batch_size"],

        shuffle=False,

        num_workers=CONFIG["num_workers"],

        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    all_predictions = []

    all_targets = []

    total_samples = 0

    for images, soft_labels, _ in loader:

        # ----------------------------------------------------
        # Move data to device
        # ----------------------------------------------------

        images = images.to(
            device,
            non_blocking=True
        )

        soft_labels = soft_labels.to(
            device,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logits = model(images)

        # ----------------------------------------------------
        # Soft-label loss
        # ----------------------------------------------------

        loss = criterion(
            logits,
            soft_labels
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Update weights
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        batch_size = images.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

        # ----------------------------------------------------
        # Majority-label predictions
        # ----------------------------------------------------

        predictions = torch.argmax(
            logits,
            dim=1
        )

        targets = torch.argmax(
            soft_labels,
            dim=1
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        all_targets.extend(
            targets.detach()
            .cpu()
            .numpy()
        )

    # --------------------------------------------------------
    # Average loss
    # --------------------------------------------------------

    epoch_loss = (
        running_loss / total_samples
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        all_targets,
        all_predictions
    )

    # --------------------------------------------------------
    # Macro F1
    # --------------------------------------------------------

    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        labels=list(range(CONFIG["num_classes"])),
        average="macro",
        zero_division=0
    )

    return (
        epoch_loss,
        accuracy,
        macro_f1
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    all_predictions = []

    all_targets = []

    total_samples = 0

    with torch.no_grad():

        for images, soft_labels, _ in loader:

            # ------------------------------------------------
            # Move data
            # ------------------------------------------------

            images = images.to(
                device,
                non_blocking=True
            )

            soft_labels = soft_labels.to(
                device,
                non_blocking=True
            )

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            logits = model(images)

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = criterion(
                logits,
                soft_labels
            )

            batch_size = images.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            predictions = torch.argmax(
                logits,
                dim=1
            )

            targets = torch.argmax(
                soft_labels,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                targets.cpu().numpy()
            )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    epoch_loss = (
        running_loss / total_samples
    )

    accuracy = accuracy_score(
        all_targets,
        all_predictions
    )

    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        labels=list(range(CONFIG["num_classes"])),
        average="macro",
        zero_division=0
    )

    return (
        epoch_loss,
        accuracy,
        macro_f1
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_loss,
    val_loss,
    val_accuracy,
    val_f1,
    path
):

    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "train_loss":
            train_loss,

        "val_loss":
            val_loss,

        "val_accuracy":
            val_accuracy,

        "val_macro_f1":
            val_f1,

        "class_names":
            CLASS_NAMES,

        "num_classes":
            CONFIG["num_classes"],

        "config":
            CONFIG,
    }

    torch.save(
        checkpoint,
        path
    )


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train():

    print("\n" + "=" * 70)
    print("RIVA EFFICIENTNET-B3 TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(
        CONFIG["seed"]
    )

    print(
        f"\nRandom seed: "
        f"{CONFIG['seed']}"
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        CONFIG["checkpoint_dir"],
        exist_ok=True
    )

    # --------------------------------------------------------
    # Transforms
    # --------------------------------------------------------

    train_transform, val_transform = (
        create_transforms()
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset, val_dataset = (
        create_datasets(
            train_transform,
            val_transform
        )
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader, val_loader = (
        create_dataloaders(
            train_dataset,
            val_dataset
        )
    )

    print("\nDataLoader information:")

    print(
        f"  Train batches: "
        f"{len(train_loader)}"
    )

    print(
        f"  Validation batches: "
        f"{len(val_loader)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)

    model = create_model(

        pretrained=CONFIG["pretrained"],

        num_classes=CONFIG["num_classes"],

        dropout=CONFIG["dropout"]
    )

    model = model.to(device)
    if CONFIG["pretrained_backbone_path"]:
        print(f"\nLoading SIPaKMeD-pretrained backbone from: "
              f"{CONFIG['pretrained_backbone_path']}")

        backbone_checkpoint = torch.load(
            CONFIG["pretrained_backbone_path"],
            map_location=device,
        )

        # load_state_dict with strict=False because we're only
        # loading backbone weights -- the classifier head keys are
        # intentionally missing from this checkpoint and will keep
        # their fresh ImageNet-init / random-init values instead.
        missing, unexpected = model.load_state_dict(
            backbone_checkpoint["backbone_state_dict"],
            strict=False,
        )

        print(f"  Loaded. Missing keys (expected: classifier head): "
              f"{len(missing)}")
        print(f"  Unexpected keys (should be 0): {len(unexpected)}")

        if len(unexpected) > 0:
            print(f"  WARNING: unexpected keys found: {unexpected}")
    else:
        print("\nNo SIPaKMeD backbone specified -- training from "
              "ImageNet-pretrained weights only.")
        
    if CONFIG["pretrained_backbone_path"] and CONFIG["freeze_backbone_epochs"] > 0:
        freeze_backbone(model)
        print(f"Backbone will stay frozen for the first "
              f"{CONFIG['freeze_backbone_epochs']} epoch(s).")
    print("\nEfficientNet-B3 created.")

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    

    if CONFIG["use_class_weighting"]:
        class_weights = compute_class_weights(
            train_dataset,
            method="inverse_freq",
            cap=10.0,
        )
        class_weights = class_weights.to(device)
        criterion = WeightedSoftLabelKLDivergence(class_weights)
        print("\nLoss: Weighted KL Divergence (class-weighted)")
    else:
        criterion = SoftLabelKLDivergence()
        print("\nLoss: KL Divergence (unweighted, baseline)")
    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    if CONFIG["pretrained_backbone_path"]:
        optimizer = create_differential_optimizer(
            model,
            backbone_lr=CONFIG["backbone_lr"],
            head_lr=CONFIG["learning_rate"],
            weight_decay=CONFIG["weight_decay"],
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CONFIG["learning_rate"],
            weight_decay=CONFIG["weight_decay"],
        )

    print(
        f"Learning rate: "
        f"{CONFIG['learning_rate']}"
    )

    print(
        f"Weight decay: "
        f"{CONFIG['weight_decay']}"
    )

    # --------------------------------------------------------
    # Learning rate scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=2
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history = {

        "train_loss": [],

        "train_accuracy": [],

        "train_macro_f1": [],

        "val_loss": [],

        "val_accuracy": [],

        "val_macro_f1": [],

        "learning_rate": []
    }

    # --------------------------------------------------------
    # Best model tracking
    # --------------------------------------------------------

    best_val_f1 = -1.0

    best_epoch = 0

    epochs_without_improvement = 0

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        CONFIG["num_epochs"] + 1
    ):

        print(
            f"\n{'-' * 70}"
        )

        print(
            f"Epoch "
            f"{epoch}/"
            f"{CONFIG['num_epochs']}"
        )
        
        # ----------------------------------------------------
        # Unfreeze backbone once the warmup period ends
        # ----------------------------------------------------

        if (
            CONFIG["pretrained_backbone_path"]
            and CONFIG["freeze_backbone_epochs"] > 0
            and epoch == CONFIG["freeze_backbone_epochs"] + 1
        ):
            unfreeze_backbone(model)

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        train_loss, train_accuracy, train_f1 = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss, val_accuracy, val_f1 = (
            validate(
                model,
                val_loader,
                criterion,
                device
            )
        )

        # ----------------------------------------------------
        # Current learning rate
        # ----------------------------------------------------

        current_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------
        # Scheduler
        #
        # We want to maximize validation F1.
        # ----------------------------------------------------

        scheduler.step(
            val_f1
        )

        # ----------------------------------------------------
        # Store history
        # ----------------------------------------------------

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["train_macro_f1"].append(
            train_f1
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_accuracy"].append(
            val_accuracy
        )

        history["val_macro_f1"].append(
            val_f1
        )

        history["learning_rate"].append(
            current_lr
        )

        # ----------------------------------------------------
        # Print metrics
        # ----------------------------------------------------

        print(
            f"\nTrain Loss      : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy  : "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Train Macro F1  : "
            f"{train_f1:.4f}"
        )

        print(
            f"Val Loss        : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Accuracy    : "
            f"{val_accuracy:.4f}"
        )

        print(
            f"Val Macro F1    : "
            f"{val_f1:.4f}"
        )

        print(
            f"Learning Rate   : "
            f"{current_lr:.6f}"
        )

        # ----------------------------------------------------
        # Save best model
        #
        # Macro F1 is used because the dataset is imbalanced.
        # ----------------------------------------------------

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            best_epoch = epoch

            epochs_without_improvement = 0

            best_checkpoint_path = os.path.join(

                CONFIG["checkpoint_dir"],

                "best_model.pth"
            )

            save_checkpoint(

                model=model,

                optimizer=optimizer,

                epoch=epoch,

                train_loss=train_loss,

                val_loss=val_loss,

                val_accuracy=val_accuracy,

                val_f1=val_f1,

                path=best_checkpoint_path
            )

            print(
                "\n✓ Best model saved."
            )

            print(
                f"  Validation Macro F1: "
                f"{val_f1:.4f}"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"\nNo improvement."
                f" ({epochs_without_improvement}/"
                f"{CONFIG['patience']})"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= CONFIG["patience"]
        ):

            print(
                "\nEarly stopping triggered."
            )

            break
    # ------------------------------------------------------------
    # Final per-class evaluation using the best checkpoint
    # ------------------------------------------------------------

    print("\nLoading best checkpoint for final per-class evaluation...")

    best_checkpoint = torch.load(
        os.path.join(CONFIG["checkpoint_dir"], "best_model.pth"),
        map_location=device,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, soft_labels, _ in val_loader:
            images = images.to(device, non_blocking=True)
            soft_labels = soft_labels.to(device, non_blocking=True)

            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            targets = torch.argmax(soft_labels, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    evaluate_predictions(
        y_true=all_targets,
        y_pred=all_predictions,
        class_names=CLASS_NAMES,
        output_dir=CONFIG["reports_dir"],
        run_name=CONFIG["run_name"],
    )
    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    history["best_epoch"] = best_epoch

    history["best_val_macro_f1"] = (
        best_val_f1
    )

    history["epochs_completed"] = len(
        history["train_loss"]
    )

    with open(
        CONFIG["history_path"],
        "w"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )

    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"\nBest epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation Macro F1: "
        f"{best_val_f1:.4f}"
    )

    print(
        f"\nBest model saved to:"
    )

    print(
        CONFIG["checkpoint_dir"]
        + "\\best_model.pth"
    )

    print(
        f"\nTraining history saved to:"
    )

    print(
        CONFIG["history_path"]
    )

    print("\nIMPORTANT:")

    print(
        "The test set was NOT used during training."
    )

    print(
        "Use the saved best model for final test evaluation."
    )

    print("\n" + "=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    train()

