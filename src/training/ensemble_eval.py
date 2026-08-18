"""
RIVA Ensemble Evaluation

Purpose:
    Combines predictions from your two best models --
    EfficientNet-B3 (SIPaKMeD-transfer) and Swin-Tiny -- via
    probability averaging (soft voting), and evaluates the
    combined result the same way as every previous run.

Why probability averaging:
    Simple, standard, and doesn't require retraining anything.
    Each model outputs a probability distribution over 8 classes;
    we average the two distributions per sample, then take the
    argmax. This works best when the two models make DIFFERENT
    kinds of errors -- which your results suggest, since Swin is
    notably stronger on HSIL/INFL while EfficientNet+SIPaKMeD has
    its own distinct error pattern.

Usage:
    python src/training/ensemble_eval.py
"""

import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset.riva_dataset import RIVADataset
from models.efficientnet import create_model as create_efficientnet, CLASS_NAMES, NUM_CLASSES
from models.swin import create_model as create_swin
from utils.metrics import evaluate_predictions


CONFIG = {
    "val_split": os.path.join(PROJECT_ROOT, "data", "processed", "splits", "val.json"),

    # Point these at your best checkpoints from each architecture.
    "efficientnet_checkpoint": os.path.join(
        PROJECT_ROOT, "models", "checkpoints", "best_model.pth"
    ),  # this should be the riva_sipakmed_transfer checkpoint --
        # confirm this is the right file before running (see note below)

    "swin_checkpoint": os.path.join(
        PROJECT_ROOT, "models", "checkpoints", "swin", "best_model.pth"
    ),

    "batch_size": 8,
    "num_workers": 0,

    "reports_dir": os.path.join(PROJECT_ROOT, "reports"),
    "run_name": "riva_ensemble_swin_weighted",

    # Weight given to EfficientNet vs Swin when averaging probabilities.
    # 0.5/0.5 = equal weight. Adjust later if one model is clearly
    # stronger overall (try 0.4/0.6 favoring Swin, given its higher
    # accuracy, as a follow-up experiment).
    "efficientnet_weight": 0.3,
    "swin_weight": 0.7,
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


def load_model(model_fn, checkpoint_path, device):
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = model_fn(pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    if "val_macro_f1" in checkpoint:
        print(f"  Checkpoint's recorded val macro F1: {checkpoint['val_macro_f1']:.4f}")
    if "epoch" in checkpoint:
        print(f"  Checkpoint's epoch: {checkpoint['epoch']}")

    return model


def main():
    print("\n" + "=" * 70)
    print("RIVA ENSEMBLE EVALUATION")
    print("=" * 70)

    device = get_device()

    # ----------------------------------------------------------
    # IMPORTANT: verify checkpoint paths before running
    # ----------------------------------------------------------
    # If you ran multiple EfficientNet experiments (baseline,
    # weighted_loss, sipakmed_transfer, gradual_unfreeze), remember
    # each one OVERWROTE models/checkpoints/best_model.pth, since
    # train.py always saves to the same filename. Confirm the file
    # currently at that path is actually from your sipakmed_transfer
    # run (the one you want in the ensemble) and not a later run
    # (like gradual_unfreeze, which was worse) that overwrote it.
    #
    # If you're not sure which run is currently saved there, you'll
    # need to re-run riva_sipakmed_transfer's config once more before
    # this script, so the correct weights are the ones on disk.
    # ----------------------------------------------------------

    print("\n" + "!" * 70)
    print("CHECKPOINT CHECK: confirm models/checkpoints/best_model.pth")
    print("is currently your riva_sipakmed_transfer run, not a later")
    print("(possibly worse) run that overwrote it. See script comments.")
    print("!" * 70)

    efficientnet_model = load_model(create_efficientnet, CONFIG["efficientnet_checkpoint"], device)
    swin_model = load_model(create_swin, CONFIG["swin_checkpoint"], device)

    val_transform = create_val_transform()
    val_dataset = RIVADataset(split_file=CONFIG["val_split"], transform=val_transform)

    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG["num_workers"],
    )

    print(f"\nValidation samples: {len(val_dataset)}")
    print(f"Ensemble weights -- EfficientNet: {CONFIG['efficientnet_weight']}, "
          f"Swin: {CONFIG['swin_weight']}")

    all_predictions = []
    all_targets = []

    print("\nRunning ensemble inference...")

    with torch.no_grad():
        for images, soft_labels, _ in val_loader:
            images = images.to(device, non_blocking=True)
            soft_labels = soft_labels.to(device, non_blocking=True)

            eff_logits = efficientnet_model(images)
            swin_logits = swin_model(images)

            # Convert each model's logits to probabilities BEFORE
            # averaging -- averaging raw logits isn't meaningful
            # since the two architectures' logit scales aren't
            # necessarily comparable.
            eff_probs = F.softmax(eff_logits, dim=1)
            swin_probs = F.softmax(swin_logits, dim=1)

            ensemble_probs = (
                CONFIG["efficientnet_weight"] * eff_probs
                + CONFIG["swin_weight"] * swin_probs
            )

            predictions = torch.argmax(ensemble_probs, dim=1)
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


if __name__ == "__main__":
    main()