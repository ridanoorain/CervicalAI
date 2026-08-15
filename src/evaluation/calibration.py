import os
import sys
import json
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATH
# ============================================================

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


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {

    # --------------------------------------------------------
    # Test split
    # --------------------------------------------------------

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

    "checkpoint": os.path.join(
        PROJECT_ROOT,
        "models",
        "checkpoints",
        "best_model.pth"
    ),

    "num_classes": 8,

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    "batch_size": 16,

    "num_workers": 0,

    # --------------------------------------------------------
    # ECE
    # --------------------------------------------------------

    "num_bins": 10,

    # --------------------------------------------------------
    # Per-class ECE
    # --------------------------------------------------------

    # Classes with fewer than this many predicted samples still get
    # reported, but with a note that the estimate is unreliable.
    "min_class_samples_for_ece": 10,

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    "output_dir": os.path.join(
        PROJECT_ROOT,
        "models",
        "evaluation",
        "calibration"
    ),
}


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print(
            f"Device: GPU "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        device = torch.device("cpu")

        print("Device: cpu")

    return device


# ============================================================
# TEST TRANSFORM
# ============================================================

def create_transform():

    return transforms.Compose([

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


# ============================================================
# LOAD TEST DATASET
# ============================================================

def load_test_dataset():

    print("\n" + "=" * 70)
    print("LOADING TEST DATASET")
    print("=" * 70)

    print(
        f"\nTest split:\n"
        f"{CONFIG['test_split']}"
    )

    dataset = RIVADataset(
        split_file=CONFIG["test_split"],
        transform=create_transform()
    )

    print(
        f"\nTest samples: "
        f"{len(dataset)}"
    )

    return dataset


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(device):

    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    print(
        f"\nCheckpoint:\n"
        f"{CONFIG['checkpoint']}"
    )

    model = create_model(
        pretrained=False,
        num_classes=CONFIG["num_classes"],
        dropout=0.3
    )

    checkpoint = torch.load(
        CONFIG["checkpoint"],
        map_location=device
    )

    # --------------------------------------------------------
    # Support both checkpoint formats
    # --------------------------------------------------------

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model = model.to(device)

    model.eval()

    print("\nModel loaded successfully.")

    return model


# ============================================================
# COLLECT PREDICTIONS
# ============================================================

def collect_predictions(
    model,
    loader,
    device
):

    print("\n" + "=" * 70)
    print("COLLECTING TEST PREDICTIONS")
    print("=" * 70)

    all_probabilities = []

    all_predictions = []

    all_targets = []

    with torch.no_grad():

        for images, soft_labels, _ in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            soft_labels = soft_labels.to(
                device,
                non_blocking=True
            )

            # ------------------------------------------------
            # Model output
            # ------------------------------------------------

            logits = model(images)

            # ------------------------------------------------
            # Convert logits to probabilities
            # ------------------------------------------------

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            # ------------------------------------------------
            # Predicted class
            # ------------------------------------------------

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            # ------------------------------------------------
            # Majority class from soft label
            # ------------------------------------------------

            targets = torch.argmax(
                soft_labels,
                dim=1
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_predictions.append(
                predictions.cpu().numpy()
            )

            all_targets.append(
                targets.cpu().numpy()
            )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0
    )

    predictions = np.concatenate(
        all_predictions,
        axis=0
    )

    targets = np.concatenate(
        all_targets,
        axis=0
    )

    print(
        f"\nTotal samples: "
        f"{len(targets)}"
    )

    return (
        probabilities,
        predictions,
        targets
    )


# ============================================================
# EXPECTED CALIBRATION ERROR
# ============================================================

def calculate_ece(
    probabilities,
    predictions,
    targets,
    num_bins=10
):

    # --------------------------------------------------------
    # Confidence = highest predicted probability
    # --------------------------------------------------------

    confidences = np.max(
        probabilities,
        axis=1
    )

    # --------------------------------------------------------
    # Correctness
    # --------------------------------------------------------

    correct = (
        predictions == targets
    ).astype(float)

    # --------------------------------------------------------
    # Equal-width confidence bins
    # --------------------------------------------------------

    bin_edges = np.linspace(
        0.0,
        1.0,
        num_bins + 1
    )

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    ece = 0.0

    total_samples = len(
        confidences
    )

    print("\n" + "=" * 70)
    print("ECE CALCULATION")
    print("=" * 70)

    print(
        "\nBin        Count    Accuracy    Confidence"
    )

    print(
        "-" * 50
    )

    for i in range(num_bins):

        lower = bin_edges[i]

        upper = bin_edges[i + 1]

        # ----------------------------------------------------
        # Include 1.0 in final bin
        # ----------------------------------------------------

        if i == num_bins - 1:

            mask = (
                (confidences >= lower)
                &
                (confidences <= upper)
            )

        else:

            mask = (
                (confidences >= lower)
                &
                (confidences < upper)
            )

        count = np.sum(mask)

        bin_counts.append(
            int(count)
        )

        # ----------------------------------------------------
        # Empty bin
        # ----------------------------------------------------

        if count == 0:

            bin_accuracies.append(
                np.nan
            )

            bin_confidences.append(
                np.nan
            )

            print(
                f"{lower:.1f}-{upper:.1f}"
                f"      0"
                f"        -"
                f"           -"
            )

            continue

        # ----------------------------------------------------
        # Accuracy inside bin
        # ----------------------------------------------------

        accuracy = np.mean(
            correct[mask]
        )

        # ----------------------------------------------------
        # Average confidence inside bin
        # ----------------------------------------------------

        confidence = np.mean(
            confidences[mask]
        )

        bin_accuracies.append(
            float(accuracy)
        )

        bin_confidences.append(
            float(confidence)
        )

        # ----------------------------------------------------
        # ECE contribution
        # ----------------------------------------------------

        ece += (
            count / total_samples
        ) * abs(
            accuracy - confidence
        )

        print(
            f"{lower:.1f}-{upper:.1f}"
            f"      {count:4d}"
            f"      {accuracy:.4f}"
            f"       {confidence:.4f}"
        )

    return (
        float(ece),
        bin_accuracies,
        bin_confidences,
        bin_counts
    )


# ============================================================
# PER-CLASS EXPECTED CALIBRATION ERROR
# ============================================================

def calculate_per_class_ece(
    probabilities,
    predictions,
    targets,
    class_names,
    num_bins=10,
    min_samples=10
):

    # --------------------------------------------------------
    # Same binning logic as calculate_ece, but restricted to
    # samples where the model PREDICTED a given class. This
    # answers: "when the model says LSIL, how trustworthy is
    # its confidence?" - which the pooled ECE hides for rare
    # classes since they get swamped by NILM / INFL volume.
    # --------------------------------------------------------

    confidences = np.max(
        probabilities,
        axis=1
    )

    correct = (
        predictions == targets
    ).astype(float)

    bin_edges = np.linspace(
        0.0,
        1.0,
        num_bins + 1
    )

    per_class_results = {}

    print("\n" + "=" * 70)
    print("PER-CLASS ECE CALCULATION")
    print("=" * 70)

    print(
        "\nClass        N       ECE       Mean Conf   Mean Acc   Note"
    )

    print(
        "-" * 70
    )

    for class_index, class_name in enumerate(
        class_names
    ):

        class_mask = (
            predictions == class_index
        )

        n_class = int(
            np.sum(class_mask)
        )

        # ----------------------------------------------------
        # No predictions made for this class at all
        # ----------------------------------------------------

        if n_class == 0:

            per_class_results[class_name] = {
                "n": 0,
                "ece": None,
                "mean_confidence": None,
                "mean_accuracy": None,
                "note": "no predictions in this class"
            }

            print(
                f"{class_name:12s}"
                f" {n_class:4d}"
                f"       -"
                f"           -"
                f"          -"
                f"        no predictions"
            )

            continue

        class_confidences = confidences[class_mask]

        class_correct = correct[class_mask]

        class_ece = 0.0

        for i in range(num_bins):

            lower = bin_edges[i]

            upper = bin_edges[i + 1]

            if i == num_bins - 1:

                bin_mask = (
                    (class_confidences >= lower)
                    &
                    (class_confidences <= upper)
                )

            else:

                bin_mask = (
                    (class_confidences >= lower)
                    &
                    (class_confidences < upper)
                )

            bin_count = np.sum(bin_mask)

            if bin_count == 0:

                continue

            bin_accuracy = np.mean(
                class_correct[bin_mask]
            )

            bin_confidence = np.mean(
                class_confidences[bin_mask]
            )

            class_ece += (
                bin_count / n_class
            ) * abs(
                bin_accuracy - bin_confidence
            )

        mean_confidence = float(
            np.mean(class_confidences)
        )

        mean_accuracy = float(
            np.mean(class_correct)
        )

        note = (
            ""
            if n_class >= min_samples
            else f"low n (<{min_samples}), unreliable"
        )

        per_class_results[class_name] = {
            "n": n_class,
            "ece": float(class_ece),
            "mean_confidence": mean_confidence,
            "mean_accuracy": mean_accuracy,
            "note": note
        }

        print(
            f"{class_name:12s}"
            f" {n_class:4d}"
            f"    {class_ece:.4f}"
            f"      {mean_confidence:.4f}"
            f"      {mean_accuracy:.4f}"
            f"     {note}"
        )

    return per_class_results


# ============================================================
# RELIABILITY DIAGRAM
# ============================================================

def create_reliability_diagram(
    bin_accuracies,
    bin_confidences,
    ece,
    output_path
):

    valid_accuracies = []

    valid_confidences = []

    for accuracy, confidence in zip(
        bin_accuracies,
        bin_confidences
    ):

        if (
            not np.isnan(accuracy)
            and
            not np.isnan(confidence)
        ):

            valid_accuracies.append(
                accuracy
            )

            valid_confidences.append(
                confidence
            )

    plt.figure(
        figsize=(8, 7)
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    plt.plot(
        valid_confidences,
        valid_accuracies,
        marker="o",
        linewidth=2,
        label="Model"
    )

    plt.xlabel(
        "Mean Predicted Confidence"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        f"Reliability Diagram\nECE = {ece:.4f}"
    )

    plt.xlim(
        0,
        1
    )

    plt.ylim(
        0,
        1
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"\nSaved: "
        f"{output_path}"
    )


# ============================================================
# PER-CLASS ECE BAR CHART
# ============================================================

def create_per_class_ece_chart(
    per_class_results,
    output_path
):

    class_names = []

    class_eces = []

    class_ns = []

    for class_name, stats in per_class_results.items():

        if stats["ece"] is None:

            continue

        class_names.append(
            class_name
        )

        class_eces.append(
            stats["ece"]
        )

        class_ns.append(
            stats["n"]
        )

    if len(class_names) == 0:

        print(
            "\nNo classes with predictions - skipping per-class ECE chart."
        )

        return

    plt.figure(
        figsize=(9, 6)
    )

    bars = plt.bar(
        class_names,
        class_eces,
        color="steelblue",
        edgecolor="black"
    )

    for bar, n in zip(
        bars,
        class_ns
    ):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    plt.axhline(
        y=0.05,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Excellent threshold (0.05)"
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "ECE (within predicted class)"
    )

    plt.title(
        "Per-Class Expected Calibration Error"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"\nSaved: "
        f"{output_path}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    ece,
    bin_accuracies,
    bin_confidences,
    bin_counts,
    per_class_results
):

    output_path = os.path.join(
        CONFIG["output_dir"],
        "calibration_results.json"
    )

    results = {

        "ece": ece,

        "num_bins": CONFIG["num_bins"],

        "bin_accuracies": bin_accuracies,

        "bin_confidences": bin_confidences,

        "bin_counts": bin_counts,

        "num_samples": int(
            sum(bin_counts)
        ),

        "per_class_ece": per_class_results,

        "ground_truth_definition":
            "Majority class obtained using argmax of soft-label distribution",

        "class_names":
            CLASS_NAMES
    }

    with open(
        output_path,
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print(
        f"\nResults saved to:\n"
        f"{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("RIVA SOFT-LABEL CALIBRATION ANALYSIS")
    print("=" * 70)

    device = get_device()

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        CONFIG["output_dir"],
        exist_ok=True
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    test_dataset = load_test_dataset()

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model(
        device
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    (
        probabilities,
        predictions,
        targets
    ) = collect_predictions(
        model,
        test_loader,
        device
    )

    # --------------------------------------------------------
    # ECE
    # --------------------------------------------------------

    (
        ece,
        bin_accuracies,
        bin_confidences,
        bin_counts
    ) = calculate_ece(
        probabilities,
        predictions,
        targets,
        CONFIG["num_bins"]
    )

    # --------------------------------------------------------
    # Per-class ECE
    # --------------------------------------------------------

    per_class_results = calculate_per_class_ece(
        probabilities,
        predictions,
        targets,
        CLASS_NAMES,
        num_bins=CONFIG["num_bins"],
        min_samples=CONFIG["min_class_samples_for_ece"]
    )

    # --------------------------------------------------------
    # Print ECE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CALIBRATION RESULT")
    print("=" * 70)

    print(
        f"\nExpected Calibration Error (ECE): "
        f"{ece:.4f}"
    )

    print(
        "\nInterpretation:"
    )

    if ece < 0.05:

        print(
            "Excellent calibration"
        )

    elif ece < 0.10:

        print(
            "Good calibration"
        )

    elif ece < 0.20:

        print(
            "Moderate calibration"
        )

    else:

        print(
            "Poor calibration"
        )

    # --------------------------------------------------------
    # Reliability diagram
    # --------------------------------------------------------

    reliability_path = os.path.join(
        CONFIG["output_dir"],
        "reliability_diagram.png"
    )

    create_reliability_diagram(
        bin_accuracies,
        bin_confidences,
        ece,
        reliability_path
    )

    # --------------------------------------------------------
    # Per-class ECE chart
    # --------------------------------------------------------

    per_class_chart_path = os.path.join(
        CONFIG["output_dir"],
        "per_class_ece.png"
    )

    create_per_class_ece_chart(
        per_class_results,
        per_class_chart_path
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    save_results(
        ece,
        bin_accuracies,
        bin_confidences,
        bin_counts,
        per_class_results
    )

    print("\n" + "=" * 70)
    print("CALIBRATION ANALYSIS COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()