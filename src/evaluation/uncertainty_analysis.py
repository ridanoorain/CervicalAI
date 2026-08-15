
"""
RIVA SOFT-LABEL UNCERTAINTY ANALYSIS

Analyzes:
1. Soft-label entropy
2. Expert disagreement
3. Model prediction confidence
4. Prediction vs soft-label agreement
5. High-uncertainty samples
6. High-confidence wrong predictions
7. Relationship between uncertainty and errors

Outputs:
models/evaluation/uncertainty/
"""

import sys
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report)
from torchvision import transforms

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from dataset.riva_dataset import RIVADataset
from models.efficientnet import create_model


# ============================================================
# CONFIGURATION
# ============================================================

TEST_SPLIT = PROJECT_ROOT / "data" / "processed" / "splits" / "test.json"

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "best_model.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "evaluation"
    / "uncertainty"
)

BATCH_SIZE = 16

CLASS_NAMES = [
    "NILM",
    "INFL",
    "LSIL",
    "HSIL",
    "SCC",
    "ENDO",
    "ASCH",
    "ASCUS",
]

NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# ENTROPY
# ============================================================

def entropy(probabilities):
    """
    Shannon entropy.

    Higher entropy = greater uncertainty.
    """

    probabilities = np.asarray(probabilities, dtype=np.float64)

    probabilities = np.clip(
        probabilities,
        1e-12,
        1.0
    )

    return -np.sum(
        probabilities * np.log(probabilities)
    )


# ============================================================
# NORMALIZED ENTROPY
# ============================================================

def normalized_entropy(probabilities):
    """
    Entropy normalized to [0, 1].
    """

    return entropy(probabilities) / np.log(NUM_CLASSES)


# ============================================================
# KL DIVERGENCE
# ============================================================

def kl_divergence(target, prediction):
    """
    KL(target || prediction)
    """

    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)

    target = np.clip(target, 1e-12, 1.0)
    prediction = np.clip(prediction, 1e-12, 1.0)

    return np.sum(
        target * np.log(target / prediction)
    )


# ============================================================
# LOAD DATASET
# ============================================================
TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
def load_dataset():

    print("\n" + "=" * 70)
    print("LOADING TEST DATASET")
    print("=" * 70)

    print("\nTest split:")
    print(TEST_SPLIT)

    dataset = RIVADataset(
    split_file=TEST_SPLIT,
    transform=TEST_TRANSFORM
)

    print(
        f"\nTest samples: {len(dataset)}"
    )

    return dataset


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    print("\nCheckpoint:")
    print(CHECKPOINT)

    model = create_model(
        pretrained=False
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.to(DEVICE)
    model.eval()

    print("\nModel loaded successfully.")

    return model


# ============================================================
# ANALYSIS
# ============================================================

def run_analysis(model, dataset):

    print("\n" + "=" * 70)
    print("UNCERTAINTY ANALYSIS")
    print("=" * 70)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    results = []

    total_correct = 0
    total_samples = 0
    
    y_true = []
    y_pred = []


    sample_index = 0

    with torch.no_grad():

        for images, soft_labels, majority_labels in loader:

            images = images.to(DEVICE)

            logits = model(images)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            probabilities_np = (
                probabilities
                .cpu()
                .numpy()
            )

            soft_labels_np = (
                soft_labels
                .numpy()
            )

            predictions_np = (
                predictions
                .cpu()
                .numpy()
            )

            majority_np = (
                majority_labels
                .numpy()
            )

            for i in range(len(images)):

                prediction = int(
                    predictions_np[i]
                )

                target = int(
                    majority_np[i]
                )
                y_true.append(target)
                y_pred.append(prediction)

                probs = probabilities_np[i]

                soft_label = soft_labels_np[i]

                confidence = float(
                    probs[prediction]
                )

                soft_entropy = float(
                    entropy(soft_label)
                )

                normalized_soft_entropy = float(
                    normalized_entropy(
                        soft_label
                    )
                )

                prediction_entropy = float(
                    entropy(probs)
                )

                normalized_prediction_entropy = float(
                    normalized_entropy(
                        probs
                    )
                )

                kl = float(
                    kl_divergence(
                        soft_label,
                        probs
                    )
                )

                # Agreement between model
                # probability and expert distribution.
                soft_label_probability = float(
                    soft_label[prediction]
                )

                correct = (
                    prediction == target
                )

                if correct:
                    total_correct += 1

                total_samples += 1

                # Expert disagreement:
                # 1 - maximum soft-label probability.
                disagreement = float(
                    1.0 - np.max(soft_label)
                )

                # Number of labels represented
                # by the experts.
                number_of_labels = int(
                    np.sum(
                        soft_label > 0
                    )
                )

                results.append(
                    {
                        "index": sample_index,

                        "image_id": dataset.samples[
                            sample_index
                        ]["image_id"],

                        "crop_path": dataset.samples[
                            sample_index
                        ]["crop_path"],

                        "actual_label":
                            CLASS_NAMES[target],

                        "predicted_label":
                            CLASS_NAMES[prediction],

                        "correct":
                            bool(correct),

                        "confidence":
                            confidence,

                        "soft_label_entropy":
                            soft_entropy,

                        "normalized_soft_label_entropy":
                            normalized_soft_entropy,

                        "prediction_entropy":
                            prediction_entropy,

                        "normalized_prediction_entropy":
                            normalized_prediction_entropy,

                        "soft_label_probability":
                            soft_label_probability,

                        "kl_divergence":
                            kl,

                        "expert_disagreement":
                            disagreement,

                        "number_of_expert_labels":
                            number_of_labels,

                        "soft_label":
                            {
                                CLASS_NAMES[j]:
                                    float(soft_label[j])
                                for j in range(NUM_CLASSES)
                                if soft_label[j] > 0
                            },
                    }
                )

                sample_index += 1

    # ========================================================
    # CLASSIFICATION METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    print(
        f"\nTotal samples : {total_samples}"
    )

    print(
        f"Correct       : {total_correct}"
    )

    print(
        f"Accuracy      : {accuracy:.4f}"
    )

    print(
        f"Macro Precision : {macro_precision:.4f}"
    )

    print(
        f"Macro Recall    : {macro_recall:.4f}"
    )

    print(
        f"Macro F1        : {macro_f1:.4f}"
    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CLASSIFICATION REPORT"
    )

    print(
        "=" * 70
    )

    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES))
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "CONFUSION MATRIX"
    )

    print(
        "=" * 70
    )

    print(cm)


    return results


# ============================================================
# SUMMARY
# ============================================================

def print_summary(results):

    print("\n" + "=" * 70)
    print("UNCERTAINTY SUMMARY")
    print("=" * 70)

    entropies = np.array(
        [
            r["normalized_soft_label_entropy"]
            for r in results
        ]
    )

    confidences = np.array(
        [
            r["confidence"]
            for r in results
        ]
    )

    disagreements = np.array(
        [
            r["expert_disagreement"]
            for r in results
        ]
    )

    errors = np.array(
        [
            not r["correct"]
            for r in results
        ]
    )

    print(
        f"\nAverage soft-label entropy       : "
        f"{entropies.mean():.4f}"
    )

    print(
        f"Average prediction confidence   : "
        f"{confidences.mean():.4f}"
    )

    print(
        f"Average expert disagreement     : "
        f"{disagreements.mean():.4f}"
    )

    print(
        f"Incorrect predictions            : "
        f"{errors.sum()}"
    )

    print(
        f"Error rate                       : "
        f"{errors.mean():.4f}"
    )

    # --------------------------------------------------------
    # HIGH UNCERTAINTY
    # --------------------------------------------------------

    high_uncertainty = sorted(
        results,
        key=lambda x:
            x["normalized_soft_label_entropy"],
        reverse=True
    )[:10]

    print("\n" + "=" * 70)
    print("TOP 10 HIGH-UNCERTAINTY SAMPLES")
    print("=" * 70)

    for item in high_uncertainty:

        print(
            f"\n{item['crop_path']}"
        )

        print(
            f"Actual       : "
            f"{item['actual_label']}"
        )

        print(
            f"Predicted    : "
            f"{item['predicted_label']}"
        )

        print(
            f"Confidence   : "
            f"{item['confidence']:.4f}"
        )

        print(
            f"Soft entropy : "
            f"{item['normalized_soft_label_entropy']:.4f}"
        )

        print(
            f"Expert disagreement : "
            f"{item['expert_disagreement']:.4f}"
        )

        print(
            f"Soft label   : "
            f"{item['soft_label']}"
        )

    # --------------------------------------------------------
    # HIGH CONFIDENCE ERRORS
    # --------------------------------------------------------

    high_confidence_errors = sorted(
        [
            r for r in results
            if not r["correct"]
        ],
        key=lambda x:
            x["confidence"],
        reverse=True
    )[:10]

    print("\n" + "=" * 70)
    print("TOP 10 HIGH-CONFIDENCE WRONG PREDICTIONS")
    print("=" * 70)

    for item in high_confidence_errors:

        print(
            f"\n{item['crop_path']}"
        )

        print(
            f"Actual       : "
            f"{item['actual_label']}"
        )

        print(
            f"Predicted    : "
            f"{item['predicted_label']}"
        )

        print(
            f"Confidence   : "
            f"{item['confidence']:.4f}"
        )

        print(
            f"Soft label   : "
            f"{item['soft_label']}"
        )


# ============================================================
# PLOT 1: SOFT LABEL ENTROPY DISTRIBUTION
# ============================================================

def plot_entropy_distribution(results):

    values = [
        r["normalized_soft_label_entropy"]
        for r in results
    ]

    plt.figure(figsize=(8, 5))

    plt.hist(
        values,
        bins=20,
        edgecolor="black"
    )

    plt.xlabel(
        "Normalized Soft-Label Entropy"
    )

    plt.ylabel(
        "Number of Samples"
    )

    plt.title(
        "Distribution of Soft-Label Uncertainty"
    )

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "soft_label_entropy.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    print(
        f"\nSaved: {path}"
    )


# ============================================================
# PLOT 2: CONFIDENCE DISTRIBUTION
# ============================================================

def plot_confidence_distribution(results):

    values = [
        r["confidence"]
        for r in results
    ]

    plt.figure(figsize=(8, 5))

    plt.hist(
        values,
        bins=20,
        edgecolor="black"
    )

    plt.xlabel(
        "Prediction Confidence"
    )

    plt.ylabel(
        "Number of Samples"
    )

    plt.title(
        "Model Prediction Confidence"
    )

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "prediction_confidence.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# PLOT 3: UNCERTAINTY VS CONFIDENCE
# ============================================================

def plot_uncertainty_vs_confidence(results):

    entropy_values = [
        r["normalized_soft_label_entropy"]
        for r in results
    ]

    confidence_values = [
        r["confidence"]
        for r in results
    ]

    correct = [
        r["correct"]
        for r in results
    ]

    plt.figure(figsize=(8, 5))

    for is_correct in [True, False]:

        x = [
            entropy_values[i]
            for i in range(len(results))
            if correct[i] == is_correct
        ]

        y = [
            confidence_values[i]
            for i in range(len(results))
            if correct[i] == is_correct
        ]

        plt.scatter(
            x,
            y,
            alpha=0.6,
            label=(
                "Correct"
                if is_correct
                else "Incorrect"
            )
        )

    plt.xlabel(
        "Soft-Label Entropy"
    )

    plt.ylabel(
        "Model Confidence"
    )

    plt.title(
        "Soft-Label Uncertainty vs Model Confidence"
    )

    plt.legend()

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "uncertainty_vs_confidence.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# PLOT 4: EXPERT DISAGREEMENT VS ERROR RATE
# ============================================================

def plot_disagreement_vs_error(results):

    bins = [
        (0.0, 0.0),
        (0.0, 0.34),
        (0.34, 0.67),
        (0.67, 1.01),
    ]

    labels = [
        "No disagreement",
        "Low",
        "Medium",
        "High",
    ]

    error_rates = []

    for low, high in bins:

        group = [
            r for r in results
            if low <= r["expert_disagreement"] < high
        ]

        if len(group) == 0:

            error_rates.append(0)

        else:

            error_rates.append(
                np.mean(
                    [
                        not r["correct"]
                        for r in group
                    ]
                )
            )

    plt.figure(figsize=(8, 5))

    plt.bar(
        labels,
        error_rates
    )

    plt.xlabel(
        "Expert Disagreement"
    )

    plt.ylabel(
        "Error Rate"
    )

    plt.title(
        "Model Error Rate vs Expert Disagreement"
    )

    plt.ylim(0, 1)

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "disagreement_vs_error.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {

        "num_samples":
            len(results),

        "average_soft_label_entropy":
            float(
                np.mean(
                    [
                        r[
                            "normalized_soft_label_entropy"
                        ]
                        for r in results
                    ]
                )
            ),

        "average_prediction_confidence":
            float(
                np.mean(
                    [
                        r["confidence"]
                        for r in results
                    ]
                )
            ),

        "average_expert_disagreement":
            float(
                np.mean(
                    [
                        r[
                            "expert_disagreement"
                        ]
                        for r in results
                    ]
                )
            ),

        "samples":
            results,
    }

    path = (
        OUTPUT_DIR
        / "uncertainty_analysis.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=4
        )

    print(
        f"\nResults saved to:\n{path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "# RIVA SOFT-LABEL UNCERTAINTY ANALYSIS"
    )

    print(
        f"\nProject root: {PROJECT_ROOT}"
    )

    print(
        f"Source directory: {SRC_DIR}"
    )

    print(
        f"\nDevice: {DEVICE}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset = load_dataset()

    model = load_model()

    results = run_analysis(
        model,
        dataset
    )

    print_summary(
        results
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "GENERATING PLOTS"
    )

    print(
        "=" * 70
    )

    plot_entropy_distribution(
        results
    )

    plot_confidence_distribution(
        results
    )

    plot_uncertainty_vs_confidence(
        results
    )

    plot_disagreement_vs_error(
        results
    )

    save_results(
        results
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "UNCERTAINTY ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()

