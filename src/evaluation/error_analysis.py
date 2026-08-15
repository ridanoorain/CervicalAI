import sys
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

print("Project root:", PROJECT_ROOT)
print("Source directory:", SRC_DIR)

# ============================================================
# IMPORTS
# ============================================================

import json

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.riva_dataset import RIVADataset
from models.efficientnet import create_model

# ============================================================
# CONFIGURATION
# ============================================================

TEST_SPLIT = Path("data/processed/splits/test.json")
CHECKPOINT = Path("models/checkpoints/best_model.pth")

OUTPUT_DIR = Path("models/evaluation/error_analysis")
OUTPUT_FILE = OUTPUT_DIR / "error_analysis.json"

BATCH_SIZE = 16

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


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# TRANSFORM
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    print(f"\nCheckpoint:\n{CHECKPOINT}")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    model = create_model(
        num_classes=len(LABELS)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")

    return model


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    print("\n" + "=" * 70)
    print("LOADING TEST DATASET")
    print("=" * 70)

    dataset = RIVADataset(
        split_file=TEST_SPLIT,
        transform=test_transform
    )

    print(f"\nTest samples: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return dataset, loader


# ============================================================
# ERROR ANALYSIS
# ============================================================

def analyze(model, dataset, loader):

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    errors = []

    correct = 0
    total = 0

    confusion = [
        [0 for _ in LABELS]
        for _ in LABELS
    ]

    with torch.no_grad():

        sample_index = 0

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

            batch_size = images.size(0)

            for i in range(batch_size):

                prediction_index = (
                    predictions[i].item()
                )

                target_index = (
                    majority_labels[i].item()
                )

                confidence = (
                    probabilities[i]
                    .max()
                    .item()
                )

                target_soft_label = (
                    soft_labels[i]
                    .tolist()
                )

                prediction_distribution = (
                    probabilities[i]
                    .tolist()
                )

                sample = dataset.samples[
                    sample_index
                ]

                image_id = sample["image_id"]
                crop_path = sample["crop_path"]

                confusion[target_index][
                    prediction_index
                ] += 1

                if prediction_index == target_index:

                    correct += 1

                else:

                    error = {
                        "index": sample_index,

                        "image_id": image_id,

                        "crop_path": crop_path,

                        "actual": LABELS[
                            target_index
                        ],

                        "predicted": LABELS[
                            prediction_index
                        ],

                        "confidence": round(
                            confidence,
                            6
                        ),

                        "soft_label": {
                            LABELS[j]: round(
                                target_soft_label[j],
                                6
                            )
                            for j in range(
                                len(LABELS)
                            )
                            if target_soft_label[j] > 0
                        },

                        "prediction_probabilities": {
                            LABELS[j]: round(
                                prediction_distribution[j],
                                6
                            )
                            for j in range(
                                len(LABELS)
                            )
                        },

                        "expert_labels": sample.get(
                            "expert_labels",
                            []
                        ),
                    }

                    errors.append(error)

                total += 1
                sample_index += 1

    accuracy = (
        correct / total
        if total > 0
        else 0
    )

    print(f"\nTotal samples : {total}")
    print(f"Correct       : {correct}")
    print(f"Incorrect     : {len(errors)}")
    print(
        f"Accuracy      : {accuracy:.4f}"
    )

    return errors, confusion


# ============================================================
# CONFUSION PAIRS
# ============================================================

def analyze_confusion_pairs(errors):

    pair_counts = {}

    for error in errors:

        actual = error["actual"]
        predicted = error["predicted"]

        pair = f"{actual} -> {predicted}"

        pair_counts[pair] = (
            pair_counts.get(pair, 0) + 1
        )

    sorted_pairs = sorted(
        pair_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print("\n" + "=" * 70)
    print("TOP CONFUSION PAIRS")
    print("=" * 70)

    for pair, count in sorted_pairs[:15]:

        print(
            f"{pair:<20} : {count}"
        )

    return sorted_pairs


# ============================================================
# CLASS-WISE ERRORS
# ============================================================

def analyze_class_errors(errors):

    class_errors = {
        label: 0
        for label in LABELS
    }

    for error in errors:

        actual = error["actual"]

        class_errors[actual] += 1

    print("\n" + "=" * 70)
    print("CLASS-WISE ERRORS")
    print("=" * 70)

    for label in LABELS:

        print(
            f"{label:<8} : "
            f"{class_errors[label]}"
        )

    return class_errors


# ============================================================
# HIGH-CONFIDENCE ERRORS
# ============================================================

def high_confidence_errors(errors):

    confident_errors = [
        error
        for error in errors
        if error["confidence"] >= 0.80
    ]

    confident_errors.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    print("\n" + "=" * 70)
    print("HIGH-CONFIDENCE WRONG PREDICTIONS")
    print("=" * 70)

    print(
        f"Errors with confidence >= 0.80: "
        f"{len(confident_errors)}"
    )

    for error in confident_errors[:10]:

        print(
            f"\n{error['image_id']} "
            f"{Path(error['crop_path']).name}"
        )

        print(
            f"Actual     : {error['actual']}"
        )

        print(
            f"Predicted  : {error['predicted']}"
        )

        print(
            f"Confidence : {error['confidence']}"
        )

        print(
            f"Soft label : {error['soft_label']}"
        )

    return confident_errors


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    errors,
    confusion_pairs,
    class_errors,
    confident_errors
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = {

        "num_errors": len(errors),

        "errors": errors,

        "top_confusion_pairs": [
            {
                "pair": pair,
                "count": count
            }
            for pair, count
            in confusion_pairs
        ],

        "class_wise_errors": class_errors,

        "high_confidence_error_count": len(
            confident_errors
        ),

        "high_confidence_errors":
            confident_errors,

        "labels": LABELS,

        "checkpoint": str(
            CHECKPOINT
        ),
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS SAVED")
    print("=" * 70)

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("RIVA ERROR ANALYSIS")
    print("=" * 70)

    print(
        f"\nDevice: {DEVICE}"
    )

    model = load_model()

    dataset, loader = load_dataset()

    errors, confusion = analyze(
        model,
        dataset,
        loader
    )

    confusion_pairs = analyze_confusion_pairs(
        errors
    )

    class_errors = analyze_class_errors(
        errors
    )

    confident_errors = high_confidence_errors(
        errors
    )

    save_results(
        errors,
        confusion_pairs,
        class_errors,
        confident_errors
    )

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()