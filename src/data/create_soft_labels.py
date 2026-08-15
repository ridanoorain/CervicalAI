import json
from pathlib import Path
from collections import Counter


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "matched_annotations.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "soft_labels.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

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
# LOAD DATA
# ============================================================

print("=" * 70)
print("RIVA SOFT LABEL GENERATION")
print("=" * 70)

print("\nLoading matched annotations from:")
print(INPUT_PATH)

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    matched_data = json.load(f)

print("Successfully loaded matched annotations.")

print(f"Number of image/cell IDs: {len(matched_data)}")


# ============================================================
# SOFT LABEL GENERATION
# ============================================================

soft_labels_data = {}

total_cells = 0
cells_with_soft_labels = 0
cells_with_one_label = 0
cells_with_disagreement = 0
total_expert_labels = 0


for image_id, image_data in matched_data.items():

    # --------------------------------------------------------
    # Get matched cells
    # --------------------------------------------------------

    matched_cells = image_data.get("matched_cells", [])

    if not matched_cells:
        continue

    soft_labels_data[image_id] = {
        "num_annotators": image_data.get("num_annotators", 0),
        "reference_annotator": image_data.get(
            "reference_annotator"
        ),
        "cells": []
    }

    # --------------------------------------------------------
    # Process every matched cell
    # --------------------------------------------------------

    for cell in matched_cells:

        annotations = cell.get("annotations", {})

        if not annotations:
            continue

        # ----------------------------------------------------
        # Collect labels from all annotators
        # ----------------------------------------------------

        labels = []

        for annotator, annotation in annotations.items():

            label = annotation.get("label")

            if label is not None:
                labels.append(label)

        if not labels:
            continue

        total_cells += 1
        total_expert_labels += len(labels)

        # ----------------------------------------------------
        # Count labels
        # ----------------------------------------------------

        label_counts = Counter(labels)

        unique_labels = len(label_counts)

        # ----------------------------------------------------
        # Create probability distribution
        # ----------------------------------------------------

        soft_distribution = {}

        for label in LABELS:

            count = label_counts.get(label, 0)

            soft_distribution[label] = round(
                count / len(labels),
                4
            )

        # ----------------------------------------------------
        # Determine disagreement
        # ----------------------------------------------------

        if unique_labels == 1:

            cells_with_one_label += 1

        else:

            cells_with_disagreement += 1
            cells_with_soft_labels += 1

        # ----------------------------------------------------
        # Hard label = majority label
        # ----------------------------------------------------

        majority_label = label_counts.most_common(1)[0][0]

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        soft_cell = {
            "reference_index": cell.get("reference_index"),
            "x": cell.get("x"),
            "y": cell.get("y"),

            "num_expert_labels": len(labels),

            "expert_labels": labels,

            "label_counts": dict(label_counts),

            "soft_label": soft_distribution,

            "majority_label": majority_label,

            "disagreement": unique_labels > 1
        }

        soft_labels_data[image_id]["cells"].append(
            soft_cell
        )


# ============================================================
# REMOVE EMPTY IMAGE ENTRIES
# ============================================================

soft_labels_data = {
    image_id: data
    for image_id, data in soft_labels_data.items()
    if data["cells"]
}


# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        soft_labels_data,
        f,
        indent=2
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SOFT LABEL GENERATION SUMMARY")
print("=" * 70)

print(
    f"\nCells processed             : {total_cells}"
)

print(
    f"Cells with soft labels     : {cells_with_soft_labels}"
)

print(
    f"Cells with one label       : {cells_with_one_label}"
)

print(
    f"Cells with disagreement   : {cells_with_disagreement}"
)

print(
    f"Total expert labels        : {total_expert_labels}"
)


# ============================================================
# SAMPLE OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE SOFT LABELS")
print("=" * 70)

sample_count = 0

for image_id, image_data in soft_labels_data.items():

    for cell in image_data["cells"]:

        if cell["disagreement"]:

            print(f"\nIMAGE/CELL ID: {image_id}")

            print(
                f"Reference index: "
                f"{cell['reference_index']}"
            )

            print(
                f"Expert labels: "
                f"{cell['expert_labels']}"
            )

            print(
                f"Label counts: "
                f"{cell['label_counts']}"
            )

            print(
                f"Soft label distribution:"
            )

            for label, probability in cell[
                "soft_label"
            ].items():

                if probability > 0:
                    print(
                        f"  {label:6s}: "
                        f"{probability:.4f}"
                    )

            print(
                f"Majority label: "
                f"{cell['majority_label']}"
            )

            sample_count += 1

            if sample_count >= 10:
                break

    if sample_count >= 10:
        break


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("OUTPUT")
print("=" * 70)

print("\nSoft labels saved to:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("SOFT LABEL GENERATION COMPLETE")
print("=" * 70)