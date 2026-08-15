import json
import math
from pathlib import Path
from collections import Counter


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANNOTATIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "RIVA"
    / "riva_1.0"
    / "annotations"
    / "annotations.json"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = PROCESSED_DIR / "matched_annotations.json"


# ============================================================
# SETTINGS
# ============================================================

# We will test several thresholds first.
# Distance is measured in percentage coordinates.
#
# Example:
# distance = 1.0 means approximately
# 1% of image width/height.

TEST_THRESHOLDS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

# This is NOT finalized scientifically yet.
# It is only used to create the diagnostic output.
SAVE_THRESHOLD = 1.0


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("RIVA ANNOTATION SPATIAL MATCHING")
print("=" * 70)

print("\nLoading annotations from:")
print(ANNOTATIONS_FILE)

with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("\nSuccessfully loaded annotations.")
print(f"Number of image/cell IDs: {len(data)}")


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_distance(point1, point2):
    """
    Calculate Euclidean distance between two annotation points.

    Coordinates are in percentage units (0-100).
    """

    dx = point1["x"] - point2["x"]
    dy = point1["y"] - point2["y"]

    return math.sqrt(dx * dx + dy * dy)


def get_annotators(image_data):
    """
    Return annotator names from an image record.
    """

    return sorted(
        [
            key
            for key in image_data.keys()
            if key.startswith("annotator_")
        ]
    )


def get_valid_annotations(annotations):
    """
    Keep only valid annotation dictionaries.
    """

    result = []

    for annotation in annotations:

        if not isinstance(annotation, dict):
            continue

        if "x" not in annotation or "y" not in annotation:
            continue

        if "keypointlabels" not in annotation:
            continue

        result.append(annotation)

    return result


# ============================================================
# MATCH TWO ANNOTATORS
# ============================================================

def match_two_annotators(
    annotations_a,
    annotations_b,
    threshold
):
    """
    Match annotations from two annotators based on spatial distance.

    Greedy one-to-one matching:

    - Each annotation from A can match at most one annotation from B.
    - Each annotation from B can match at most one annotation from A.
    - Among possible matches, closest points are selected first.

    Returns:
        list of matched pairs
    """

    possible_matches = []

    for index_a, annotation_a in enumerate(annotations_a):

        for index_b, annotation_b in enumerate(annotations_b):

            distance = get_distance(
                annotation_a,
                annotation_b
            )

            if distance <= threshold:

                possible_matches.append(
                    (
                        distance,
                        index_a,
                        index_b
                    )
                )

    # Closest pairs first
    possible_matches.sort(
        key=lambda x: x[0]
    )

    used_a = set()
    used_b = set()

    matches = []

    for distance, index_a, index_b in possible_matches:

        if index_a in used_a:
            continue

        if index_b in used_b:
            continue

        used_a.add(index_a)
        used_b.add(index_b)

        matches.append(
            {
                "index_a": index_a,
                "index_b": index_b,
                "distance": distance
            }
        )

    return matches


# ============================================================
# MATCH ALL ANNOTATORS FOR ONE IMAGE
# ============================================================

def match_image(image_id, image_data, threshold):
    """
    Match annotations across all annotators for one image.

    The first annotator is used as the reference.
    Other annotators are matched to it.

    NOTE:
    This is the initial version for dataset diagnostics.
    We will improve the matching strategy after examining
    the resulting statistics.
    """

    annotators = get_annotators(image_data)

    if len(annotators) < 2:

        return {
            "image_id": image_id,
            "num_annotators": len(annotators),
            "matches": []
        }

    reference_annotator = annotators[0]

    reference_annotations = get_valid_annotations(
        image_data[reference_annotator]
    )

    # --------------------------------------------------------
    # Create one entry for every reference annotation
    # --------------------------------------------------------

    matched_cells = []

    for index, annotation in enumerate(reference_annotations):

        cell = {
            "reference_annotator": reference_annotator,
            "reference_index": index,
            "x": annotation["x"],
            "y": annotation["y"],
            "annotations": {
                reference_annotator: {
                    "x": annotation["x"],
                    "y": annotation["y"],
                    "label": annotation["keypointlabels"]
                }
            },
            "distances": {}
        }

        matched_cells.append(cell)

    # --------------------------------------------------------
    # Match every other annotator
    # --------------------------------------------------------

    for annotator in annotators:

        if annotator == reference_annotator:
            continue

        current_annotations = get_valid_annotations(
            image_data[annotator]
        )

        matches = match_two_annotators(
            reference_annotations,
            current_annotations,
            threshold
        )

        for match in matches:

            reference_index = match["index_a"]
            current_index = match["index_b"]
            distance = match["distance"]

            cell = matched_cells[reference_index]

            annotation = current_annotations[current_index]

            cell["annotations"][annotator] = {
                "x": annotation["x"],
                "y": annotation["y"],
                "label": annotation["keypointlabels"]
            }

            cell["distances"][annotator] = distance

    # --------------------------------------------------------
    # Keep only cells that have at least 2 annotators
    # --------------------------------------------------------

    matched_cells = [
        cell
        for cell in matched_cells
        if len(cell["annotations"]) >= 2
    ]

    return {
        "image_id": image_id,
        "num_annotators": len(annotators),
        "reference_annotator": reference_annotator,
        "matched_cells": matched_cells
    }


# ============================================================
# DIAGNOSTIC ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD DIAGNOSTICS")
print("=" * 70)

print(
    "\nTesting spatial matching thresholds:"
)

for threshold in TEST_THRESHOLDS:

    total_matched_cells = 0
    total_pairs = 0
    images_with_matches = 0

    for image_id, image_data in data.items():

        if not isinstance(image_data, dict):
            continue

        annotators = get_annotators(image_data)

        if len(annotators) < 2:
            continue

        result = match_image(
            image_id,
            image_data,
            threshold
        )

        matched_cells = result["matched_cells"]

        if matched_cells:
            images_with_matches += 1

        total_matched_cells += len(matched_cells)

        for cell in matched_cells:
            total_pairs += len(cell["annotations"]) - 1

    print(
        f"\nThreshold: {threshold:.2f}%"
    )

    print(
        f"  Images with matches : {images_with_matches}"
    )

    print(
        f"  Matched cells       : {total_matched_cells}"
    )

    print(
        f"  Matched annotator pairs: {total_pairs}"
    )


# ============================================================
# CREATE FINAL MATCHED DATASET
# ============================================================

print("\n" + "=" * 70)
print(
    f"CREATING MATCHED DATASET "
    f"(threshold = {SAVE_THRESHOLD}%)"
)
print("=" * 70)

matched_dataset = {}

total_matched_cells = 0
total_pair_matches = 0

label_disagreement_counter = Counter()

distance_values = []

for image_id, image_data in data.items():

    if not isinstance(image_data, dict):
        continue

    annotators = get_annotators(image_data)

    # We only need multi-annotator images here.
    if len(annotators) < 2:
        continue

    result = match_image(
        image_id,
        image_data,
        SAVE_THRESHOLD
    )

    matched_cells = result["matched_cells"]

    if not matched_cells:
        continue

    matched_dataset[image_id] = {
        "num_annotators": result["num_annotators"],
        "reference_annotator": result["reference_annotator"],
        "matched_cells": matched_cells
    }

    total_matched_cells += len(matched_cells)

    for cell in matched_cells:

        annotations = cell["annotations"]

        total_pair_matches += len(annotations) - 1

        # Collect distances
        for distance in cell["distances"].values():
            distance_values.append(distance)

        # ----------------------------------------------------
        # Check whether experts disagree
        # ----------------------------------------------------

        labels = [
            annotation["label"]
            for annotation in annotations.values()
        ]

        unique_labels = set(labels)

        if len(unique_labels) > 1:

            sorted_labels = tuple(
                sorted(unique_labels)
            )

            label_disagreement_counter[
                sorted_labels
            ] += 1


# ============================================================
# SAVE OUTPUT
# ============================================================

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        matched_dataset,
        f,
        indent=2
    )


# ============================================================
# FINAL STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("MATCHING SUMMARY")
print("=" * 70)

print(
    f"\nImages with matched annotations: "
    f"{len(matched_dataset)}"
)

print(
    f"Total matched cells: "
    f"{total_matched_cells}"
)

print(
    f"Total annotator-to-reference matches: "
    f"{total_pair_matches}"
)


if distance_values:

    average_distance = (
        sum(distance_values)
        / len(distance_values)
    )

    print(
        f"\nAverage matching distance: "
        f"{average_distance:.4f}%"
    )

    print(
        f"Minimum matching distance: "
        f"{min(distance_values):.4f}%"
    )

    print(
        f"Maximum matching distance: "
        f"{max(distance_values):.4f}%"
    )


# ============================================================
# DISAGREEMENT ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("ANNOTATOR DISAGREEMENT")
print("=" * 70)

print(
    f"\nCells with different labels among matched annotators: "
    f"{sum(label_disagreement_counter.values())}"
)

print("\nMost common disagreement combinations:")

for labels, count in label_disagreement_counter.most_common(20):

    print(
        f"  {' + '.join(labels):30s} "
        f"-> {count}"
    )


# ============================================================
# OUTPUT INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("OUTPUT")
print("=" * 70)

print(
    f"\nMatched annotations saved to:"
)

print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("MATCHING ANALYSIS COMPLETE")
print("=" * 70)