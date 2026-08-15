import json
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


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("RIVA DATASET ANALYSIS")
print("=" * 70)

print(f"\nLoading annotations from:")
print(ANNOTATIONS_FILE)

with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("\nSuccessfully loaded annotations.")


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("1. BASIC DATASET INFORMATION")
print("=" * 70)

print(f"Number of image/cell IDs: {len(data)}")


# ============================================================
# ANNOTATOR ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("2. ANNOTATOR INFORMATION")
print("=" * 70)

annotator_count_distribution = Counter()
annotator_presence = Counter()

for image_id, image_data in data.items():

    if not isinstance(image_data, dict):
        continue

    annotators = [
        key
        for key in image_data.keys()
        if key.startswith("annotator_")
    ]

    annotator_count_distribution[len(annotators)] += 1

    for annotator in annotators:
        annotator_presence[annotator] += 1


print("\nImages according to number of annotators:")

for count in sorted(annotator_count_distribution):
    print(
        f"  {count} annotator(s): "
        f"{annotator_count_distribution[count]} images"
    )


print("\nAnnotator presence:")

for annotator in sorted(annotator_presence):
    print(
        f"  {annotator}: "
        f"{annotator_presence[annotator]} images"
    )


# ============================================================
# LABEL ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("3. LABEL DISTRIBUTION")
print("=" * 70)

global_label_counts = Counter()
label_counts_by_annotator = {}

total_annotations = 0

for image_id, image_data in data.items():

    if not isinstance(image_data, dict):
        continue

    for annotator, annotations in image_data.items():

        if not annotator.startswith("annotator_"):
            continue

        if annotator not in label_counts_by_annotator:
            label_counts_by_annotator[annotator] = Counter()

        if not isinstance(annotations, list):
            continue

        for annotation in annotations:

            if not isinstance(annotation, dict):
                continue

            label = annotation.get("keypointlabels")

            if label is None:
                continue

            global_label_counts[label] += 1
            label_counts_by_annotator[annotator][label] += 1
            total_annotations += 1


print(f"\nTotal individual annotations: {total_annotations}")

print("\nOverall label distribution:")

for label, count in global_label_counts.most_common():
    percentage = (count / total_annotations) * 100

    print(
        f"  {label:8s}: "
        f"{count:6d} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# LABEL DISTRIBUTION BY ANNOTATOR
# ============================================================

print("\n" + "=" * 70)
print("4. LABEL DISTRIBUTION BY ANNOTATOR")
print("=" * 70)

for annotator in sorted(label_counts_by_annotator):

    counts = label_counts_by_annotator[annotator]
    total = sum(counts.values())

    print(f"\n{annotator}")
    print("-" * 40)

    print(f"Total annotations: {total}")

    for label, count in counts.most_common():

        percentage = (count / total) * 100

        print(
            f"  {label:8s}: "
            f"{count:6d} "
            f"({percentage:6.2f}%)"
        )


# ============================================================
# ANNOTATIONS PER IMAGE
# ============================================================

print("\n" + "=" * 70)
print("5. ANNOTATIONS PER IMAGE")
print("=" * 70)

annotations_per_image = []

for image_id, image_data in data.items():

    total_for_image = 0

    if isinstance(image_data, dict):

        for annotator, annotations in image_data.items():

            if not annotator.startswith("annotator_"):
                continue

            if isinstance(annotations, list):
                total_for_image += len(annotations)

    annotations_per_image.append(total_for_image)


if annotations_per_image:

    print(
        f"Minimum annotations in an image: "
        f"{min(annotations_per_image)}"
    )

    print(
        f"Maximum annotations in an image: "
        f"{max(annotations_per_image)}"
    )

    average = sum(annotations_per_image) / len(annotations_per_image)

    print(
        f"Average annotations per image: "
        f"{average:.2f}"
    )


# ============================================================
# ANNOTATIONS PER ANNOTATOR
# ============================================================

print("\n" + "=" * 70)
print("6. ANNOTATIONS PER ANNOTATOR")
print("=" * 70)

annotations_per_annotator = Counter()

for image_id, image_data in data.items():

    if not isinstance(image_data, dict):
        continue

    for annotator, annotations in image_data.items():

        if not annotator.startswith("annotator_"):
            continue

        if isinstance(annotations, list):
            annotations_per_annotator[annotator] += len(annotations)


for annotator in sorted(annotations_per_annotator):

    print(
        f"  {annotator}: "
        f"{annotations_per_annotator[annotator]} annotations"
    )


# ============================================================
# MISSING LABEL / INVALID ANNOTATION CHECK
# ============================================================

print("\n" + "=" * 70)
print("7. DATA QUALITY CHECK")
print("=" * 70)

missing_label_count = 0
invalid_annotation_count = 0
invalid_coordinate_count = 0

valid_labels = set(global_label_counts.keys())

for image_id, image_data in data.items():

    if not isinstance(image_data, dict):
        continue

    for annotator, annotations in image_data.items():

        if not annotator.startswith("annotator_"):
            continue

        if not isinstance(annotations, list):
            continue

        for annotation in annotations:

            if not isinstance(annotation, dict):
                invalid_annotation_count += 1
                continue

            # Check label
            label = annotation.get("keypointlabels")

            if label is None:
                missing_label_count += 1

            # Check coordinates
            x = annotation.get("x")
            y = annotation.get("y")

            if x is None or y is None:
                invalid_coordinate_count += 1
                continue

            if not (0 <= x <= 100 and 0 <= y <= 100):
                invalid_coordinate_count += 1


print(f"Annotations missing labels: {missing_label_count}")
print(f"Invalid annotation objects: {invalid_annotation_count}")
print(f"Invalid coordinates: {invalid_coordinate_count}")


# ============================================================
# IMAGES WITH MULTIPLE ANNOTATORS
# ============================================================

print("\n" + "=" * 70)
print("8. MULTI-ANNOTATOR IMAGES")
print("=" * 70)

multi_annotator_images = []

for image_id, image_data in data.items():

    if not isinstance(image_data, dict):
        continue

    annotators = [
        key
        for key in image_data.keys()
        if key.startswith("annotator_")
    ]

    if len(annotators) >= 2:
        multi_annotator_images.append(
            (image_id, annotators)
        )


print(
    f"Images with 2 or more annotators: "
    f"{len(multi_annotator_images)}"
)

print("\nFirst 20 examples:")

for image_id, annotators in multi_annotator_images[:20]:

    print(
        f"  {image_id}: "
        f"{', '.join(sorted(annotators))}"
    )


# ============================================================
# LABELS PER IMAGE
# ============================================================

print("\n" + "=" * 70)
print("9. IMAGES CONTAINING DIFFERENT LABELS")
print("=" * 70)

mixed_label_images = []
single_label_images = []

for image_id, image_data in data.items():

    labels = set()

    if not isinstance(image_data, dict):
        continue

    for annotator, annotations in image_data.items():

        if not annotator.startswith("annotator_"):
            continue

        if not isinstance(annotations, list):
            continue

        for annotation in annotations:

            label = annotation.get("keypointlabels")

            if label is not None:
                labels.add(label)

    if len(labels) > 1:
        mixed_label_images.append(
            (image_id, labels)
        )

    elif len(labels) == 1:
        single_label_images.append(
            (image_id, labels)
        )


print(
    f"Images containing multiple label types: "
    f"{len(mixed_label_images)}"
)

print(
    f"Images containing only one label type: "
    f"{len(single_label_images)}"
)

print("\nExamples of mixed-label images:")

for image_id, labels in mixed_label_images[:20]:

    print(
        f"  {image_id}: "
        f"{', '.join(sorted(labels))}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("10. FINAL SUMMARY")
print("=" * 70)

print(f"""
Images / annotation records : {len(data)}
Total cell annotations       : {total_annotations}

Annotator distribution:
""")

for count in sorted(annotator_count_distribution):

    print(
        f"  {count} annotator(s) -> "
        f"{annotator_count_distribution[count]} images"
    )

print("\nLabels:")

for label, count in global_label_counts.most_common():

    percentage = (count / total_annotations) * 100

    print(
        f"  {label:8s} -> "
        f"{count:6d} ({percentage:.2f}%)"
    )

print("\nData quality:")

print(
    f"  Missing labels       : {missing_label_count}"
)

print(
    f"  Invalid annotations  : {invalid_annotation_count}"
)

print(
    f"  Invalid coordinates  : {invalid_coordinate_count}"
)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)