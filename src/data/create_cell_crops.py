import json
from pathlib import Path
from PIL import Image


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

IMAGE_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "RIVA"
    / "riva_1.0"
    / "images"
)

SOFT_LABEL_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "soft_labels.json"
)

CROP_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "cell_crops"
)

METADATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "cell_crops_metadata.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Size of the extracted crop
CROP_SIZE = 224

# RIVA coordinates are percentages.
# Example:
# x = 50 means middle of image
# y = 50 means middle of image
#
# We use a 224x224 crop around the annotated point.


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

CROP_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD SOFT LABELS
# ============================================================

print("=" * 70)
print("RIVA CELL CROP GENERATION")
print("=" * 70)

print("\nLoading soft labels from:")
print(SOFT_LABEL_PATH)

with open(
    SOFT_LABEL_PATH,
    "r",
    encoding="utf-8"
) as f:

    soft_labels_data = json.load(f)

print("Successfully loaded soft labels.")

print(
    f"Number of image IDs: "
    f"{len(soft_labels_data)}"
)


# ============================================================
# DATASET METADATA
# ============================================================

metadata = []

total_cells = 0
successful_crops = 0
missing_images = 0
failed_crops = 0


# ============================================================
# PROCESS EACH IMAGE
# ============================================================

for image_id, image_data in soft_labels_data.items():

    image_filename = f"{image_id}.png"

    image_path = IMAGE_DIR / image_filename

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if not image_path.exists():

        print(
            f"\nWARNING: Image not found: "
            f"{image_filename}"
        )

        missing_images += 1

        continue

    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    try:

        image = Image.open(image_path).convert("RGB")

    except Exception as e:

        print(
            f"\nERROR opening {image_filename}: {e}"
        )

        failed_crops += 1

        continue

    width, height = image.size

    # --------------------------------------------------------
    # Create folder for this image
    # --------------------------------------------------------

    image_crop_dir = CROP_DIR / image_id

    image_crop_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Process cells
    # --------------------------------------------------------

    cells = image_data.get("cells", [])

    for cell_number, cell in enumerate(cells):

        total_cells += 1

        # ----------------------------------------------------
        # Get normalized coordinates
        # ----------------------------------------------------

        x_percent = cell.get("x")
        y_percent = cell.get("y")

        if x_percent is None or y_percent is None:

            failed_crops += 1

            continue

        # ----------------------------------------------------
        # Convert percentage -> pixel coordinates
        # ----------------------------------------------------

        pixel_x = (
            x_percent / 100.0
        ) * width

        pixel_y = (
            y_percent / 100.0
        ) * height

        pixel_x = int(round(pixel_x))
        pixel_y = int(round(pixel_y))

        # ----------------------------------------------------
        # Calculate crop boundaries
        # ----------------------------------------------------

        half = CROP_SIZE // 2

        left = pixel_x - half
        top = pixel_y - half
        right = pixel_x + half
        bottom = pixel_y + half

        # ----------------------------------------------------
        # Handle image boundaries
        # ----------------------------------------------------

        if left < 0:

            right += -left
            left = 0

        if top < 0:

            bottom += -top
            top = 0

        if right > width:

            left -= right - width
            right = width

        if bottom > height:

            top -= bottom - height
            bottom = height

        # ----------------------------------------------------
        # Final boundary protection
        # ----------------------------------------------------

        left = max(0, left)
        top = max(0, top)

        right = min(width, right)
        bottom = min(height, bottom)

        # ----------------------------------------------------
        # Crop
        # ----------------------------------------------------

        try:

            crop = image.crop(
                (
                    left,
                    top,
                    right,
                    bottom
                )
            )

            # Resize if boundary handling produced
            # a crop smaller than 224x224

            if crop.size != (
                CROP_SIZE,
                CROP_SIZE
            ):

                crop = crop.resize(
                    (
                        CROP_SIZE,
                        CROP_SIZE
                    ),
                    Image.Resampling.LANCZOS
                )

            # ------------------------------------------------
            # Save crop
            # ------------------------------------------------

            crop_filename = (
                f"cell_{cell_number:04d}.png"
            )

            crop_path = (
                image_crop_dir
                / crop_filename
            )

            crop.save(crop_path)

            # ------------------------------------------------
            # Store metadata
            # ------------------------------------------------

            metadata.append(
                {
                    "image_id": image_id,

                    "crop_filename": crop_filename,

                    "crop_path": str(
                        crop_path.relative_to(BASE_DIR)
                    ),

                    "reference_index":
                        cell.get(
                            "reference_index"
                        ),

                    "x_percent": x_percent,

                    "y_percent": y_percent,

                    "pixel_x": pixel_x,

                    "pixel_y": pixel_y,

                    "crop_size": CROP_SIZE,

                    "num_expert_labels":
                        cell.get(
                            "num_expert_labels"
                        ),

                    "expert_labels":
                        cell.get(
                            "expert_labels",
                            []
                        ),

                    "label_counts":
                        cell.get(
                            "label_counts",
                            {}
                        ),

                    "soft_label":
                        cell.get(
                            "soft_label",
                            {}
                        ),

                    "majority_label":
                        cell.get(
                            "majority_label"
                        ),

                    "disagreement":
                        cell.get(
                            "disagreement",
                            False
                        )
                }
            )

            successful_crops += 1

        except Exception as e:

            print(
                f"\nERROR creating crop "
                f"for {image_id}, "
                f"cell {cell_number}: {e}"
            )

            failed_crops += 1


# ============================================================
# SAVE METADATA
# ============================================================

with open(
    METADATA_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CELL CROP GENERATION SUMMARY")
print("=" * 70)

print(
    f"\nTotal cells found       : "
    f"{total_cells}"
)

print(
    f"Successful crops       : "
    f"{successful_crops}"
)

print(
    f"Missing images         : "
    f"{missing_images}"
)

print(
    f"Failed crops           : "
    f"{failed_crops}"
)


# ============================================================
# SAMPLE METADATA
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE CROPS")
print("=" * 70)

for item in metadata[:10]:

    print(
        f"\n{item['image_id']} "
        f"-> {item['crop_filename']}"
    )

    print(
        f"  Coordinates (%): "
        f"({item['x_percent']:.2f}, "
        f"{item['y_percent']:.2f})"
    )

    print(
        f"  Coordinates (px): "
        f"({item['pixel_x']}, "
        f"{item['pixel_y']})"
    )

    print(
        f"  Expert labels: "
        f"{item['expert_labels']}"
    )

    print(
        f"  Soft label: "
        f"{item['soft_label']}"
    )


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("OUTPUT")
print("=" * 70)

print(
    "\nCell crops saved to:"
)

print(CROP_DIR)

print(
    "\nMetadata saved to:"
)

print(METADATA_PATH)

print("\n" + "=" * 70)
print("CELL CROP GENERATION COMPLETE")
print("=" * 70)