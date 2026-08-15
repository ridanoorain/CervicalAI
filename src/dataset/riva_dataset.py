import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image


class RIVADataset(Dataset):
    """
    PyTorch Dataset for the RIVA cervical cytology dataset.

    Each sample contains:
        - cell crop image
        - soft-label distribution
        - expert labels
        - majority label
        - image ID
    """

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

    LABEL_TO_INDEX = {
        label: index
        for index, label in enumerate(LABELS)
    }

    def __init__(
        self,
        split_file,
        transform=None,
    ):
        """
        Args:
            split_file: Path to train.json / val.json / test.json
            transform: torchvision transforms
        """

        self.split_file = Path(split_file)
        self.transform = transform

        # ---------------------------------------------------------
        # Load JSON
        # ---------------------------------------------------------

        print()
        print("Loading split from:")
        print(self.split_file)

        with open(self.split_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        print("Successfully loaded split.")

        # ---------------------------------------------------------
        # Dataset information
        # ---------------------------------------------------------

        self.split = data["split"]
        self.random_seed = data["random_seed"]
        self.num_images = data["num_images"]
        self.num_cells = data["num_cells"]
        self.image_ids = data["image_ids"]

        # IMPORTANT:
        # Samples are stored inside the "samples" key
        self.samples = data["samples"]

        print(f"Split: {self.split}")
        print(f"Number of images: {self.num_images}")
        print(f"Number of samples: {len(self.samples)}")

        # ---------------------------------------------------------
        # Validate samples
        # ---------------------------------------------------------

        self._validate_samples()

    def _validate_samples(self):
        """
        Check that all crop files referenced by the split exist.
        """

        missing = []

        for sample in self.samples:

            crop_path = Path(sample["crop_path"])

            if not crop_path.exists():
                missing.append(crop_path)

        print()

        if missing:

            print(
                f"WARNING: {len(missing)} crop files are missing."
            )

            for path in missing[:10]:
                print(f"Missing: {path}")

        else:

            print("All crop files exist.")

    def __len__(self):
        """
        Return number of cell samples.
        """

        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        # ---------------------------------------------------------
        # IMAGE
        # ---------------------------------------------------------

        crop_path = Path(sample["crop_path"])

        image = Image.open(crop_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # ---------------------------------------------------------
        # SOFT LABEL
        # ---------------------------------------------------------

        soft_label_dict = sample["soft_label"]

        soft_label = torch.tensor(
            [
                soft_label_dict.get(label, 0.0)
                for label in self.LABELS
            ],
            dtype=torch.float32,
        )

        # ---------------------------------------------------------
        # EXPERT LABELS
        # ---------------------------------------------------------

        expert_labels = sample.get(
            "expert_labels",
            []
        )

        # ---------------------------------------------------------
        # MAJORITY LABEL
        # ---------------------------------------------------------

        majority_label = sample.get(
            "majority_label"
        )

        if majority_label is not None:

            majority_index = self.LABEL_TO_INDEX[
                majority_label
            ]

        else:

            # Calculate majority label if not stored
            if len(expert_labels) > 0:

                counts = {}

                for label in expert_labels:
                    counts[label] = counts.get(label, 0) + 1

                majority_label = max(
                    counts,
                    key=counts.get
                )

                majority_index = self.LABEL_TO_INDEX[
                    majority_label
                ]

            else:

                majority_index = -1

        # ---------------------------------------------------------
        # RETURN SAMPLE
        # ---------------------------------------------------------

        return (
    image,
    soft_label,
    torch.tensor(
        majority_index,
        dtype=torch.long
    )
)

def create_datasets(
    processed_dir="data/processed",
    transform=None,
):
    """
    Create train, validation and test RIVA datasets.
    """

    processed_dir = Path(processed_dir)

    split_dir = processed_dir / "splits"

    # -------------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------------

    train_dataset = RIVADataset(
        split_file=split_dir / "train.json",
        transform=transform,
    )

    # -------------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------------

    val_dataset = RIVADataset(
        split_file=split_dir / "val.json",
        transform=transform,
    )

    # -------------------------------------------------------------
    # TEST
    # -------------------------------------------------------------

    test_dataset = RIVADataset(
        split_file=split_dir / "test.json",
        transform=transform,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )


# =================================================================
# TEST
# =================================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("RIVA DATASET LOADER TEST")
    print("=" * 70)

    # -------------------------------------------------------------
    # Create datasets
    # -------------------------------------------------------------

    train_dataset, val_dataset, test_dataset = create_datasets()

    # -------------------------------------------------------------
    # Dataset sizes
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("DATASET SIZES")
    print("=" * 70)

    print(
        f"Train      : {len(train_dataset)} cells"
    )

    print(
        f"Validation : {len(val_dataset)} cells"
    )

    print(
        f"Test       : {len(test_dataset)} cells"
    )

    # -------------------------------------------------------------
    # Inspect first training sample
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("FIRST TRAINING SAMPLE")
    print("=" * 70)

    sample = train_dataset[0]

    print()
    print("Image:")
    print(
        f"Type: {type(sample['image'])}"
    )

    print(
        f"Size: {sample['image'].size}"
    )

    print()
    print("Image ID:")
    print(sample["image_id"])

    print()
    print("Crop path:")
    print(sample["crop_path"])

    print()
    print("Expert labels:")
    print(sample["expert_labels"])

    print()
    print("Soft label:")
    print(sample["soft_label"])

    print()
    print("Soft label shape:")
    print(sample["soft_label"].shape)

    print()
    print("Soft label sum:")
    print(
        sample["soft_label"].sum().item()
    )

    print()
    print("Majority label index:")
    print(
        sample["majority_label"].item()
    )

    # -------------------------------------------------------------
    # Verify soft label
    # -------------------------------------------------------------

    assert sample["soft_label"].shape == (8,)

    assert abs(
        sample["soft_label"].sum().item() - 1.0
    ) < 1e-5

    print()
    print("Soft label validation: PASSED")

    print()
    print("=" * 70)
    print("DATASET TEST COMPLETE")
    print("=" * 70)