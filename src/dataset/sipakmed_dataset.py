"""
SIPaKMeD Dataset (mapped to RIVA classes)

Purpose:
    Loads SIPaKMeD's 5 cell-type folders and maps them onto RIVA's
    8-class scheme, so the same model architecture and loss functions
    can be reused for backbone pretraining.

SIPaKMeD's 5 classes -> RIVA mapping used here (see rationale below):

    Superficial-Intermediate -> NILM   (normal squamous cells)
    Parabasal                -> NILM   (normal squamous cells)
    Koilocytotic              -> LSIL   (koilocytosis = classic LSIL/HPV feature)
    Dyskeratotic              -> HSIL   (dyskeratosis associated with higher-grade change)
    Metaplastic                -> EXCLUDED by default (see note below)

Rationale / things to state explicitly in your paper:
    - This mapping is a deliberate simplification, not a perfect
      clinical equivalence. SIPaKMeD's categories are morphological
      classes from a DIFFERENT annotation scheme (not Bethesda), so
      this mapping is an approximation used purely for backbone
      pretraining -- NOT as ground truth RIVA labels.
    - Metaplastic cells are benign but morphologically distinct from
      both superficial/parabasal cells and dysplastic cells. There is
      no clean Bethesda equivalent. Default: EXCLUDED (set
      INCLUDE_METAPLASTIC=True below to map it to NILM instead, but
      note this is the weaker of the two options and worth an
      ablation in your paper if you use it).
    - SCC, ENDO, ASCH, ASCUS, INFL have NO SIPaKMeD equivalent at all.
      This dataset will NOT help those classes directly -- expect
      most of the benefit to land on NILM/LSIL/HSIL boundary
      separation, consistent with the plan we discussed.

Because SIPaKMeD provides hard ground-truth labels (not multi-expert
soft labels like RIVA), this dataset yields ONE-HOT soft-label vectors
so it stays compatible with your existing SoftLabelCrossEntropy /
SoftLabelKLDivergence / WeightedSoftLabelKLDivergence losses without
any changes to those loss functions.
"""

import os
import glob

import torch
from torch.utils.data import Dataset
from PIL import Image


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

CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

# ----------------------------------------------------------------
# Toggle: include Metaplastic cells (mapped to NILM) or drop them.
# Default is False -- see rationale in the module docstring above.
# ----------------------------------------------------------------

INCLUDE_METAPLASTIC = False

# ----------------------------------------------------------------
# SIPaKMeD folder-name -> RIVA class mapping.
#
# Folder names in the standard Kaggle distribution typically look
# like "im_Superficial-Intermediate", "im_Parabasal", etc. We match
# on a lowercase substring so minor naming variations don't break
# the loader.
# ----------------------------------------------------------------

FOLDER_KEYWORD_TO_RIVA_CLASS = {
    "superficial": "NILM",
    "parabasal": "NILM",
    "koilocytotic": "LSIL",
    "dyskeratotic": "HSIL",
}

if INCLUDE_METAPLASTIC:
    FOLDER_KEYWORD_TO_RIVA_CLASS["metaplastic"] = "NILM"


class SIPaKMeDDataset(Dataset):
    """
    Loads SIPaKMeD images and yields
    (image_tensor, one_hot_soft_label, meta_dict) tuples, matching
    the same interface as RIVADataset.
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir:
                Path to the extracted SIPaKMeD folder, containing the
                5 class subfolders (e.g.
                C:\\CervicalAI\\data\\raw\\sipakmed).

            transform:
                torchvision transform to apply to each image. Use the
                SAME transform pipeline (including ImageNet
                normalization) as your RIVA training, so the backbone
                sees consistently normalized inputs across both
                datasets.
        """

        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # list of (filepath, riva_class_index)

        self._scan_folders()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No SIPaKMeD images found under {root_dir}. "
                f"Check that the folder structure matches the "
                f"standard Kaggle distribution (5 class subfolders, "
                f"each containing a CROPPED folder of images)."
            )

        self._print_summary()

    def _scan_folders(self):
        # Look for any subfolder whose name contains one of our
        # known keywords (case-insensitive), then search inside it
        # (including a CROPPED subfolder if present) for images.

        for entry in os.listdir(self.root_dir):
            entry_path = os.path.join(self.root_dir, entry)

            if not os.path.isdir(entry_path):
                continue

            entry_lower = entry.lower()

            matched_riva_class = None
            for keyword, riva_class in FOLDER_KEYWORD_TO_RIVA_CLASS.items():
                if keyword in entry_lower:
                    matched_riva_class = riva_class
                    break

            if matched_riva_class is None:
                # Not one of our mapped classes (e.g. metaplastic
                # when INCLUDE_METAPLASTIC=False) -- skip it.
                continue

            riva_index = CLASS_TO_INDEX[matched_riva_class]

            # Search recursively for images under this folder,
            # since SIPaKMeD nests images inside a CROPPED subfolder.
            image_paths = []
            for ext in ("*.bmp", "*.png", "*.jpg", "*.jpeg"):
                image_paths.extend(
                    glob.glob(os.path.join(entry_path, "**", ext), recursive=True)
                )

            for image_path in image_paths:
                self.samples.append((image_path, riva_index))

    def _print_summary(self):
        print(f"\nSIPaKMeD dataset loaded from: {self.root_dir}")
        print(f"Total samples: {len(self.samples)}")

        counts = {name: 0 for name in CLASS_NAMES}
        for _, riva_index in self.samples:
            counts[CLASS_NAMES[riva_index]] += 1

        print("Mapped class distribution:")
        for name, count in counts.items():
            if count > 0:
                print(f"  {name:6s}: {count}")

        excluded_note = "" if INCLUDE_METAPLASTIC else " (excluded: Metaplastic)"
        print(f"Mapping used: Superficial/Parabasal->NILM, "
              f"Koilocytotic->LSIL, Dyskeratotic->HSIL{excluded_note}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, riva_index = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        soft_label = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        soft_label[riva_index] = 1.0

        meta = {"path": image_path, "source": "sipakmed"}

        return image, soft_label, meta


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else r"C:\CervicalAI\data\raw\sipakmed"

    print(f"Scanning: {root}")
    dataset = SIPaKMeDDataset(root_dir=root, transform=None)
    print(f"\nLoaded {len(dataset)} samples successfully.")