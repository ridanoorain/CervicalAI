"""
Check Class Distribution

Purpose:
    Prints the class distribution (based on soft-label expected
    counts, same method used by compute_class_weights) for any
    RIVA split file -- train, val, or test.

Usage:
    python src/check_class_distribution.py
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from dataset.riva_dataset import RIVADataset
from training.loss_weighted import compute_class_weights

PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

SPLITS = {
    "train": os.path.join(PROJECT_ROOT, "data", "processed", "splits", "train.json"),
    "val": os.path.join(PROJECT_ROOT, "data", "processed", "splits", "val.json"),
    "test": os.path.join(PROJECT_ROOT, "data", "processed", "splits", "test.json"),
}

if __name__ == "__main__":
    for split_name, split_path in SPLITS.items():
        if not os.path.exists(split_path):
            print(f"\n[{split_name}] split file not found: {split_path}")
            continue

        print(f"\n{'=' * 70}")
        print(f"SPLIT: {split_name.upper()}")
        print(f"{'=' * 70}")

        dataset = RIVADataset(split_file=split_path, transform=None)

        # compute_class_weights() already prints the per-class
        # expected counts as part of its output.
        compute_class_weights(dataset)