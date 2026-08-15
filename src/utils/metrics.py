"""
RIVA Per-Class Evaluation Utilities

Purpose:
    Overall accuracy hides what's actually happening on an imbalanced
    8-class problem. This module produces the diagnostics you need
    before deciding what to fix next:

      - per-class precision / recall / F1
      - confusion matrix (raw counts + normalized)
      - a saved PNG of the confusion matrix (for the paper)
      - a saved JSON/text report (for the paper's results table)

Usage (inside or after your validation/test loop):

    from utils.metrics import evaluate_predictions

    report = evaluate_predictions(
        all_targets,       # list/array of true class indices
        all_predictions,   # list/array of predicted class indices
        class_names=CLASS_NAMES,
        output_dir="reports/baseline",
        run_name="baseline_riva_only",
    )
"""

import os
import json

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)


def evaluate_predictions(
    y_true,
    y_pred,
    class_names,
    output_dir,
    run_name="run",
):
    """
    Compute and save a full per-class evaluation report.

    Args:
        y_true:
            Array-like of true class indices, shape [N].

        y_pred:
            Array-like of predicted class indices, shape [N].

        class_names:
            List of class name strings, in index order.

        output_dir:
            Directory to save the confusion matrix image and
            report JSON/text into. Created if it doesn't exist.

        run_name:
            Identifier used in filenames and printed headers, e.g.
            "baseline_riva_only", "weighted_loss", "sipakmed_transfer".
            Use a distinct name per experiment so you can compare
            saved reports later for your ablation table.

    Returns:
        dict containing:
            "accuracy": overall accuracy
            "macro_f1": macro-averaged F1
            "weighted_f1": support-weighted F1
            "per_class": dict of per-class precision/recall/f1/support
            "confusion_matrix": raw confusion matrix (list of lists)
    """

    os.makedirs(output_dir, exist_ok=True)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    num_classes = len(class_names)
    labels = list(range(num_classes))

    # ----------------------------------------------------------
    # Scalar metrics
    # ----------------------------------------------------------

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )

    weighted_f1 = f1_score(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    # ----------------------------------------------------------
    # Per-class precision / recall / F1 / support
    # ----------------------------------------------------------

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # ----------------------------------------------------------
    # Confusion matrix (raw + row-normalized)
    # ----------------------------------------------------------

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid divide-by-zero for empty classes
    cm_normalized = cm.astype(np.float64) / row_sums

    # ----------------------------------------------------------
    # Print summary to console
    # ----------------------------------------------------------

    print("\n" + "=" * 70)
    print(f"EVALUATION REPORT: {run_name}")
    print("=" * 70)

    print(f"\nOverall Accuracy : {accuracy:.4f}")
    print(f"Macro F1         : {macro_f1:.4f}")
    print(f"Weighted F1      : {weighted_f1:.4f}")

    print("\nPer-class breakdown:")
    print(f"  {'Class':8s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")

    for name in class_names:
        stats = report_dict[name]
        print(
            f"  {name:8s} "
            f"{stats['precision']:10.4f} "
            f"{stats['recall']:10.4f} "
            f"{stats['f1-score']:10.4f} "
            f"{int(stats['support']):10d}"
        )

    # Flag classes the model is effectively failing on --
    # this is usually where your next fix should target.
    weak_classes = [
        name for name in class_names
        if report_dict[name]["recall"] < 0.3 and report_dict[name]["support"] > 0
    ]
    if weak_classes:
        print(f"\n  ⚠ Classes with recall < 0.30: {', '.join(weak_classes)}")
        print("    These are dragging down macro F1 the most.")

    # ----------------------------------------------------------
    # Save confusion matrix plot
    # ----------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, matrix, title, fmt in [
        (axes[0], cm, "Confusion Matrix (counts)", "d"),
        (axes[1], cm_normalized, "Confusion Matrix (row-normalized)", ".2f"),
    ]:
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)

        thresh = matrix.max() / 2.0
        for i in range(num_classes):
            for j in range(num_classes):
                value = matrix[i, j]
                text = f"{value:{fmt}}"
                color = "white" if value > thresh else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{run_name}  |  Accuracy={accuracy:.4f}  Macro F1={macro_f1:.4f}")
    fig.tight_layout()

    cm_path = os.path.join(output_dir, f"{run_name}_confusion_matrix.png")
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nConfusion matrix saved to: {cm_path}")

    # ----------------------------------------------------------
    # Save JSON report (for pulling numbers into the paper later)
    # ----------------------------------------------------------

    result = {
        "run_name": run_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": {
            name: {
                "precision": report_dict[name]["precision"],
                "recall": report_dict[name]["recall"],
                "f1": report_dict[name]["f1-score"],
                "support": int(report_dict[name]["support"]),
            }
            for name in class_names
        },
        "confusion_matrix": cm.tolist(),
    }

    json_path = os.path.join(output_dir, f"{run_name}_report.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Report JSON saved to: {json_path}")
    print("=" * 70)

    return result


def compare_runs(report_paths, output_path=None):
    """
    Load multiple saved JSON reports and print/save a comparison
    table -- directly usable as your paper's ablation table.

    Args:
        report_paths:
            List of paths to *_report.json files, in the order you
            want them to appear (e.g. baseline, +weighted loss,
            +SIPaKMeD, +second model).

        output_path:
            Optional path to save the comparison as a CSV.

    Returns:
        List of dicts, one per run, with the key summary numbers.
    """

    rows = []

    for path in report_paths:
        with open(path) as f:
            data = json.load(f)

        rows.append({
            "run_name": data["run_name"],
            "accuracy": data["accuracy"],
            "macro_f1": data["macro_f1"],
            "weighted_f1": data["weighted_f1"],
        })

    print("\n" + "=" * 70)
    print("ABLATION COMPARISON")
    print("=" * 70)
    print(f"{'Run':30s} {'Accuracy':>10s} {'Macro F1':>10s} {'Weighted F1':>12s}")

    for row in rows:
        print(
            f"{row['run_name']:30s} "
            f"{row['accuracy']:10.4f} "
            f"{row['macro_f1']:10.4f} "
            f"{row['weighted_f1']:12.4f}"
        )

    if output_path:
        import csv

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nComparison table saved to: {output_path}")

    print("=" * 70)

    return rows