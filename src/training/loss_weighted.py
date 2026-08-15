"""
RIVA Weighted Soft-Label Loss

Purpose:
    Extends the existing SoftLabelCrossEntropy / SoftLabelKLDivergence
    losses with per-class weighting, so rare classes (ASCUS, ASCH, SCC)
    contribute more to the gradient than majority classes (NILM, INFL).

Why this is needed:
    The original losses treat every sample equally regardless of its
    class. On RIVA's imbalanced distribution, that means the model can
    minimize loss mostly by getting NILM/INFL right and largely
    ignoring rare classes. Class weighting fixes this without touching
    the data itself.

How weighting works with soft labels:
    Each sample's target is a distribution over 8 classes, not a
    single label. We compute the sample's weight as the weighted sum
    of class_weights against its soft-label distribution:

        sample_weight = sum_i( target_i * class_weight_i )

    This means a sample that is 75% INFL / 25% ASCUS gets a weight
    that blends both classes' importance, proportional to how much
    each contributes to that sample's label. This is more faithful
    to soft labels than just using the argmax class.

Usage:
    from training.loss_weighted import (
        compute_class_weights,
        WeightedSoftLabelCrossEntropy,
        WeightedSoftLabelKLDivergence,
    )

    class_weights = compute_class_weights(train_dataset)  # torch.Tensor [8]
    criterion = WeightedSoftLabelKLDivergence(class_weights)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
# CLASS WEIGHT COMPUTATION
# ============================================================

def compute_class_weights(dataset, method="inverse_freq", cap=10.0):
    """
    Compute per-class weights from a dataset's soft-label targets.

    Args:
        dataset:
            A RIVADataset (or any dataset yielding
            (image, soft_label, meta) tuples) used for TRAINING.
            Must NOT be the val/test set.

        method:
            "inverse_freq"      -> weight = total / (num_classes * freq)
            "inverse_sqrt_freq" -> weight = sqrt(inverse_freq)
                                    (gentler, avoids extreme weights)

        cap:
            Maximum allowed weight, to avoid a near-zero-frequency
            class exploding the loss and destabilizing training.

    Returns:
        torch.Tensor of shape [num_classes], one weight per class.
    """

    if method not in ("inverse_freq", "inverse_sqrt_freq"):
        raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------
    # Accumulate expected class counts across all soft labels.
    # Using soft labels (not just argmax) gives a more accurate
    # picture of how much "weight" each class actually carries
    # in the dataset.
    # ------------------------------------------------------------

    class_counts = torch.zeros(NUM_CLASSES, dtype=torch.float64)

    for i in range(len(dataset)):
        _, soft_label, _ = dataset[i]

        if not torch.is_tensor(soft_label):
            soft_label = torch.tensor(soft_label, dtype=torch.float64)

        class_counts += soft_label.double()

    total = class_counts.sum()

    print("\nClass distribution (soft-label expected counts):")
    for name, count in zip(CLASS_NAMES, class_counts.tolist()):
        pct = 100.0 * count / total.item()
        print(f"  {name:6s}: {count:10.2f}  ({pct:5.2f}%)")

    # ------------------------------------------------------------
    # Avoid division by zero for classes with (near) no examples.
    # ------------------------------------------------------------

    class_counts = class_counts.clamp(min=1.0)

    if method == "inverse_freq":
        weights = total / (NUM_CLASSES * class_counts)
    else:  # inverse_sqrt_freq
        weights = torch.sqrt(total / (NUM_CLASSES * class_counts))

    # ------------------------------------------------------------
    # Cap extreme weights and renormalize so mean weight == 1.0
    # (keeps overall loss magnitude comparable to the unweighted
    # version, which makes learning-rate tuning easier).
    # ------------------------------------------------------------

    weights = weights.clamp(max=cap)
    weights = weights / weights.mean()

    print("\nComputed class weights:")
    for name, weight in zip(CLASS_NAMES, weights.tolist()):
        print(f"  {name:6s}: {weight:.4f}")

    return weights.float()


# ============================================================
# WEIGHTED SOFT-LABEL CROSS ENTROPY
# ============================================================

class WeightedSoftLabelCrossEntropy(nn.Module):
    """
    Soft-label cross entropy with per-class weighting.
    """

    def __init__(self, class_weights, reduction="mean"):
        super().__init__()

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")

        self.reduction = reduction

        if not torch.is_tensor(class_weights):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)

        self.register_buffer("class_weights", class_weights.float())

    def forward(self, logits, targets):
        if logits.shape != targets.shape:
            raise ValueError(
                f"logits and targets must have the same shape. "
                f"Got logits={logits.shape}, targets={targets.shape}"
            )

        targets = targets.float()

        log_probabilities = F.log_softmax(logits, dim=1)

        # Per-class cross-entropy contribution, unweighted.
        per_class_loss = -targets * log_probabilities  # [batch, num_classes]

        # Apply class weights before summing over classes.
        weighted_loss = per_class_loss * self.class_weights.unsqueeze(0)

        loss = weighted_loss.sum(dim=1)  # [batch]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ============================================================
# WEIGHTED SOFT-LABEL KL DIVERGENCE
# ============================================================

class WeightedSoftLabelKLDivergence(nn.Module):
    """
    KL divergence between expert soft labels and model predictions,
    with a per-sample weight derived from the class weights and
    the sample's own soft-label distribution.
    """

    def __init__(self, class_weights, reduction="mean"):
        super().__init__()

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")

        self.reduction = reduction

        if not torch.is_tensor(class_weights):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)

        self.register_buffer("class_weights", class_weights.float())

    def forward(self, logits, targets):
        if logits.shape != targets.shape:
            raise ValueError(
                f"logits and targets must have the same shape. "
                f"Got {logits.shape} and {targets.shape}"
            )

        targets = targets.float()

        log_probabilities = F.log_softmax(logits, dim=1)

        # Standard per-sample KL(P || Q), summed over classes,
        # not yet reduced over the batch.
        target_log_probabilities = torch.log(targets.clamp(min=1e-8))

        per_sample_kl = torch.sum(
            targets * (target_log_probabilities - log_probabilities),
            dim=1,
        )  # [batch]

        # Sample weight = weighted sum of class weights against this
        # sample's soft-label distribution. A sample that is mostly
        # ASCUS gets close to ASCUS's weight; a blended sample gets
        # a blended weight.
        sample_weights = torch.sum(
            targets * self.class_weights.unsqueeze(0),
            dim=1,
        )  # [batch]

        loss = per_sample_kl * sample_weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ============================================================
# WEIGHTED RANDOM SAMPLER (optional, complements loss weighting)
# ============================================================

def build_weighted_sampler(dataset, class_weights=None):
    """
    Build a torch WeightedRandomSampler that oversamples rare-class
    examples. This is complementary to loss weighting -- using both
    together is fine and common, but if you find training becomes
    unstable, try just ONE of the two first (loss weighting is
    usually the safer starting point).

    Args:
        dataset:
            Training dataset (RIVADataset), yielding
            (image, soft_label, meta).

        class_weights:
            Optional precomputed weights [num_classes]. If None,
            computed automatically via compute_class_weights().

    Returns:
        torch.utils.data.WeightedRandomSampler
    """

    from torch.utils.data import WeightedRandomSampler

    if class_weights is None:
        class_weights = compute_class_weights(dataset)

    sample_weights = []

    for i in range(len(dataset)):
        _, soft_label, _ = dataset[i]

        if not torch.is_tensor(soft_label):
            soft_label = torch.tensor(soft_label, dtype=torch.float32)

        # Weight each sample by its majority class's weight.
        majority_class = torch.argmax(soft_label).item()
        sample_weights.append(class_weights[majority_class].item())

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler