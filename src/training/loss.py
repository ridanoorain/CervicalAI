
"""
RIVA Soft-Label Loss

Purpose:
    Loss functions for training EfficientNet-B3 using
    multi-expert soft-label distributions.

RIVA classes:
    0 -> NILM
    1 -> INFL
    2 -> LSIL
    3 -> HSIL
    4 -> SCC
    5 -> ENDO
    6 -> ASCH
    7 -> ASCUS

Example target:

    Experts:
        ASCUS
        INFL
        INFL
        INFL

    Soft label:
        [0.00, 0.75, 0.00, 0.00,
         0.00, 0.00, 0.00, 0.25]

The model outputs logits:

    [z1, z2, ..., z8]

The loss compares the predicted probability
distribution with the expert soft-label distribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CLASS CONFIGURATION
# ============================================================

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
# SOFT-LABEL CROSS ENTROPY
# ============================================================

class SoftLabelCrossEntropy(nn.Module):
    """
    Cross-entropy loss for soft targets.

    Unlike standard CrossEntropyLoss, this accepts a
    probability distribution as the target.

    Example:

        target =
        [0.0, 0.75, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.25]

    Args:
        reduction:
            "mean", "sum", or "none"
    """

    def __init__(self, reduction="mean"):
        super().__init__()

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                "reduction must be 'mean', 'sum', or 'none'"
            )

        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits:
                Model outputs.

                Shape:
                    [batch_size, num_classes]

            targets:
                Soft-label probability distributions.

                Shape:
                    [batch_size, num_classes]

        Returns:
            Cross-entropy loss.
        """

        # ----------------------------------------------------
        # Validate dimensions
        # ----------------------------------------------------

        if logits.ndim != 2:
            raise ValueError(
                f"logits must have shape "
                f"[batch_size, num_classes], "
                f"got {logits.shape}"
            )

        if targets.ndim != 2:
            raise ValueError(
                f"targets must have shape "
                f"[batch_size, num_classes], "
                f"got {targets.shape}"
            )

        if logits.shape != targets.shape:
            raise ValueError(
                f"logits and targets must have the same shape. "
                f"Got logits={logits.shape}, "
                f"targets={targets.shape}"
            )

        # ----------------------------------------------------
        # Validate number of classes
        # ----------------------------------------------------

        if logits.size(1) != NUM_CLASSES:
            raise ValueError(
                f"Expected {NUM_CLASSES} classes, "
                f"got {logits.size(1)}"
            )

        # ----------------------------------------------------
        # Make sure targets are floating point
        # ----------------------------------------------------

        targets = targets.float()

        # ----------------------------------------------------
        # Validate target values
        # ----------------------------------------------------

        if torch.any(targets < 0):
            raise ValueError(
                "Soft-label targets cannot contain "
                "negative values."
            )

        if torch.any(targets > 1):
            raise ValueError(
                "Soft-label targets cannot contain "
                "values greater than 1."
            )

        # ----------------------------------------------------
        # Validate probability distribution
        # ----------------------------------------------------

        target_sums = targets.sum(dim=1)

        if not torch.allclose(
            target_sums,
            torch.ones_like(target_sums),
            atol=1e-4
        ):
            raise ValueError(
                "Soft-label targets must sum to 1. "
                f"Target sums: {target_sums}"
            )

        # ----------------------------------------------------
        # Convert logits to log probabilities
        # ----------------------------------------------------

        log_probabilities = F.log_softmax(
            logits,
            dim=1
        )

        # ----------------------------------------------------
        # Soft-label cross entropy
        #
        # L = - Σ y_i log(p_i)
        # ----------------------------------------------------

        loss = -torch.sum(
            targets * log_probabilities,
            dim=1
        )

        # ----------------------------------------------------
        # Reduction
        # ----------------------------------------------------

        if self.reduction == "mean":
            return loss.mean()

        elif self.reduction == "sum":
            return loss.sum()

        return loss


# ============================================================
# KL DIVERGENCE LOSS
# ============================================================

class SoftLabelKLDivergence(nn.Module):
    """
    KL divergence between the expert soft-label distribution
    and the model's predicted distribution.

    KL(P || Q)

    where:

        P = expert soft-label distribution
        Q = model prediction

    This can be used as an alternative or additional
    objective to soft-label cross entropy.
    """

    def __init__(self, reduction="batchmean"):
        super().__init__()

        if reduction not in [
            "none",
            "batchmean",
            "sum",
            "mean"
        ]:
            raise ValueError(
                "Invalid reduction for KL divergence."
            )

        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits:
                [batch_size, num_classes]

            targets:
                [batch_size, num_classes]

        Returns:
            KL divergence loss.
        """

        # ----------------------------------------------------
        # Validate shape
        # ----------------------------------------------------

        if logits.shape != targets.shape:
            raise ValueError(
                f"logits and targets must have "
                f"the same shape. "
                f"Got {logits.shape} and {targets.shape}"
            )

        targets = targets.float()

        # ----------------------------------------------------
        # Validate target distribution
        # ----------------------------------------------------

        if torch.any(targets < 0):
            raise ValueError(
                "Soft-label targets cannot be negative."
            )

        target_sums = targets.sum(dim=1)

        if not torch.allclose(
            target_sums,
            torch.ones_like(target_sums),
            atol=1e-4
        ):
            raise ValueError(
                "Soft-label targets must sum to 1."
            )

        # ----------------------------------------------------
        # Log probability of model prediction
        # ----------------------------------------------------

        log_probabilities = F.log_softmax(
            logits,
            dim=1
        )

        # ----------------------------------------------------
        # KL divergence
        #
        # PyTorch expects:
        #
        # KL(P || Q)
        #
        # with P represented as log-probabilities.
        # ----------------------------------------------------

        target_log_probabilities = torch.log(
            targets.clamp(min=1e-8)
        )

        loss = F.kl_div(
            log_probabilities,
            target_log_probabilities,
            reduction=self.reduction,
            log_target=True
        )

        return loss


# ============================================================
# LOSS FACTORY
# ============================================================

def create_loss(loss_type="soft_cross_entropy"):
    """
    Create the requested loss function.

    Available options:

        soft_cross_entropy
        kl_divergence

    Args:
        loss_type:
            Name of the loss function.

    Returns:
        PyTorch loss module.
    """

    if loss_type == "soft_cross_entropy":

        return SoftLabelCrossEntropy()

    elif loss_type == "kl_divergence":

        return SoftLabelKLDivergence()

    else:

        raise ValueError(
            f"Unknown loss type: {loss_type}"
        )


# ============================================================
# HELPER FUNCTION
# ============================================================

def soft_label_accuracy(logits, targets):
    """
    Calculate accuracy using the majority class
    represented by the soft-label distribution.

    This is useful for conventional classification
    metrics such as accuracy and F1.

    Example:

        target:
        [0.0, 0.75, 0.0, 0.25, ...]

        majority class = INFL

    Args:
        logits:
            [batch_size, num_classes]

        targets:
            [batch_size, num_classes]

    Returns:
        Accuracy as a float.
    """

    predictions = torch.argmax(
        logits,
        dim=1
    )

    target_classes = torch.argmax(
        targets,
        dim=1
    )

    correct = (
        predictions == target_classes
    ).float()

    return correct.mean().item()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("RIVA SOFT-LABEL LOSS TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    # --------------------------------------------------------
    # Create sample logits
    # --------------------------------------------------------

    logits = torch.tensor(
        [
            [
                0.1,
                2.0,
                0.2,
                0.1,
                0.0,
                0.0,
                0.0,
                1.0
            ],
            [
                2.5,
                0.2,
                0.1,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0
            ],
        ],
        dtype=torch.float32
    ).to(device)

    # --------------------------------------------------------
    # Example soft labels
    #
    # Sample 1:
    #
    # ASCUS + INFL + INFL + INFL
    #
    # => INFL = 0.75
    #    ASCUS = 0.25
    #
    # Sample 2:
    #
    # NILM + NILM + NILM
    #
    # => NILM = 1.0
    # --------------------------------------------------------

    targets = torch.tensor(
        [
            [
                0.0,   # NILM
                0.75,  # INFL
                0.0,   # LSIL
                0.0,   # HSIL
                0.0,   # SCC
                0.0,   # ENDO
                0.0,   # ASCH
                0.25   # ASCUS
            ],
            [
                1.0,   # NILM
                0.0,   # INFL
                0.0,   # LSIL
                0.0,   # HSIL
                0.0,   # SCC
                0.0,   # ENDO
                0.0,   # ASCH
                0.0    # ASCUS
            ],
        ],
        dtype=torch.float32
    ).to(device)

    print("\nLogits:")
    print(logits)

    print("\nSoft-label targets:")
    print(targets)

    print("\nTarget sums:")

    for i, target in enumerate(targets):

        print(
            f"  Sample {i}: "
            f"{target.sum().item():.4f}"
        )

    # --------------------------------------------------------
    # Soft cross entropy
    # --------------------------------------------------------

    soft_ce = SoftLabelCrossEntropy()

    ce_loss = soft_ce(
        logits,
        targets
    )

    print(
        f"\nSoft-label Cross Entropy: "
        f"{ce_loss.item():.6f}"
    )

    # --------------------------------------------------------
    # KL divergence
    # --------------------------------------------------------

    kl_loss = SoftLabelKLDivergence()

    kl_value = kl_loss(
        logits,
        targets
    )

    print(
        f"KL Divergence: "
        f"{kl_value.item():.6f}"
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = soft_label_accuracy(
        logits,
        targets
    )

    print(
        f"Majority-label Accuracy: "
        f"{accuracy:.4f}"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    predictions = torch.argmax(
        probabilities,
        dim=1
    )

    print("\nPredictions:")

    for i, prediction in enumerate(predictions):

        print(
            f"  Sample {i}: "
            f"{CLASS_NAMES[prediction.item()]}"
        )

    # --------------------------------------------------------
    # Target majority classes
    # --------------------------------------------------------

    target_classes = torch.argmax(
        targets,
        dim=1
    )

    print("\nTarget majority labels:")

    for i, target_class in enumerate(target_classes):

        print(
            f"  Sample {i}: "
            f"{CLASS_NAMES[target_class.item()]}"
        )

    print("\n" + "=" * 70)
    print("SOFT-LABEL LOSS TEST COMPLETE")
    print("=" * 70)

