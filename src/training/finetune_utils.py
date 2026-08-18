"""
Fine-tuning Utilities: Differential LR + Gradual Unfreezing

Purpose:
    When fine-tuning a pretrained backbone (e.g. SIPaKMeD-pretrained
    EfficientNet) on a new dataset (RIVA) with a FRESH, randomly
    initialized classifier head, using one learning rate for both is
    a common cause of the pretrained features getting overwritten
    too aggressively -- the head's large early gradients (since it
    starts random) propagate back through the backbone and can wash
    out the useful features the backbone already learned.

    Two complementary fixes, both implemented here:

    1. Freeze the backbone for the first few epochs, so only the
       head trains initially and stabilizes before the backbone is
       touched at all.

    2. Once unfrozen, use a much smaller learning rate for the
       backbone than for the head, so the backbone adapts gently
       instead of being pulled hard toward RIVA-specific patterns.

Usage inside train.py's train() function:

    from training.finetune_utils import (
        freeze_backbone,
        unfreeze_backbone,
        create_differential_optimizer,
    )

    # After creating the model and optionally loading SIPaKMeD weights:
    if CONFIG["freeze_backbone_epochs"] > 0:
        freeze_backbone(model)

    optimizer = create_differential_optimizer(
        model,
        backbone_lr=CONFIG["backbone_lr"],
        head_lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    # Inside the epoch loop, before training that epoch:
    if epoch == CONFIG["freeze_backbone_epochs"] + 1:
        unfreeze_backbone(model)
        print(f"\\nBackbone unfrozen at epoch {epoch}. "
              f"Now training with backbone_lr={CONFIG['backbone_lr']}.")
"""

import torch


def freeze_backbone(model):
    """
    Freeze all backbone parameters (everything except the
    classifier head). The classifier head remains trainable.
    """
    frozen_count = 0
    for name, param in model.named_parameters():
        if not name.startswith("backbone.classifier"):
            param.requires_grad = False
            frozen_count += 1

    print(f"Backbone frozen: {frozen_count} parameter tensors set to requires_grad=False")


def unfreeze_backbone(model):
    """
    Unfreeze all backbone parameters, making the whole model
    trainable again.
    """
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            param.requires_grad = True
            unfrozen_count += 1

    print(f"Backbone unfrozen: {unfrozen_count} parameter tensors set to requires_grad=True")


def create_differential_optimizer(model, backbone_lr, head_lr, weight_decay):
    """
    Create an AdamW optimizer with two parameter groups: the
    backbone (lower LR, since it's pretrained and should shift
    gently) and the classifier head (higher LR, since it starts
    randomly initialized and needs to move further).

    Args:
        model:
            The RIVAEfficientNet model.

        backbone_lr:
            Learning rate for backbone parameters. Typically 5-10x
            smaller than head_lr. Start with something like 1e-5
            when head_lr is 1e-4.

        head_lr:
            Learning rate for the classifier head parameters (same
            value as your existing CONFIG["learning_rate"]).

        weight_decay:
            Weight decay, applied to both groups.

    Returns:
        torch.optim.AdamW with two parameter groups.

    Note:
        This works correctly whether the backbone is currently
        frozen or not -- frozen parameters (requires_grad=False)
        simply won't receive gradient updates regardless of which
        group they're in, so it's safe to build this optimizer
        once, before any freeze/unfreeze calls, and use it for the
        whole training run.
    """

    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if name.startswith("backbone.classifier"):
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )

    print(f"\nDifferential optimizer created:")
    print(f"  Backbone: {len(backbone_params)} tensors, lr={backbone_lr}")
    print(f"  Head:     {len(head_params)} tensors, lr={head_lr}")

    return optimizer