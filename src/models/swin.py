"""
RIVA Swin-Tiny Model

Purpose:
    Defines a Swin Transformer (Tiny) model for cervical cytology
    cell classification, as a second architecture alongside
    EfficientNet-B3 -- this mirrors the backbone family used by
    top-performing published RIVA Challenge submissions, making it
    a meaningful architectural comparison for your paper.

Input:
    Cell crop image of size 224 x 224

Output:
    8-class logits corresponding to the RIVA classes (same as
    RIVAEfficientNet, for drop-in compatibility with existing
    loss functions and evaluation code).

Classes:
    0 -> NILM
    1 -> INFL
    2 -> LSIL
    3 -> HSIL
    4 -> SCC
    5 -> ENDO
    6 -> ASCH
    7 -> ASCUS

Memory note:
    Swin-Tiny is chosen specifically (over Small/Base) to fit
    comfortably on a 6GB laptop GPU at batch_size=8, 224x224 input.
"""

import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights


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


class RIVASwinTiny(nn.Module):
    """
    Swin-Tiny for RIVA cervical cytology classification.

    Like RIVAEfficientNet, outputs raw logits -- softmax is handled
    by the loss function, not inside the model.
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, dropout=0.3):
        super().__init__()

        self.num_classes = num_classes

        weights = Swin_T_Weights.DEFAULT if pretrained else None
        self.backbone = swin_t(weights=weights)

        # ------------------------------------------------------------
        # Swin-Tiny's classifier is a single Linear layer:
        #   self.backbone.head = Linear(768, 1000)
        # Replace it with an 8-class head, matching the Dropout+Linear
        # pattern used in RIVAEfficientNet so both models are treated
        # consistently everywhere (checkpoint saving, freezing logic).
        # ------------------------------------------------------------

        in_features = self.backbone.head.in_features

        self.backbone.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, 3, 224, 224]
        Returns:
            logits: Tensor of shape [batch_size, 8]
        """
        return self.backbone(x)


def create_model(pretrained=True, num_classes=NUM_CLASSES, dropout=0.3):
    """
    Create a Swin-Tiny model for RIVA classification.
    Mirrors create_model() in models/efficientnet.py.
    """
    return RIVASwinTiny(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )


if __name__ == "__main__":
    print("=" * 70)
    print("RIVA SWIN-TINY MODEL TEST")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = create_model(pretrained=True).to(device)
    print("\nModel created successfully.")

    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    print(f"\nInput shape: {dummy_input.shape}")

    model.eval()
    with torch.no_grad():
        logits = model(dummy_input)

    print(f"Output logits shape: {logits.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")

    print("\n" + "=" * 70)
    print("MODEL TEST COMPLETE")
    print("=" * 70)