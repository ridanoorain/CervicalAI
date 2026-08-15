
"""
RIVA EfficientNet-B3 Model

Purpose:
    Defines the EfficientNet-B3 model for cervical cytology
    cell classification using soft-label distributions.

Input:
    Cell crop image of size 224 x 224

Output:
    8-class logits corresponding to the RIVA classes.

Classes:
    0 -> NILM
    1 -> INFL
    2 -> LSIL
    3 -> HSIL
    4 -> SCC
    5 -> ENDO
    6 -> ASCH
    7 -> ASCUS
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


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
# EFFICIENTNET-B3 MODEL
# ============================================================

class RIVAEfficientNet(nn.Module):
    """
    EfficientNet-B3 for RIVA cervical cytology classification.

    The model outputs raw logits.
    Softmax should NOT be applied inside the model because
    the training loss will handle the probability conversion.
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        pretrained=True,
        dropout=0.3
    ):
        super().__init__()

        self.num_classes = num_classes

        # ----------------------------------------------------
        # Load EfficientNet-B3
        # ----------------------------------------------------

        if pretrained:
            weights = EfficientNet_B3_Weights.DEFAULT
        else:
            weights = None

        self.backbone = efficientnet_b3(weights=weights)

        # ----------------------------------------------------
        # Replace the original classifier
        # ----------------------------------------------------

        # EfficientNet-B3 classifier:
        #
        # Sequential(
        #     (0): Dropout(...)
        #     (1): Linear(1536, 1000)
        # )
        #
        # We replace it with an 8-class classifier.

        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    def forward(self, x):
        """
        Args:
            x: Tensor of shape
               [batch_size, 3, 224, 224]

        Returns:
            logits: Tensor of shape
                    [batch_size, 8]
        """

        return self.backbone(x)


# ============================================================
# MODEL FACTORY
# ============================================================

def create_model(
    pretrained=True,
    num_classes=NUM_CLASSES,
    dropout=0.3
):
    """
    Create an EfficientNet-B3 model for RIVA classification.

    Args:
        pretrained:
            Whether to use ImageNet pretrained weights.

        num_classes:
            Number of output classes.

        dropout:
            Dropout probability before the final classifier.

    Returns:
        RIVAEfficientNet model.
    """

    model = RIVAEfficientNet(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )

    return model


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("RIVA EFFICIENTNET-B3 MODEL TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = create_model(pretrained=True)

    model = model.to(device)

    print("\nModel created successfully.")

    # --------------------------------------------------------
    # Print class information
    # --------------------------------------------------------

    print("\nClasses:")

    for index, class_name in enumerate(CLASS_NAMES):
        print(f"  {index}: {class_name}")

    print(f"\nNumber of classes: {NUM_CLASSES}")

    # --------------------------------------------------------
    # Test input
    # --------------------------------------------------------

    dummy_input = torch.randn(
        2,
        3,
        224,
        224
    ).to(device)

    print("\nInput shape:")
    print(dummy_input.shape)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():
        logits = model(dummy_input)

    print("\nOutput logits shape:")
    print(logits.shape)

    # --------------------------------------------------------
    # Convert logits to probabilities
    # --------------------------------------------------------

    probabilities = torch.softmax(logits, dim=1)

    print("\nProbability shape:")
    print(probabilities.shape)

    print("\nProbability sums:")

    for i, sample in enumerate(probabilities):
        print(
            f"  Sample {i}: "
            f"{sample.sum().item():.4f}"
        )

    # --------------------------------------------------------
    # Predicted classes
    # --------------------------------------------------------

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
    # Parameter count
    # --------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nModel parameters:")
    print(f"  Total parameters     : {total_parameters:,}")
    print(f"  Trainable parameters : {trainable_parameters:,}")

    print("\n" + "=" * 70)
    print("MODEL TEST COMPLETE")
    print("=" * 70)

