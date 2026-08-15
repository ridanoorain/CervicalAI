
import os
import sys
import json
import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms


# ============================================================
# PROJECT PATH
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

SRC_DIR = os.path.dirname(CURRENT_DIR)

PROJECT_ROOT = os.path.dirname(SRC_DIR)

# Add src directory BEFORE importing project modules
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from dataset.riva_dataset import RIVADataset
from models.efficientnet import create_model, CLASS_NAMES



# ============================================================
# CONFIGURATION
# ============================================================

TEST_SPLIT = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "splits",
    "test.json"
)

CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "models",
    "checkpoints",
    "best_model.pth"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "evaluation",
    "gradcam"
)

HEATMAP_DIR = os.path.join(
    OUTPUT_DIR,
    "heatmaps"
)

RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "gradcam_results.json"
)

NUM_CLASSES = 8


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

TRANSFORM = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    print(f"\nCheckpoint:")
    print(CHECKPOINT)

    model = create_model(
        pretrained=False,
        num_classes=NUM_CLASSES,
        dropout=0.3
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Handle checkpoint format
    # --------------------------------------------------------

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model = model.to(DEVICE)

    model.eval()

    print("\nModel loaded successfully.")

    return model


# ============================================================
# GRAD-CAM CLASS
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model

        self.target_layer = target_layer

        self.activations = None

        self.gradients = None

        # ----------------------------------------------------
        # Register hooks
        # ----------------------------------------------------

        self.forward_hook = (
            target_layer.register_forward_hook(
                self.save_activation
            )
        )

        self.backward_hook = (
            target_layer.register_full_backward_hook(
                self.save_gradient
            )
        )

    # --------------------------------------------------------
    # Save forward activation
    # --------------------------------------------------------

    def save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output

    # --------------------------------------------------------
    # Save backward gradient
    # --------------------------------------------------------

    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]

    # --------------------------------------------------------
    # Generate Grad-CAM
    # --------------------------------------------------------

    def generate(
        self,
        image_tensor,
        target_class
    ):

        self.model.zero_grad()

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logits = self.model(
            image_tensor
        )

        # ----------------------------------------------------
        # Select target class score
        # ----------------------------------------------------

        score = logits[:, target_class]

        # ----------------------------------------------------
        # Backward pass
        # ----------------------------------------------------

        score.backward()

        # ----------------------------------------------------
        # Get activations and gradients
        # ----------------------------------------------------

        activations = self.activations

        gradients = self.gradients

        # ----------------------------------------------------
        # Global average pooling of gradients
        # ----------------------------------------------------

        weights = torch.mean(
            gradients,
            dim=(2, 3),
            keepdim=True
        )

        # ----------------------------------------------------
        # Weighted activation maps
        # ----------------------------------------------------

        cam = torch.sum(
            weights * activations,
            dim=1
        )

        # ----------------------------------------------------
        # ReLU
        # ----------------------------------------------------

        cam = F.relu(cam)

        # ----------------------------------------------------
        # Remove batch dimension
        # ----------------------------------------------------

        cam = cam[0]

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        cam = cam.detach().cpu().numpy()

        cam -= cam.min()

        if cam.max() > 0:

            cam /= cam.max()

        return cam, logits

    # --------------------------------------------------------
    # Remove hooks
    # --------------------------------------------------------

    def remove_hooks(self):

        self.forward_hook.remove()

        self.backward_hook.remove()


# ============================================================
# FIND TARGET LAYER
# ============================================================

def get_target_layer(model):

    """
    EfficientNet-B3 feature extractor.

    We use the final convolutional feature block before
    global pooling and classification.
    """

    return model.backbone.features[-1]


# ============================================================
# LOAD ORIGINAL IMAGE
# ============================================================

def load_original_image(dataset, index):

    """
    Get the original crop image path directly
    from RIVADataset.samples.
    """

    sample_metadata = dataset.samples[index]

    image_path = sample_metadata["crop_path"]

    if not os.path.isabs(image_path):

        image_path = os.path.join(
            PROJECT_ROOT,
            image_path
        )

    image_path = os.path.abspath(image_path)

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"Image does not exist: {image_path}"
        )

    return image_path

# ============================================================
# CREATE HEATMAP
# ============================================================

def create_heatmap(
    original_image,
    cam
):

    height, width = (
        original_image.shape[:2]
    )

    # --------------------------------------------------------
    # Resize CAM
    # --------------------------------------------------------

    cam_resized = cv2.resize(
        cam,
        (width, height)
    )

    # --------------------------------------------------------
    # Convert to 0-255
    # --------------------------------------------------------

    heatmap = np.uint8(
        255 * cam_resized
    )

    # --------------------------------------------------------
    # Apply color map
    # --------------------------------------------------------

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    # --------------------------------------------------------
    # Overlay heatmap
    # --------------------------------------------------------

    overlay = cv2.addWeighted(
        original_image,
        0.6,
        heatmap,
        0.4,
        0
    )

    return heatmap, overlay


# ============================================================
# MAIN GRAD-CAM ANALYSIS
# ============================================================

def run_gradcam():


    print("\n" + "=" * 70)
    print("RIVA GRAD-CAM VISUALIZATION")
    print("=" * 70)

    print(
        f"\nDevice: {DEVICE}"
    )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    os.makedirs(
        HEATMAP_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING TEST DATASET")
    print("=" * 70)

    print(
        f"\nTest split:\n"
        f"{TEST_SPLIT}"
    )

    dataset = RIVADataset(
        split_file=TEST_SPLIT,
        transform=TRANSFORM
    )

    print(
        f"\nTest samples: "
        f"{len(dataset)}"
    )

    # --------------------------------------------------------
    # Target layer
    # --------------------------------------------------------

    target_layer = get_target_layer(
        model
    )

    print(
        "\nGrad-CAM target layer:"
    )

    print(
        target_layer
    )

    # ========================================================
    # STEP 1: ANALYZE ALL TEST SAMPLES
    # ========================================================

    print("\n" + "=" * 70)
    print("ANALYZING TEST SAMPLES")
    print("=" * 70)

    analysis_results = []

    model.eval()

    with torch.no_grad():

        for index in range(len(dataset)):

            image_tensor, soft_label, majority_label = (
                dataset[index]
            )

            image_input = (
                image_tensor
                .unsqueeze(0)
                .to(DEVICE)
            )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            logits = model(image_input)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predicted_class = torch.argmax(
                probabilities,
                dim=1
            ).item()

            confidence = probabilities[
                0,
                predicted_class
            ].item()

            actual_class = torch.argmax(
                soft_label
            ).item()

            # ------------------------------------------------
            # Soft-label entropy
            # ------------------------------------------------

            entropy = -torch.sum(
                soft_label
                * torch.log(
                    soft_label + 1e-8
                )
            ).item()

            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            analysis_results.append({

                "index": index,

                "actual_class":
                    actual_class,

                "predicted_class":
                    predicted_class,

                "confidence":
                    confidence,

                "entropy":
                    entropy,

                "correct":
                    actual_class == predicted_class
            })

    # ========================================================
    # STEP 2: SELECT IMPORTANT SAMPLES
    # ========================================================

    print("\n" + "=" * 70)
    print("SELECTING IMPORTANT SAMPLES")
    print("=" * 70)

    selected_indices = []
    selected_categories = {}

    # --------------------------------------------------------
    # Helper function
    # --------------------------------------------------------

    def add_sample(
        category,
        candidates,
        number
    ):

        count = 0

        for result in candidates:

            index = result["index"]

            if index in selected_indices:
                continue

            selected_indices.append(index)

            selected_categories[index] = category

            count += 1

            if count >= number:
                break

    # --------------------------------------------------------
    # 1. High-confidence wrong predictions
    # --------------------------------------------------------

    high_confidence_wrong = sorted(

        [
            result
            for result in analysis_results
            if not result["correct"]
        ],

        key=lambda x: x["confidence"],

        reverse=True
    )

    add_sample(
        "high_confidence_wrong",
        high_confidence_wrong,
        3
    )

    # --------------------------------------------------------
    # 2. High-uncertainty samples
    # --------------------------------------------------------

    high_uncertainty = sorted(

        analysis_results,

        key=lambda x: x["entropy"],

        reverse=True
    )

    add_sample(
        "high_uncertainty",
        high_uncertainty,
        3
    )

    # --------------------------------------------------------
    # 3. Correct predictions
    # --------------------------------------------------------

    correct_predictions = sorted(

        [
            result
            for result in analysis_results
            if result["correct"]
        ],

        key=lambda x: x["confidence"]
    )

    add_sample(
        "correct_prediction",
        correct_predictions,
        2
    )

    # --------------------------------------------------------
    # 4. Other incorrect predictions
    # --------------------------------------------------------

    incorrect_predictions = sorted(

        [
            result
            for result in analysis_results
            if not result["correct"]
        ],

        key=lambda x: x["confidence"]
    )

    add_sample(
        "incorrect_prediction",
        incorrect_predictions,
        2
    )

    # --------------------------------------------------------
    # Limit to maximum 10 samples
    # --------------------------------------------------------

    selected_indices = selected_indices[:10]

    print(
        f"\nSelected "
        f"{len(selected_indices)} samples:"
    )

    for index in selected_indices:

        result = analysis_results[index]

        print(
            f"\nSample {index}"
        )

        print(
            f"  Category   : "
            f"{selected_categories[index]}"
        )

        print(
            f"  Actual     : "
            f"{CLASS_NAMES[result['actual_class']]}"
        )

        print(
            f"  Predicted  : "
            f"{CLASS_NAMES[result['predicted_class']]}"
        )

        print(
            f"  Confidence : "
            f"{result['confidence']:.4f}"
        )

        print(
            f"  Entropy    : "
            f"{result['entropy']:.4f}"
        )

    # ========================================================
    # STEP 3: CREATE GRAD-CAM
    # ========================================================

    gradcam = GradCAM(
        model,
        target_layer
    )

    results = []

    print("\n" + "=" * 70)
    print("GENERATING SELECTED GRAD-CAM HEATMAPS")
    print("=" * 70)

    # --------------------------------------------------------
    # Process selected samples only
    # --------------------------------------------------------

    for count, index in enumerate(
        selected_indices,
        start=1
    ):

        image_tensor, soft_label, majority_label = (
            dataset[index]
        )

        # ----------------------------------------------------
        # Add batch dimension
        # ----------------------------------------------------

        image_input = (
            image_tensor
            .unsqueeze(0)
            .to(DEVICE)
        )

        # ----------------------------------------------------
        # Determine target class
        # ----------------------------------------------------

        target_class = torch.argmax(
            soft_label
        ).item()

        # ----------------------------------------------------
        # Generate CAM
        # ----------------------------------------------------

        cam, logits = gradcam.generate(
            image_input,
            target_class
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()

        # ----------------------------------------------------
        # Original image
        # ----------------------------------------------------

        try:

            image_path = load_original_image(
                dataset,
                index
            )

        except Exception as error:

            print(
                f"\nSkipping sample {index}: "
                f"{error}"
            )

            continue

        original_image = cv2.imread(
            image_path
        )

        if original_image is None:

            print(
                f"\nCould not read image:"
                f"\n{image_path}"
            )

            continue

        # ----------------------------------------------------
        # Generate heatmap
        # ----------------------------------------------------

        heatmap, overlay = create_heatmap(
            original_image,
            cam
        )

        # ----------------------------------------------------
        # Create output filename
        # ----------------------------------------------------

        category = selected_categories[index]

        base_name = os.path.splitext(
            os.path.basename(image_path)
        )[0]

        output_filename = (

            f"{count:02d}_"
            f"{category}_"
            f"{base_name}_"
            f"actual_{CLASS_NAMES[target_class]}_"
            f"pred_{CLASS_NAMES[predicted_class]}.png"
        )

        output_path = os.path.join(
            HEATMAP_DIR,
            output_filename
        )

        # ----------------------------------------------------
        # Save overlay
        # ----------------------------------------------------

        cv2.imwrite(
            output_path,
            overlay
        )

        # ----------------------------------------------------
        # Save result information
        # ----------------------------------------------------

        results.append({

            "index":
                index,

            "category":
                category,

            "image_path":
                image_path,

            "actual_class":
                CLASS_NAMES[target_class],

            "predicted_class":
                CLASS_NAMES[predicted_class],

            "confidence":
                float(confidence),

            "soft_label_entropy":
                float(
                    analysis_results[index]["entropy"]
                ),

            "correct":
                bool(
                    target_class
                    == predicted_class
                ),

            "heatmap_path":
                output_path
        })

        print(
            f"Processed "
            f"{count}/"
            f"{len(selected_indices)}: "
            f"{category} | "
            f"{CLASS_NAMES[target_class]} -> "
            f"{CLASS_NAMES[predicted_class]}"
        )

    # --------------------------------------------------------
    # Remove hooks
    # --------------------------------------------------------

    gradcam.remove_hooks()

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    with open(
        RESULTS_PATH,
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    correct_count = sum(
        result["correct"]
        for result in results
    )

    incorrect_count = (
        len(results)
        - correct_count
    )

    print("\n" + "=" * 70)
    print("GRAD-CAM COMPLETE")
    print("=" * 70)

    print(
        f"\nHeatmaps generated: "
        f"{len(results)}"
    )

    print(
        f"Correct predictions: "
        f"{correct_count}"
    )

    print(
        f"Incorrect predictions: "
        f"{incorrect_count}"
    )

    print(
        f"\nHeatmaps saved to:"
    )

    print(
        HEATMAP_DIR
    )

    print(
        f"\nResults saved to:"
    )

    print(
        RESULTS_PATH
    )

    print("\n" + "=" * 70)



# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_gradcam()

