from pathlib import Path

import torch
import torch.nn.functional as F

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import get_model


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = BASE_DIR / "DATASET"
TEST_DIR = DATASET_DIR / "test"

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "pneumonia_model_finetuned.pth"
)

# Threshold selected using validation-set analysis
PNEUMONIA_THRESHOLD = 0.55


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)
print("Test dataset:", TEST_DIR)
print("Model:", MODEL_PATH)
print("Decision threshold:", PNEUMONIA_THRESHOLD)


# ============================================================
# 3. TEST PREPROCESSING
# ============================================================

# No random augmentation during testing.
# Images are resized and normalized using ImageNet statistics,
# matching the ResNet18 preprocessing.

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])


# ============================================================
# 4. LOAD TEST DATASET
# ============================================================

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test directory not found: {TEST_DIR}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Fine-tuned model not found: {MODEL_PATH}\n"
        "Run train.py first."
    )


test_dataset = datasets.ImageFolder(
    root=TEST_DIR,
    transform=test_transform,
)


test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)


print("Test images:", len(test_dataset))
print("Classes:", test_dataset.classes)
print("Class mapping:", test_dataset.class_to_idx)


# ============================================================
# 5. LOAD FINE-TUNED RESNET18
# ============================================================

model = get_model(num_classes=2)


state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True,
)


model.load_state_dict(state_dict)

model = model.to(device)

model.eval()


# ============================================================
# 6. RUN INFERENCE
# ============================================================

all_preds = []
all_labels = []
all_probabilities = []


with torch.no_grad():

    for images, labels in test_loader:

        # Move images to GPU if CUDA is available
        images = images.to(device)

        # Forward pass
        outputs = model(images)

        # Convert model logits into probabilities
        probabilities = F.softmax(outputs, dim=1)

        # Probability of PNEUMONIA
        pneumonia_probs = probabilities[:, 1]

        # ----------------------------------------------------
        # IMPORTANT:
        # Use our validation-selected threshold of 0.55
        # instead of the default 0.50 argmax decision.
        # ----------------------------------------------------

        preds = (
            pneumonia_probs >= PNEUMONIA_THRESHOLD
        ).long()


        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )


# ImageFolder keeps samples in the same order as DataLoader
all_paths = [
    path
    for path, _ in test_dataset.samples
]


# ============================================================
# 7. FINAL TEST METRICS
# ============================================================

print("\n")
print("=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

print(
    f"Threshold used: {PNEUMONIA_THRESHOLD}"
)


# ------------------------------------------------------------
# Confusion Matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_preds
)

print("\nConfusion Matrix:")
print(cm)


# ------------------------------------------------------------
# Classification Report
# ------------------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=[
            "Normal",
            "Pneumonia"
        ],
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# 8. ERROR ANALYSIS
# ============================================================

print("\n")
print("=" * 60)
print("ERROR ANALYSIS")
print("=" * 60)


false_positives = []
false_negatives = []


for i in range(len(all_labels)):

    actual = all_labels[i]
    predicted = all_preds[i]

    normal_probability = (
        all_probabilities[i][0]
    )

    pneumonia_probability = (
        all_probabilities[i][1]
    )

    path = all_paths[i]


    # --------------------------------------------------------
    # FALSE POSITIVE
    # Actual = NORMAL
    # Predicted = PNEUMONIA
    # --------------------------------------------------------

    if actual == 0 and predicted == 1:

        false_positives.append({
            "path": path,
            "normal_probability": normal_probability,
            "pneumonia_probability": pneumonia_probability,
        })


    # --------------------------------------------------------
    # FALSE NEGATIVE
    # Actual = PNEUMONIA
    # Predicted = NORMAL
    # --------------------------------------------------------

    elif actual == 1 and predicted == 0:

        false_negatives.append({
            "path": path,
            "normal_probability": normal_probability,
            "pneumonia_probability": pneumonia_probability,
        })


# ============================================================
# 9. FALSE POSITIVE ANALYSIS
# ============================================================

print(
    f"\nFalse Positives: {len(false_positives)}"
)

print(
    "NORMAL images predicted as PNEUMONIA"
)


print("\nTop 10 most confident false positives:")


false_positives.sort(
    key=lambda x: x["pneumonia_probability"],
    reverse=True
)


for item in false_positives[:10]:

    print(
        f"\nImage: {item['path']}"
        f"\nNormal probability:    "
        f"{item['normal_probability']:.4f}"
        f"\nPneumonia probability: "
        f"{item['pneumonia_probability']:.4f}"
    )


# ============================================================
# 10. FALSE NEGATIVE ANALYSIS
# ============================================================

print(
    f"\nFalse Negatives: {len(false_negatives)}"
)

print(
    "PNEUMONIA images predicted as NORMAL"
)


print("\nTop 10 most confident false negatives:")


false_negatives.sort(
    key=lambda x: x["normal_probability"],
    reverse=True
)


for item in false_negatives[:10]:

    print(
        f"\nImage: {item['path']}"
        f"\nNormal probability:    "
        f"{item['normal_probability']:.4f}"
        f"\nPneumonia probability: "
        f"{item['pneumonia_probability']:.4f}"
    )


# ============================================================
# 11. FINAL ERROR SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("ERROR ANALYSIS SUMMARY")
print("=" * 60)

print(
    f"Total test images: {len(all_labels)}"
)

print(
    f"False positives:   {len(false_positives)}"
)

print(
    f"False negatives:   {len(false_negatives)}"
)

print(
    f"\nDecision threshold: {PNEUMONIA_THRESHOLD}"
)

print("\nFinal test evaluation complete!")