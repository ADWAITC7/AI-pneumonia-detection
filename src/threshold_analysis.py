from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from data_loader import get_dataloaders
from model import get_model


# ============================================================
# 1. Paths and device
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = BASE_DIR / "DATASET"
MODEL_PATH = BASE_DIR / "model" / "pneumonia_model_finetuned.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

print("Device:", device)
print("Dataset:", DATASET_DIR)
print("Model:", MODEL_PATH)


# ============================================================
# 2. Load validation data
# ============================================================

train_loader, val_loader = get_dataloaders(
    DATASET_DIR,
    batch_size=32,
    val_split=0.2,
    seed=42,
)


# ============================================================
# 3. Load fine-tuned ResNet-18
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Fine-tuned model not found: {MODEL_PATH}"
    )

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
# 4. Get validation probabilities
# ============================================================

all_labels = []
all_pneumonia_probs = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = F.softmax(outputs, dim=1)

        pneumonia_probs = probabilities[:, 1]

        all_labels.extend(labels.cpu().numpy())
        all_pneumonia_probs.extend(
            pneumonia_probs.cpu().numpy()
        )


print()
print("Validation images:", len(all_labels))
print("Collected probabilities:", len(all_pneumonia_probs))


# ============================================================
# 5. Test different decision thresholds
# ============================================================

thresholds = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]


print()
print("=" * 70)
print("THRESHOLD RESULTS")
print("=" * 70)

print(
    f"{'Threshold':<10}"
    f"{'Accuracy':<11}"
    f"{'Precision':<11}"
    f"{'Recall':<11}"
    f"{'Specificity':<13}"
    f"{'F1':<10}"
)

print("-" * 70)


results = []


for threshold in thresholds:

    predictions = [
        1 if probability >= threshold else 0
        for probability in all_pneumonia_probs
    ]

    accuracy = accuracy_score(
        all_labels,
        predictions,
    )

    precision = precision_score(
        all_labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        predictions,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        all_labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp)

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    })

    print(
        f"{threshold:<10.2f}"
        f"{accuracy:<11.4f}"
        f"{precision:<11.4f}"
        f"{recall:<11.4f}"
        f"{specificity:<13.4f}"
        f"{f1:<10.4f}"
    )


# ============================================================
# 6. Find best thresholds
# ============================================================

best_f1 = max(
    results,
    key=lambda x: x["f1"],
)

best_accuracy = max(
    results,
    key=lambda x: x["accuracy"],
)

# We especially care about maintaining high pneumonia recall.
high_recall_results = [
    result
    for result in results
    if result["recall"] >= 0.95
]

if high_recall_results:

    best_high_recall = max(
        high_recall_results,
        key=lambda x: x["specificity"],
    )

else:

    best_high_recall = None


# ============================================================
# 7. Summary
# ============================================================

print()
print("=" * 70)
print("BEST THRESHOLDS")
print("=" * 70)

print(
    f"\nBest F1 threshold: "
    f"{best_f1['threshold']:.2f}"
)

print(
    f"Accuracy:    {best_f1['accuracy']:.4f}"
)

print(
    f"Precision:   {best_f1['precision']:.4f}"
)

print(
    f"Recall:      {best_f1['recall']:.4f}"
)

print(
    f"Specificity: {best_f1['specificity']:.4f}"
)

print(
    f"F1:          {best_f1['f1']:.4f}"
)


print(
    f"\nBest accuracy threshold: "
    f"{best_accuracy['threshold']:.2f}"
)

print(
    f"Accuracy:    {best_accuracy['accuracy']:.4f}"
)

print(
    f"Recall:      {best_accuracy['recall']:.4f}"
)

print(
    f"Specificity: {best_accuracy['specificity']:.4f}"
)


if best_high_recall:

    print(
        f"\nBest threshold with "
        f"recall >= 95%: "
        f"{best_high_recall['threshold']:.2f}"
    )

    print(
        f"Accuracy:    "
        f"{best_high_recall['accuracy']:.4f}"
    )

    print(
        f"Precision:   "
        f"{best_high_recall['precision']:.4f}"
    )

    print(
        f"Recall:      "
        f"{best_high_recall['recall']:.4f}"
    )

    print(
        f"Specificity: "
        f"{best_high_recall['specificity']:.4f}"
    )

    print(
        f"F1:          "
        f"{best_high_recall['f1']:.4f}"
    )


print()
print("=" * 70)
print("Threshold analysis complete!")
print("=" * 70)