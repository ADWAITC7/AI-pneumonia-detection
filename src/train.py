from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import get_dataloaders
from model import get_model


# ------------------------
# 1. Setup
# ------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "DATASET"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
print("Dataset:", DATASET_DIR)

train_loader, val_loader = get_dataloaders(
    DATASET_DIR,
    batch_size=32,
    val_split=0.2,
    seed=42,
)


# ------------------------
# 2. Load baseline model
# ------------------------
model = get_model(num_classes=2)

baseline_path = MODEL_DIR / "pneumonia_model.pth"

if not baseline_path.exists():
    raise FileNotFoundError(
        f"Baseline model not found: {baseline_path}"
    )

print("Loading baseline model:", baseline_path)

model.load_state_dict(
    torch.load(
        baseline_path,
        map_location=device,
        weights_only=True,
    )
)

model = model.to(device)


# ------------------------
# 3. Fine-tuning setup
# ------------------------
criterion = nn.CrossEntropyLoss()

# layer4 was unfrozen in model.py.
# We fine-tune it with a small learning rate.
# The classifier gets a slightly larger learning rate.
optimizer = optim.Adam(
    [
        {
            "params": model.layer4.parameters(),
            "lr": 1e-5,
        },
        {
            "params": model.fc.parameters(),
            "lr": 1e-4,
        },
    ]
)


# ------------------------
# 4. Training config
# ------------------------
num_epochs = 5
best_val_acc = 0.0

fine_tuned_model_path = MODEL_DIR / "pneumonia_model_finetuned.pth"


# ------------------------
# 5. Training loop
# ------------------------
for epoch in range(num_epochs):

    # -------- TRAIN --------
    model.train()

    train_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss /= len(train_loader)
    train_acc = correct / total


    # -------- VALIDATION --------
    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total


    # -------- LOGGING --------
    print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")


    # -------- SAVE BEST FINE-TUNED MODEL --------
    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            fine_tuned_model_path,
        )

        print(
            f"💾 Best fine-tuned model saved: "
            f"{fine_tuned_model_path}"
        )


print("\nFine-tuning complete!")
print(f"Best validation accuracy: {best_val_acc:.4f}")
print(f"Model saved at: {fine_tuned_model_path}")