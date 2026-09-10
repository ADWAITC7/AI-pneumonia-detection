from pathlib import Path

from data_loader import get_dataloaders


# Project root
BASE_DIR = Path(__file__).resolve().parents[1]

# Actual dataset location
DATASET_DIR = BASE_DIR / "DATASET"

print("Dataset path:", DATASET_DIR)

train_loader, val_loader = get_dataloaders(
    DATASET_DIR,
    batch_size=32
)

print("\n✅ DataLoader working!")

images, labels = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("First labels:", labels[:10].tolist())