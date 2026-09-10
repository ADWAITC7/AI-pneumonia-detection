from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split


def get_dataloaders(data_dir, batch_size=32, val_split=0.2, seed=42):

    data_dir = Path(data_dir)
    train_dir = data_dir / "train"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}"
        )

    # ------------------------
    # Training transforms
    # ------------------------
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    # ------------------------
    # Validation transforms
    # ------------------------
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    # Dataset with training augmentation
    train_full = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform
    )

    # Same images, but deterministic validation transform
    val_full = datasets.ImageFolder(
        root=train_dir,
        transform=val_transform
    )

    total_size = len(train_full)

    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(seed)

    train_subset, val_subset = random_split(
        range(total_size),
        [train_size, val_size],
        generator=generator
    )

    train_dataset = Subset(
        train_full,
        train_subset.indices
    )

    val_dataset = Subset(
        val_full,
        val_subset.indices
    )

    # ------------------------
    # DataLoaders
    # ------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # ------------------------
    # Information
    # ------------------------

    print("Classes:", train_full.classes)
    print("Class mapping:", train_full.class_to_idx)
    print("Total training images:", total_size)
    print("Train images:", len(train_dataset))
    print("Validation images:", len(val_dataset))

    return train_loader, val_loader