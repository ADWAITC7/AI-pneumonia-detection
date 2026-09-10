import torch
import torch.nn as nn
import torchvision.models as models


def get_model(num_classes=2):

    # Load ImageNet-pretrained ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze the entire backbone first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the final ResNet block for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the classifier
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model