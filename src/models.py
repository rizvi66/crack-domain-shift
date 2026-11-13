"""Model factory: ImageNet-pretrained backbones with a 2-class head."""
from __future__ import annotations

import torch.nn as nn
from torchvision import models as tvm

MODEL_NAMES = ("resnet50", "efficientnet_b0", "densenet121", "resnet18")


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    if name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif name == "resnet18":  # small model, useful for quick tests
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    elif name == "densenet121":
        m = tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Linear(m.classifier.in_features, 2)
    else:
        raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}")
    return m
