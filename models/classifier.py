"""
Downstream classifier — ResNet-50 fine-tuned for DR grading.
Used in the ablation study: real-only vs real+synthetic data.
"""
import torch
import torch.nn as nn
from torchvision import models


class DRClassifier(nn.Module):
    """
    ResNet-50 with a custom classification head for 5-class DR grading.
    Pre-trained on ImageNet; only the final FC layer is replaced.
    """
    def __init__(self, n_classes=5, freeze_backbone=False):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    model = DRClassifier(n_classes=5)
    x     = torch.randn(4, 3, 224, 224)
    out   = model(x)
    print(f"Classifier output: {out.shape}")  # (4, 5)
    print("Classifier model check passed.")
