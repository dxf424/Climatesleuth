import torch
import torch.nn as nn
from torchvision import models

class MeanPooledResNet18(nn.Module):
    """
    ResNet-18 backbone with mean pooling and a projection head for contrastive learning.

    Args:
        in_channels (int): Number of input channels (default: 5).
        projection_dim (int): Output dimensionality of the projection head (default: 128).
    """
    def __init__(self, in_channels=5, projection_dim=128):
        super(MeanPooledResNet18, self).__init__()

        # Load ResNet-18 and replace input conv layer
        base_model = models.resnet18(pretrained=False)
        base_model.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Backbone: ResNet layers up to final pooling
        self.backbone = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
            nn.AdaptiveAvgPool2d((1, 1))  # Mean pooling over spatial dimensions
        )

        # Projection head for embeddings
        self.projector = nn.Sequential(
            nn.Flatten(),              # (B, C, 1, 1) → (B, C)
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, projection_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        projection = self.projector(features)
        return projection
