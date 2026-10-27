import torch
import torch.nn as nn

class MeanPooledVGG(nn.Module):
    def __init__(self, in_channels=5, projection_dim=128):
        super(MeanPooledVGG, self).__init__()

        # Feature extractor using only Conv2d and mean pooling
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.GELU(), 

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GELU(), 
            nn.AvgPool2d(kernel_size=4,stride=4),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GELU(), 
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GELU(), 
            nn.AvgPool2d(kernel_size=4,stride=4),
        )

        # Projector head
        self.projector = nn.Sequential(
            nn.Flatten(),              # (B, 512, 1, 1) -> (B, 512)
            nn.Linear(55296, 2048),
            nn.GELU(),
            nn.Linear(2048, 256),
            nn.GELU(),
            nn.Linear(256, projection_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        projection = self.projector(features)
        return projection
