import torch
import torch.nn as nn


class PoseNet(nn.Module):
    """Small pose CNN predicting axis-angle + translation between two frames."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, 7, stride=2, padding=3), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(256, 6, 1)

    def forward(self, target, source):
        x = torch.cat([target, source], dim=1)
        x = self.encoder(x)
        x = self.head(x).mean(dim=(2, 3))
        # Small initialization-like scale stabilizes early training.
        return 0.01 * x[:, :3], 0.01 * x[:, 3:]
