import torch.nn as nn
import torch.nn.functional as F


class CNNBaseline(nn.Module):
    """Simple encoder-decoder baseline retained to separate CNN and ViT experiments."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.enc(x)
        disp = self.dec(feat)
        disp = F.interpolate(disp, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {0: disp}
