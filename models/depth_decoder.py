import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class MultiScaleDepthDecoder(nn.Module):
    """CNN decoder producing disparity at full, half and quarter scales."""

    def __init__(self, in_ch=384):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, 256, 1)
        self.qtr = ConvBlock(256, 128)
        self.half_block = ConvBlock(128, 64)
        self.full = ConvBlock(64, 32)
        self.pred_qtr = nn.Conv2d(128, 1, 3, padding=1)
        self.pred_half = nn.Conv2d(64, 1, 3, padding=1)
        self.pred_full = nn.Conv2d(32, 1, 3, padding=1)

    @staticmethod
    def _disp(x):
        return torch.sigmoid(x)

    def forward(self, feat, output_size):
        h, w = output_size
        x = self.proj(feat)
        x = self.qtr(x)
        disp_qtr = self._disp(self.pred_qtr(x))

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.half_block(x)
        disp_half = self._disp(self.pred_half(x))

        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        x = self.full(x)
        disp_full = self._disp(self.pred_full(x))

        return {
            0: disp_full,
            1: disp_half,
            2: disp_qtr,
        }
