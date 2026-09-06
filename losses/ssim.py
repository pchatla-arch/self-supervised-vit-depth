import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIM(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.pad = nn.ReflectionPad2d(1)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.pad(x)
        y = self.pad(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x * x) - mu_x * mu_x
        sigma_y = self.sig_y_pool(y * y) - mu_y * mu_y
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        num = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        den = (mu_x * mu_x + mu_y * mu_y + self.C1) * (sigma_x + sigma_y + self.C2)
        return torch.clamp((1 - num / den) / 2, 0, 1)
