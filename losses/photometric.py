import torch
from .ssim import SSIM


_ssim = SSIM()


def reprojection_loss(predicted_image, target_image, ssim_weight=0.85):
    global _ssim
    if next(_ssim.parameters(), None) is not None:
        _ssim = _ssim.to(predicted_image.device)
    l1 = (target_image - predicted_image).abs().mean(1, keepdim=True)
    ssim = _ssim(predicted_image, target_image).mean(1, keepdim=True)
    return ssim_weight * ssim + (1.0 - ssim_weight) * l1
