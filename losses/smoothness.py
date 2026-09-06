import torch


def edge_aware_smoothness(disp, image):
    """Normalized edge-aware disparity smoothness."""
    mean_disp = disp.mean(dim=(2, 3), keepdim=True).clamp(min=1e-7)
    norm_disp = disp / mean_disp
    grad_disp_x = (norm_disp[:, :, :, :-1] - norm_disp[:, :, :, 1:]).abs()
    grad_disp_y = (norm_disp[:, :, :-1, :] - norm_disp[:, :, 1:, :]).abs()
    grad_img_x = (image[:, :, :, :-1] - image[:, :, :, 1:]).abs().mean(1, keepdim=True)
    grad_img_y = (image[:, :, :-1, :] - image[:, :, 1:, :]).abs().mean(1, keepdim=True)
    return (grad_disp_x * torch.exp(-grad_img_x)).mean() + (grad_disp_y * torch.exp(-grad_img_y)).mean()
