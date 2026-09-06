import torch
import torch.nn.functional as F


def disp_to_depth(disp, min_depth=0.1, max_depth=100.0):
    """Convert sigmoid disparity in (0,1) to metric-like inverse depth range."""
    min_disp = 1.0 / max_depth
    max_disp = 1.0 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1.0 / scaled_disp
    return scaled_disp, depth


def rot_from_axisangle(vec):
    """Convert Bx3 axis-angle vectors to Bx4x4 transforms."""
    angle = torch.norm(vec, dim=1, keepdim=True).clamp(min=1e-7)
    axis = vec / angle
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    ca, sa = torch.cos(angle[:, 0]), torch.sin(angle[:, 0])
    C = 1 - ca

    R = torch.zeros((vec.shape[0], 4, 4), device=vec.device, dtype=vec.dtype)
    R[:, 0, 0] = x * x * C + ca
    R[:, 0, 1] = x * y * C - z * sa
    R[:, 0, 2] = x * z * C + y * sa
    R[:, 1, 0] = y * x * C + z * sa
    R[:, 1, 1] = y * y * C + ca
    R[:, 1, 2] = y * z * C - x * sa
    R[:, 2, 0] = z * x * C - y * sa
    R[:, 2, 1] = z * y * C + x * sa
    R[:, 2, 2] = z * z * C + ca
    R[:, 3, 3] = 1.0
    return R


def transformation_from_parameters(axisangle, translation, invert=False):
    R = rot_from_axisangle(axisangle)
    t = translation.unsqueeze(-1)
    T = torch.eye(4, device=axisangle.device, dtype=axisangle.dtype).unsqueeze(0).repeat(axisangle.shape[0], 1, 1)
    if invert:
        R = R.transpose(1, 2)
        t = -torch.bmm(R[:, :3, :3], t)
    T[:, :3, :3] = R[:, :3, :3]
    T[:, :3, 3:4] = t
    return T


def backproject_depth(depth, inv_K):
    """B,1,H,W depth -> homogeneous camera points B,4,HW."""
    b, _, h, w = depth.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=depth.dtype),
        torch.arange(w, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    pix = torch.stack([xs, ys, torch.ones_like(xs)], dim=0).view(3, -1)
    pix = pix.unsqueeze(0).repeat(b, 1, 1)
    cam = torch.bmm(inv_K[:, :3, :3], pix) * depth.view(b, 1, -1)
    ones = torch.ones((b, 1, h * w), device=depth.device, dtype=depth.dtype)
    return torch.cat([cam, ones], dim=1)


def project_3d(points, K, T, height, width):
    """Project homogeneous 3D points to grid_sample coordinates."""
    P = torch.bmm(K, T)[:, :3, :]
    cam = torch.bmm(P, points)
    z = cam[:, 2:3].clamp(min=1e-7)
    pix = cam[:, :2] / z
    x = pix[:, 0].view(-1, height, width)
    y = pix[:, 1].view(-1, height, width)
    x = 2 * (x / max(width - 1, 1)) - 1
    y = 2 * (y / max(height - 1, 1)) - 1
    return torch.stack([x, y], dim=-1)


def warp_source_to_target(source, depth, K, inv_K, T):
    b, _, h, w = depth.shape
    points = backproject_depth(depth, inv_K)
    grid = project_3d(points, K, T, h, w)
    return F.grid_sample(source, grid, mode="bilinear", padding_mode="border", align_corners=True)
