import torch
import torch.nn.functional as F
from .photometric import reprojection_loss
from .smoothness import edge_aware_smoothness
from geometry.projection import disp_to_depth, transformation_from_parameters, warp_source_to_target


def self_supervised_depth_loss(depth_outputs, pose_net, target, sources, K, inv_K,
                               smoothness_weight=1e-3, scale_weights=(1.0, 0.5, 0.25),
                               min_depth=0.1, max_depth=100.0):
    """Monodepth-style temporal reconstruction objective.

    No ground-truth depth is used. At each scale, source frames are warped into
    the target view using predicted depth + predicted relative pose. The minimum
    reprojection error across available source frames is optimized.
    """
    total = target.new_tensor(0.0)
    details = {}

    for i, (scale, disp) in enumerate(sorted(depth_outputs.items())):
        weight = scale_weights[i] if i < len(scale_weights) else 1.0 / (2 ** i)
        h, w = disp.shape[-2:]
        tgt = F.interpolate(target, (h, w), mode="bilinear", align_corners=False)
        srcs = [F.interpolate(s, (h, w), mode="bilinear", align_corners=False) for s in sources]

        # Scale intrinsics to this prediction resolution.
        K_s = K.clone()
        K_s[:, 0, :] *= w / target.shape[-1]
        K_s[:, 1, :] *= h / target.shape[-2]
        inv_K_s = torch.linalg.inv(K_s)

        _, depth = disp_to_depth(disp, min_depth=min_depth, max_depth=max_depth)
        reprojections = []
        for src_full, src in zip(sources, srcs):
            axisangle, translation = pose_net(target, src_full)
            T = transformation_from_parameters(axisangle, translation)
            warped = warp_source_to_target(src, depth, K_s, inv_K_s, T)
            reprojections.append(reprojection_loss(warped, tgt))

        reproj = torch.cat(reprojections, dim=1).min(dim=1, keepdim=True).values.mean()
        smooth = edge_aware_smoothness(disp, tgt)
        scale_loss = reproj + (smoothness_weight / (2 ** scale)) * smooth
        total = total + weight * scale_loss
        details[f"photo_s{scale}"] = float(reproj.detach())
        details[f"smooth_s{scale}"] = float(smooth.detach())

    return total, details
