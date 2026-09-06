import torch


def depth_metrics(gt, pred, min_depth=1e-3, max_depth=80.0, median_scale=True):
    mask = torch.isfinite(gt) & torch.isfinite(pred) & (gt > min_depth) & (gt < max_depth)
    gt = gt[mask]
    pred = pred[mask].clamp(min=min_depth, max=max_depth)
    if gt.numel() == 0:
        raise ValueError("No valid ground-truth depth pixels after masking")

    # Standard monocular evaluation uses median scaling because metric scale is ambiguous.
    if median_scale:
        pred = pred * (torch.median(gt) / torch.median(pred).clamp(min=1e-7))
        pred = pred.clamp(min=min_depth, max=max_depth)

    thresh = torch.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()
    rmse = torch.sqrt(((gt - pred) ** 2).mean())
    rmse_log = torch.sqrt(((torch.log(gt) - torch.log(pred)) ** 2).mean())
    abs_rel = (torch.abs(gt - pred) / gt).mean()
    sq_rel = (((gt - pred) ** 2) / gt).mean()
    return {
        "abs_rel": float(abs_rel), "sq_rel": float(sq_rel),
        "rmse": float(rmse), "rmse_log": float(rmse_log),
        "a1": float(a1), "a2": float(a2), "a3": float(a3),
    }
