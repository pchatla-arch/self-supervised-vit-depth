import argparse
import csv
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.depth_model import MonocularDepthModel
from geometry.projection import disp_to_depth
from utils.metrics import depth_metrics


def load_depth(path):
    p = str(path)
    if p.endswith(".npy"):
        return torch.from_numpy(np.load(p)).float()
    if p.endswith(".pt") or p.endswith(".pth"):
        return torch.load(p, map_location="cpu").float()
    return torch.from_numpy(np.array(Image.open(p))).float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="CSV with image,depth columns")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--height", type=int, default=192)
    p.add_argument("--width", type=int, default=640)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MonocularDepthModel(image_size=(args.height, args.width)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["depth_model"] if "depth_model" in ckpt else ckpt)
    model.eval()
    tfm = transforms.Compose([transforms.Resize((args.height, args.width), antialias=True), transforms.ToTensor()])

    totals = {k: 0.0 for k in ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]}
    n = 0
    with open(args.manifest, newline="") as f:
        for row in csv.DictReader(f):
            image = tfm(Image.open(row["image"]).convert("RGB")).unsqueeze(0).to(device)
            gt = load_depth(row["depth"]).to(device)
            if gt.ndim == 2: gt = gt.unsqueeze(0).unsqueeze(0)
            elif gt.ndim == 3: gt = gt.unsqueeze(0)
            with torch.no_grad():
                _, pred = disp_to_depth(model(image)[0])
                pred = torch.nn.functional.interpolate(pred, gt.shape[-2:], mode="bilinear", align_corners=False)
            m = depth_metrics(gt, pred, max_depth=80.0, median_scale=True)
            for k, v in m.items(): totals[k] += v
            n += 1
    print({k: v / max(n, 1) for k, v in totals.items()})


if __name__ == "__main__":
    main()
