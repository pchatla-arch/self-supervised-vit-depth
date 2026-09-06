from pathlib import Path
import csv
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class FrameTripletDataset(Dataset):
    """Generic temporal triplet dataset.

    CSV columns:
      prev,target,next,fx,fy,cx,cy[,depth]

    Paths may be absolute or relative to root. Depth, when present, should be a
    torch-saved tensor or NumPy-compatible image prepared by the user for evaluation.
    """

    def __init__(self, manifest, root=".", height=192, width=640, training=True):
        self.root = Path(root)
        self.height = height
        self.width = width
        self.training = training
        with open(manifest, newline="") as f:
            self.rows = list(csv.DictReader(f))

        aug = []
        if training:
            aug += [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)]
        self.color = transforms.Compose(aug) if aug else None
        self.resize = transforms.Resize((height, width), antialias=True)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.rows)

    def _load(self, path):
        img = Image.open(self.root / path).convert("RGB")
        img = self.resize(img)
        if self.color is not None:
            img = self.color(img)
        return self.to_tensor(img)

    def __getitem__(self, idx):
        r = self.rows[idx]
        prev = self._load(r["prev"])
        target = self._load(r["target"])
        nxt = self._load(r["next"])

        fx, fy = float(r["fx"]), float(r["fy"])
        cx, cy = float(r["cx"]), float(r["cy"])
        K = torch.eye(4, dtype=torch.float32)
        K[0, 0], K[1, 1] = fx, fy
        K[0, 2], K[1, 2] = cx, cy
        inv_K = torch.linalg.inv(K)
        return {
            "target": target,
            "sources": torch.stack([prev, nxt], dim=0),
            "K": K,
            "inv_K": inv_K,
            "index": idx,
        }
