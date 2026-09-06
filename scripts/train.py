import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from datasets.triplet_dataset import FrameTripletDataset
from models.depth_model import MonocularDepthModel
from models.pose_net import PoseNet
from losses.self_supervised import self_supervised_depth_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--root", default=".")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--height", type=int, default=192)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out", default="checkpoints")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = FrameTripletDataset(args.manifest, args.root, args.height, args.width, training=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    depth_model = MonocularDepthModel(image_size=(args.height, args.width)).to(device)
    pose_net = PoseNet().to(device)
    optimizer = torch.optim.AdamW(
        list(depth_model.parameters()) + list(pose_net.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        depth_model.train(); pose_net.train()
        running = 0.0
        for batch in loader:
            target = batch["target"].to(device)
            sources_t = batch["sources"].to(device)
            sources = [sources_t[:, i] for i in range(sources_t.shape[1])]
            K = batch["K"].to(device)
            inv_K = batch["inv_K"].to(device)

            outputs = depth_model(target)
            loss, _ = self_supervised_depth_loss(outputs, pose_net, target, sources, K, inv_K)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(depth_model.parameters()) + list(pose_net.parameters()), 1.0
            )
            optimizer.step()
            running += loss.item()

        scheduler.step()
        mean_loss = running / max(len(loader), 1)
        print(f"epoch={epoch+1:03d} loss={mean_loss:.6f}")
        torch.save({
            "depth_model": depth_model.state_dict(),
            "pose_net": pose_net.state_dict(),
            "epoch": epoch + 1,
        }, out / "last.pt")


if __name__ == "__main__":
    main()
