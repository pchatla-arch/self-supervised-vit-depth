import argparse
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from models.depth_model import MonocularDepthModel
from geometry.projection import disp_to_depth


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--height", type=int, default=192)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--output", default="depth.png")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MonocularDepthModel(image_size=(args.height, args.width)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["depth_model"] if "depth_model" in ckpt else ckpt)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((args.height, args.width), antialias=True),
        transforms.ToTensor(),
    ])
    image = tfm(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        disp = model(image)[0]
        _, depth = disp_to_depth(disp)
    plt.imsave(args.output, depth[0, 0].cpu().numpy(), cmap="magma")
    print(args.output)


if __name__ == "__main__":
    main()
