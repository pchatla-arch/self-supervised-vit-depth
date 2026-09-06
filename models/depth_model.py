import torch.nn as nn
from .vit_encoder import ViTEncoder
from .depth_decoder import MultiScaleDepthDecoder


class MonocularDepthModel(nn.Module):
    def __init__(self, image_size=(192, 640), embed_dim=384, depth=6, heads=6):
        super().__init__()
        self.encoder = ViTEncoder(
            image_size=image_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=heads,
        )
        self.decoder = MultiScaleDepthDecoder(in_ch=embed_dim)

    def forward(self, x):
        feat = self.encoder(x)
        return self.decoder(feat, x.shape[-2:])
