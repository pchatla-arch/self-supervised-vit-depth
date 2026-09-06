import math
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=384, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # B,C,H/P,W/P
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)  # B,N,C
        return x, (h, w)


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim=384, num_heads=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)

    def forward(self, x):
        y = self.norm1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """Compact Vision Transformer encoder for dense depth estimation.

    Unlike image-classification ViTs, this model keeps all patch tokens and
    returns a spatial feature map for the decoder.
    """

    def __init__(
        self,
        image_size=(192, 640),
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size)
        gh = math.ceil(image_size[0] / patch_size)
        gw = math.ceil(image_size[1] / patch_size)
        self.base_grid = (gh, gw)
        self.pos_embed = nn.Parameter(torch.zeros(1, gh * gw, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def _interpolate_pos_embed(self, grid_hw):
        gh, gw = grid_hw
        bgh, bgw = self.base_grid
        if (gh, gw) == (bgh, bgw):
            return self.pos_embed
        pos = self.pos_embed.transpose(1, 2).reshape(1, self.embed_dim, bgh, bgw)
        pos = torch.nn.functional.interpolate(pos, size=(gh, gw), mode="bicubic", align_corners=False)
        return pos.flatten(2).transpose(1, 2)

    def forward(self, x):
        tokens, (gh, gw) = self.patch_embed(x)
        tokens = tokens + self._interpolate_pos_embed((gh, gw))
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        feat = tokens.transpose(1, 2).reshape(x.shape[0], self.embed_dim, gh, gw)
        return feat
