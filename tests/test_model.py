import torch
from models.depth_model import MonocularDepthModel


def test_forward_shapes():
    model = MonocularDepthModel(image_size=(64, 128), embed_dim=96, depth=2, heads=4)
    x = torch.rand(2, 3, 64, 128)
    out = model(x)
    assert out[0].shape == (2, 1, 64, 128)
    assert set(out) == {0, 1, 2}
