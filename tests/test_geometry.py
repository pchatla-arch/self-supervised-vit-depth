import torch
from geometry.projection import disp_to_depth, transformation_from_parameters


def test_disp_to_depth_positive():
    _, depth = disp_to_depth(torch.full((1,1,4,4), 0.5))
    assert torch.all(depth > 0)


def test_identity_pose():
    aa = torch.zeros(2, 3)
    t = torch.zeros(2, 3)
    T = transformation_from_parameters(aa, t)
    assert torch.allclose(T, torch.eye(4).unsqueeze(0).repeat(2,1,1), atol=1e-5)
