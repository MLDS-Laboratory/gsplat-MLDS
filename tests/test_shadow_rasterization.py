"""Regression coverage for differentiable metadata-only shadow rasterization."""

import pytest
import torch


device = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("use_receiver_bias", [False, True])
def test_shadow_only_rasterization_backpropagates_to_gaussian_parameters(use_receiver_bias: bool) -> None:
    """Shadow numerator/denominator losses must reach the projected Gaussian inputs."""
    from gsplat.rendering import rasterization

    torch.manual_seed(7)
    count, width, height = 64, 32, 32
    means = torch.empty((count, 3), device=device).uniform_(-0.25, 0.25)
    means[:, 2].uniform_(2.0, 3.0)
    means.requires_grad_()
    quats = torch.randn((count, 4), device=device, requires_grad=True)
    scales = torch.full((count, 3), 0.08, device=device, requires_grad=True)
    opacities = torch.full((count,), 0.5, device=device, requires_grad=True)
    colors = torch.zeros((count, 1), device=device)
    viewmats = torch.eye(4, device=device).unsqueeze(0)
    Ks = torch.tensor(
        [[[32.0, 0.0, width / 2.0], [0.0, 32.0, height / 2.0], [0.0, 0.0, 1.0]]], device=device
    )
    receiver_bias = torch.zeros((count,), device=device) if use_receiver_bias else None

    _, _, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        packed=True,
        shadow_mode=True,
        n_total_gaussians=count,
        shadow_depth_group_eps=0.01 if use_receiver_bias else 0.0,
        use_shadow_receiver_bias=use_receiver_bias,
        shadow_receiver_bias=receiver_bias,
    )
    loss = meta["shadow_vis_num"].sum() + 0.1 * meta["shadow_vis_den"].sum()
    loss.backward()

    for parameter in (means, quats, scales, opacities):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
