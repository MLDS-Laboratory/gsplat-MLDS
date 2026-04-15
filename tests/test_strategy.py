"""Tests for the functions in the CUDA extension.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

import pytest
import torch

device = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_strategy():
    from gsplat.rendering import rasterization
    from gsplat.strategy import DefaultStrategy, MCMCStrategy

    torch.manual_seed(42)

    # Prepare Gaussians
    N = 100
    params = torch.nn.ParameterDict(
        {
            "means": torch.randn(N, 3),
            "scales": torch.rand(N, 3),
            "quats": torch.randn(N, 4),
            "opacities": torch.rand(N),
            "colors": torch.rand(N, 3),
        }
    ).to(device)
    optimizers = {k: torch.optim.Adam([v], lr=1e-3) for k, v in params.items()}

    # A dummy rendering call
    render_colors, render_alphas, info = rasterization(
        means=params["means"],
        quats=params["quats"],  # F.normalize is fused into the kernel
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=params["colors"],
        viewmats=torch.eye(4).unsqueeze(0).to(device),
        Ks=torch.eye(3).unsqueeze(0).to(device),
        width=10,
        height=10,
        packed=False,
    )

    # Test DefaultStrategy
    strategy = DefaultStrategy(verbose=True)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()
    strategy.step_pre_backward(params, optimizers, state, step=600, info=info)
    render_colors.mean().backward(retain_graph=True)
    strategy.step_post_backward(params, optimizers, state, step=600, info=info)

    # Test MCMCStrategy
    strategy = MCMCStrategy(verbose=True)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()
    render_colors.mean().backward(retain_graph=True)
    strategy.step_post_backward(params, optimizers, state, step=600, info=info, lr=1e-3)


def _make_meshaware_test_case():
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(
                torch.tensor(
                    [
                        [10.0, 0.0, 0.0],
                        [20.0, 0.0, 0.0],
                        [30.0, 0.0, 0.0],
                        [40.0, 0.0, 0.0],
                        [50.0, 0.0, 0.0],
                    ]
                )
            ),
            "scales": torch.nn.Parameter(torch.zeros(5, 3)),
            "quats": torch.nn.Parameter(torch.randn(5, 4)),
            "opacities": torch.nn.Parameter(torch.tensor([4.0, 2.0, 1.0, 1.0, 3.0])),
        }
    )
    optimizers = {k: torch.optim.Adam([v], lr=1e-3) for k, v in params.items()}
    state = {
        "scene_scale": 1.0,
        "grad2d": torch.zeros(5),
        "count": torch.tensor([8.0, 5.0, 1.0, 1.0, 3.0]),
    }
    info = {
        "mesh_sdf": torch.tensor([-0.40, -0.10, 0.25, 0.30, -0.02]),
        "mesh_outside_mask": torch.tensor([False, False, True, True, False]),
        "mesh_inside_mask": torch.tensor([True, True, False, False, False]),
        "mesh_boundary_mask": torch.tensor([False, False, False, False, True]),
    }
    original_inside_parents = {
        tuple(params["means"][0].detach().tolist()),
        tuple(params["means"][1].detach().tolist()),
    }
    return params, optimizers, state, info, original_inside_parents


def test_meshaware_min_gaussians_backfills_inside_candidates():
    from gsplat.strategy import MeshAwareStrategy

    params, optimizers, state, info, original_inside_parents = _make_meshaware_test_case()
    strategy = MeshAwareStrategy(
        prune_opa=0.05,
        prune_scale3d=10.0,
        prune_scale2d=10.0,
        refine_scale2d_stop_iter=0,
        min_gaussians=4,
        min_gaussians_mode="outside_only",
    )

    n_prune = strategy._prune_gs(params, optimizers, state, step=101, info=info)

    assert n_prune == 2
    assert params["means"].shape[0] == 4
    assert state["count"].shape[0] == 4
    assert tuple(params["means"][-1].detach().tolist()) in original_inside_parents

    remaining_means = {tuple(row.tolist()) for row in params["means"].detach()}
    assert (30.0, 0.0, 0.0) not in remaining_means
    assert (40.0, 0.0, 0.0) not in remaining_means



# Modification test: prune Gaussians that leave the random_scale box.
def test_default_strategy_prunes_gaussians_outside_extent():
    from gsplat.strategy import DefaultStrategy

    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [0.6, 0.0, 0.0],
                        [0.0, -0.7, 0.0],
                    ]
                )
            ),
            "scales": torch.nn.Parameter(torch.zeros(3, 3)),
            "quats": torch.nn.Parameter(torch.randn(3, 4)),
            "opacities": torch.nn.Parameter(torch.full((3,), 4.0)),
        }
    )
    optimizers = {k: torch.optim.Adam([v], lr=1e-3) for k, v in params.items()}
    state = {"scene_scale": 1.0}

    strategy = DefaultStrategy(prune_opa=0.05, prune_outside_extent=1.0)
    n_prune = strategy._prune_gs(params, optimizers, state, step=1)

    assert n_prune == 2
    assert params["means"].shape[0] == 1
    assert torch.allclose(params["means"][0], torch.tensor([0.0, 0.0, 0.0]))


# Modification test: prune Gaussians that leave the random_scale box.
def test_meshaware_strategy_prunes_outside_extent_even_if_mesh_protected():
    from gsplat.strategy import MeshAwareStrategy

    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [0.8, 0.0, 0.0],
                        [0.2, 0.0, 0.0],
                    ]
                )
            ),
            "scales": torch.nn.Parameter(torch.zeros(3, 3)),
            "quats": torch.nn.Parameter(torch.randn(3, 4)),
            "opacities": torch.nn.Parameter(torch.full((3,), 4.0)),
        }
    )
    optimizers = {k: torch.optim.Adam([v], lr=1e-3) for k, v in params.items()}
    state = {"scene_scale": 1.0, "grad2d": torch.zeros(3), "count": torch.ones(3)}
    info = {
        "mesh_outside_mask": torch.zeros(3, dtype=torch.bool),
        "mesh_inside_mask": torch.ones(3, dtype=torch.bool),
        "mesh_boundary_mask": torch.zeros(3, dtype=torch.bool),
    }

    strategy = MeshAwareStrategy(prune_opa=0.05, prune_outside_extent=1.0)
    n_prune = strategy._prune_gs(params, optimizers, state, step=1, info=info)

    assert n_prune == 1
    assert params["means"].shape[0] == 2
    assert torch.all(params["means"].abs().max(dim=-1).values <= 0.5)


def test_meshaware_strategy_can_prune_big_inside_gaussians():
    from gsplat.strategy import MeshAwareStrategy

    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [0.2, 0.0, 0.0],
                    ]
                )
            ),
            "scales": torch.nn.Parameter(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [3.0, 3.0, 3.0],
                    ]
                )
            ),
            "quats": torch.nn.Parameter(torch.randn(2, 4)),
            "opacities": torch.nn.Parameter(torch.full((2,), 4.0)),
        }
    )
    optimizers = {k: torch.optim.Adam([v], lr=1e-3) for k, v in params.items()}
    state = {"scene_scale": 1.0, "grad2d": torch.zeros(2), "count": torch.ones(2)}
    info = {
        "mesh_outside_mask": torch.zeros(2, dtype=torch.bool),
        "mesh_inside_mask": torch.ones(2, dtype=torch.bool),
        "mesh_boundary_mask": torch.zeros(2, dtype=torch.bool),
    }

    strategy = MeshAwareStrategy(prune_opa=0.05, prune_scale3d=10.0, refine_scale2d_stop_iter=0)
    n_prune = strategy._prune_gs(params, optimizers, state, step=5001, info=info)

    assert n_prune == 1
    assert params["means"].shape[0] == 1
    assert torch.allclose(params["means"][0], torch.tensor([0.0, 0.0, 0.0]))

def test_meshaware_min_gaussians_mode_outside_only_does_not_backfill_non_outside_prunes():
    from gsplat.strategy import MeshAwareStrategy

    params, optimizers, state, info, _ = _make_meshaware_test_case()
    info["mesh_outside_mask"] = torch.zeros(5, dtype=torch.bool)
    info["mesh_inside_mask"] = torch.tensor([True, True, False, False, False])
    info["mesh_boundary_mask"] = torch.zeros(5, dtype=torch.bool)
    params["scales"] = torch.nn.Parameter(torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [3.0, 3.0, 3.0],
        [3.0, 3.0, 3.0],
        [3.0, 3.0, 3.0],
    ]))
    optimizers["scales"] = torch.optim.Adam([params["scales"]], lr=1e-3)

    strategy = MeshAwareStrategy(
        prune_opa=0.05,
        prune_scale3d=10.0,
        prune_scale2d=10.0,
        refine_scale2d_stop_iter=0,
        min_gaussians=4,
        min_gaussians_mode="outside_only",
    )

    n_prune = strategy._prune_gs(params, optimizers, state, step=5001, info=info)

    assert n_prune == 3
    assert params["means"].shape[0] == 2


def test_meshaware_min_gaussians_mode_always_backfills_non_outside_prunes():
    from gsplat.strategy import MeshAwareStrategy

    params, optimizers, state, info, original_inside_parents = _make_meshaware_test_case()
    info["mesh_outside_mask"] = torch.zeros(5, dtype=torch.bool)
    info["mesh_inside_mask"] = torch.tensor([True, True, False, False, False])
    info["mesh_boundary_mask"] = torch.zeros(5, dtype=torch.bool)
    params["scales"] = torch.nn.Parameter(torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [3.0, 3.0, 3.0],
        [3.0, 3.0, 3.0],
        [3.0, 3.0, 3.0],
    ]))
    optimizers["scales"] = torch.optim.Adam([params["scales"]], lr=1e-3)

    strategy = MeshAwareStrategy(
        prune_opa=0.05,
        prune_scale3d=10.0,
        prune_scale2d=10.0,
        refine_scale2d_stop_iter=0,
        min_gaussians=4,
        min_gaussians_mode="always",
    )

    n_prune = strategy._prune_gs(params, optimizers, state, step=5001, info=info)

    assert n_prune == 3
    assert params["means"].shape[0] == 4
    assert tuple(params["means"][-1].detach().tolist()) in original_inside_parents


def test_meshaware_min_gaussians_mode_outside_or_big_backfills_big_prunes():
    from gsplat.strategy import MeshAwareStrategy

    params, optimizers, state, info, original_inside_parents = _make_meshaware_test_case()
    info["mesh_outside_mask"] = torch.zeros(5, dtype=torch.bool)
    info["mesh_inside_mask"] = torch.tensor([True, True, False, False, False])
    info["mesh_boundary_mask"] = torch.zeros(5, dtype=torch.bool)
    params["scales"] = torch.nn.Parameter(torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [3.0, 3.0, 3.0],
        [3.0, 3.0, 3.0],
        [3.0, 3.0, 3.0],
    ]))
    optimizers["scales"] = torch.optim.Adam([params["scales"]], lr=1e-3)

    strategy = MeshAwareStrategy(
        prune_opa=0.05,
        prune_scale3d=10.0,
        prune_scale2d=10.0,
        refine_scale2d_stop_iter=0,
        min_gaussians=4,
        min_gaussians_mode="outside_or_big",
    )

    n_prune = strategy._prune_gs(params, optimizers, state, step=5001, info=info)

    assert n_prune == 3
    assert params["means"].shape[0] == 4
    assert tuple(params["means"][-1].detach().tolist()) in original_inside_parents


if __name__ == "__main__":
    test_strategy()
