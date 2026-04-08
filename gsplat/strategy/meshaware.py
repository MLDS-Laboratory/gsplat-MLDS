from dataclasses import dataclass
from typing import Any, Dict, Literal, Tuple, Union

import torch

from .default import DefaultStrategy
from .ops import duplicate, duplicate_selected, remove, reset_opa, split


@dataclass
class MeshAwareStrategy(DefaultStrategy):
    use_mesh_pruning: bool = True
    protect_boundary: bool = True
    boundary_grow_grad_scale: float = 1.0  # OPTIONAL: set < 1.0 to densify boundary-shell Gaussians more easily
    min_gaussians: int = 0
    min_gaussians_mode: Literal["outside_only", "outside_or_big", "always"] = "outside_only"

    # bc densification can add Gaussians after og masks created, pad to ensure numerical stability
    @staticmethod
    def _align_length(values: torch.Tensor, target_len: int, fill_value: float | bool = 0.0) -> torch.Tensor:
        cur_len = values.shape[0]
        if cur_len == target_len:
            return values
        if cur_len > target_len:
            return values[:target_len]
        pad_shape = (target_len - cur_len, *values.shape[1:])
        pad = torch.full(pad_shape, fill_value, dtype=values.dtype, device=values.device)
        return torch.cat([values, pad], dim=0)

    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        """Callback function to be executed before the `loss.backward()` call."""
        assert self.key_for_gradient in info, "The 2D means of the Gaussians is required but missing."
        info[self.key_for_gradient].retain_grad()

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """Callback function to be executed after the `loss.backward()` call."""
        if step >= self.refine_stop_iter:
            return

        self._update_state(params, state, info, packed=packed)

        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        ):
            # grow GSs
            n_dupli, n_split = self._grow_gs(params, optimizers, state, step, info)
            if self.verbose:
                print(
                    f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                    f"Now having {len(params['means'])} GSs."
                )

            # prune GSs
            n_prune = self._prune_gs(params, optimizers, state, step, info)
            if self.verbose:
                print(f"Step {step}: {n_prune} GSs pruned. Now having {len(params['means'])} GSs.")

            # reset running stats
            state["grad2d"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            torch.cuda.empty_cache()

        if step % self.reset_every == 0:
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=self.prune_opa * 2.0,
            )

    @torch.no_grad()
    def _grow_gs(
        self,
        params,
        optimizers,
        state,
        step: int,
        info: Dict[str, Any],
    ) -> Tuple[int, int]:
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        valid_parent = torch.ones_like(grads, dtype=torch.bool)
        boundary_mask = None

        if self.use_mesh_pruning and "mesh_outside_mask" in info:
            outside_mask = self._align_length(info["mesh_outside_mask"], grads.shape[0], fill_value=False)
            valid_parent &= ~outside_mask
            if "mesh_boundary_mask" in info:
                boundary_mask = self._align_length(info["mesh_boundary_mask"], grads.shape[0], fill_value=False)

        grow_thresh = torch.full_like(grads, self.grow_grad2d)
        if boundary_mask is not None and self.boundary_grow_grad_scale != 1.0:
            # OPTIONAL: allow boundary-shell GSs to duplicate/split with a lower image-plane gradient.
            grow_thresh = torch.where(
                boundary_mask,
                grow_thresh * self.boundary_grow_grad_scale,
                grow_thresh,
            )

        is_grad_high = grads > self.grow_grad2d
        is_small = torch.exp(params["scales"]).max(dim=-1).values <= self.grow_scale3d * state["scene_scale"]

        is_dupli = is_grad_high & is_small & valid_parent
        n_dupli = is_dupli.sum().item()

        is_large = ~is_small
        is_split = is_grad_high & is_large & valid_parent
        if step < self.refine_scale2d_stop_iter:
            is_split |= (state["radii"] > self.grow_scale2d) & valid_parent
        n_split = is_split.sum().item()

        if n_dupli > 0:
            duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)

        is_split = torch.cat([is_split, torch.zeros(n_dupli, dtype=torch.bool, device=device)])

        if n_split > 0:
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                revised_opacity=self.revised_opacity,
            )

        return n_dupli, n_split

    @torch.no_grad()
    def _select_backfill_parents(
        self,
        params,
        state,
        candidate_mask: torch.Tensor,
        inside_mask: torch.Tensor,
        boundary_mask: torch.Tensor,
        sdf: torch.Tensor,
        n_needed: int,
    ) -> torch.Tensor:
        if n_needed <= 0:
            return torch.empty(0, dtype=torch.long, device=candidate_mask.device)

        preferred_pools = [candidate_mask & inside_mask, candidate_mask & boundary_mask, candidate_mask]
        pool_mask = next((mask for mask in preferred_pools if torch.any(mask)), None)
        if pool_mask is None:
            return torch.empty(0, dtype=torch.long, device=candidate_mask.device)

        pool = torch.where(pool_mask)[0]
        opacities = torch.sigmoid(params["opacities"].flatten()[pool])
        negative_sdf = torch.clamp(-sdf[pool], min=0.0)

        scores = opacities + 1e-6
        if torch.any(negative_sdf > 0):
            scores = scores + (negative_sdf / negative_sdf.max().clamp_min(1e-6))

        counts = state.get("count", None)
        if isinstance(counts, torch.Tensor):
            counts = self._align_length(counts, candidate_mask.shape[0], fill_value=0.0)[pool]
            if torch.any(counts > 0):
                scores = scores + 0.25 * (counts / counts.max().clamp_min(1.0))

        scores = scores + 0.5 * inside_mask[pool].float() + 0.1 * boundary_mask[pool].float()

        unique_take = min(int(n_needed), int(pool.numel()))
        selected = torch.empty(0, dtype=torch.long, device=pool.device)
        if unique_take > 0:
            selected = pool[torch.topk(scores, k=unique_take).indices]

        remaining = n_needed - int(selected.numel())
        if remaining <= 0:
            return selected

        sampled_local = torch.multinomial(scores.clamp_min(1e-6), remaining, replacement=True)
        return torch.cat([selected, pool[sampled_local]], dim=0)

    @torch.no_grad()
    def _backfill_min_gaussians(
        self,
        params,
        optimizers,
        state,
        info: Dict[str, Any],
        prune_mask: torch.Tensor,
        outside_mask: torch.Tensor,
        inside_mask: torch.Tensor,
        boundary_mask: torch.Tensor,
        weak_prune: torch.Tensor,
        big_prune: torch.Tensor,
    ) -> int:
        if self.min_gaussians <= 0:
            return 0

        outside_pruned = prune_mask & outside_mask
        big_pruned = prune_mask & big_prune
        any_pruned = bool(torch.any(prune_mask))

        if self.min_gaussians_mode == "outside_only":
            should_backfill = bool(torch.any(outside_pruned))
        elif self.min_gaussians_mode == "outside_or_big":
            should_backfill = bool(torch.any(outside_pruned) or torch.any(big_pruned))
        elif self.min_gaussians_mode == "always":
            should_backfill = any_pruned
        else:
            raise ValueError(f"Unknown min_gaussians_mode: {self.min_gaussians_mode}")

        if not should_backfill:
            return 0

        n_survivors = int(prune_mask.numel() - prune_mask.sum().item())
        n_needed = int(self.min_gaussians - n_survivors)
        if n_needed <= 0:
            return 0

        candidate_mask = (~prune_mask) & (~outside_mask)
        if not torch.any(candidate_mask):
            return 0

        sdf = info.get("mesh_sdf", torch.zeros_like(params["opacities"].flatten()))
        sdf = self._align_length(sdf, prune_mask.shape[0], fill_value=0.0)
        selected = self._select_backfill_parents(
            params=params,
            state=state,
            candidate_mask=candidate_mask,
            inside_mask=inside_mask,
            boundary_mask=boundary_mask,
            sdf=sdf,
            n_needed=n_needed,
        )
        if selected.numel() == 0:
            return 0

        duplicate_selected(params=params, optimizers=optimizers, state=state, sel=selected)
        return int(selected.numel())

    @torch.no_grad()
    def _prune_gs(
        self,
        params,
        optimizers,
        state,
        step: int,
        info: Dict[str, Any],
    ) -> int:
        opac = torch.sigmoid(params["opacities"].flatten())

        weak_prune = opac < self.prune_opa
        big_prune = torch.zeros_like(weak_prune)
        if step > self.reset_every:
            big_prune = torch.exp(params["scales"]).max(dim=-1).values > self.prune_scale3d * state["scene_scale"]
            if step < self.refine_scale2d_stop_iter:
                big_prune |= state["radii"] > self.prune_scale2d

        if self.use_mesh_pruning and "mesh_outside_mask" in info:
            target_len = weak_prune.shape[0]
            outside_mask = self._align_length(info["mesh_outside_mask"], target_len, fill_value=False)
            inside_mask = self._align_length(
                info.get("mesh_inside_mask", torch.zeros_like(outside_mask)),
                target_len,
                fill_value=False,
            )
            boundary_mask = self._align_length(
                info.get("mesh_boundary_mask", torch.zeros_like(outside_mask)),
                target_len,
                fill_value=False,
            )

            protected_mask = inside_mask
            if self.protect_boundary:
                protected_mask = protected_mask | boundary_mask

            is_prune = outside_mask | (weak_prune & ~protected_mask) | (big_prune & ~protected_mask)

            n_backfill = self._backfill_min_gaussians(
                params=params,
                optimizers=optimizers,
                state=state,
                info=info,
                prune_mask=is_prune,
                outside_mask=outside_mask,
                inside_mask=inside_mask,
                boundary_mask=boundary_mask,
                weak_prune=weak_prune,
                big_prune=big_prune,
            )
            if n_backfill > 0:
                is_prune = torch.cat(
                    [is_prune, torch.zeros(n_backfill, dtype=torch.bool, device=is_prune.device)],
                    dim=0,
                )
        else:
            is_prune = weak_prune | big_prune

        n_prune = int(is_prune.sum().item())
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune
