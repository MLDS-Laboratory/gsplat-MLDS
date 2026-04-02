from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch

from .default import DefaultStrategy
from .ops import duplicate, remove, reset_opa, split


@dataclass
class MeshAwareStrategy(DefaultStrategy):
    use_mesh_pruning: bool = True
    # bc densification can add Gaussians after og masks created, pad to ensure numerical stability
    @staticmethod
    def _align_mask_length(mask: torch.Tensor, target_len: int, fill_value: bool = False) -> torch.Tensor:
        cur_len = mask.shape[0]
        if cur_len == target_len:
            return mask
        if cur_len > target_len:
            return mask[:target_len]
        pad = torch.full((target_len - cur_len,), fill_value, dtype=torch.bool, device=mask.device)
        return torch.cat([mask, pad], dim=0)

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

        is_grad_high = grads > self.grow_grad2d
        is_small = torch.exp(params["scales"]).max(dim=-1).values <= self.grow_scale3d * state["scene_scale"]

        valid_parent = torch.ones_like(is_grad_high, dtype=torch.bool)
        if self.use_mesh_pruning and "mesh_outside_mask" in info:
            valid_parent &= ~info["mesh_outside_mask"]

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

        # bc densification can add Gaussians after og masks created, pad to ensure numerical stability
        def _align_mask_length(mask: torch.Tensor, target_len: int, fill_value: bool = False) -> torch.Tensor:
            cur_len = mask.shape[0]
            if cur_len == target_len:
                return mask
            if cur_len > target_len:
                return mask[:target_len]
            pad = torch.full((target_len - cur_len,), fill_value, dtype=torch.bool, device=mask.device)
            return torch.cat([mask, pad], dim=0)

        if self.use_mesh_pruning and "mesh_outside_mask" in info:
            target_len = weak_prune.shape[0]
            outside_mask = _align_mask_length(info["mesh_outside_mask"], target_len, fill_value=False)
            inside_mask = _align_mask_length(
                info.get("mesh_inside_mask", torch.zeros_like(outside_mask)),
                target_len,
                fill_value=False,
            )
            is_prune = outside_mask | (weak_prune & ~inside_mask) | (big_prune & ~inside_mask)
        else:
            is_prune = weak_prune | big_prune

        n_prune = int(is_prune.sum().item())
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune
