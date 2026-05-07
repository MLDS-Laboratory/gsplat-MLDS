from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch

from .default import DefaultStrategy
from .ops import duplicate, remove, split


@dataclass
class SupportAwareStrategy(DefaultStrategy):
    """Persistent multi-view support-aware densification/pruning strategy.

    This subclasses DefaultStrategy and keeps the default growth logic mostly
    unchanged, but replaces opacity-only weak pruning with support-aware weak
    pruning.

    Default weak pruning:
        prune_i = alpha_i < prune_opa

    Support-aware weak pruning:
        prune_i = (alpha_i < prune_opa) AND (support_i < support_score_thresh)

    The support score is a persistent EMA accumulated across training from:
        - visibility/count support
        - image-plane gradient support
        - screen-space radius support

    The normal short-window DefaultStrategy statistics, such as state["grad2d"]
    and state["count"], are still reset after each refinement event. The new
    support_*_ema statistics are intentionally not reset, because they are meant
    to represent long-horizon multi-view evidence for whether a Gaussian is
    useful scene geometry.
    """

    # How slowly persistent support decays. Larger = longer memory.
    # With refine_every=10, 0.98 or 0.99 is usually more stable than 0.95.
    support_ema_decay: float = 0.98

    # Weights for the support score components.
    support_count_weight: float = 1.0
    support_grad_weight: float = 1.0
    support_radii_weight: float = 0.25

    # If support_score >= this value, low opacity alone will not prune the Gaussian.
    support_score_thresh: float = 0.05

    # If support_score >= this value, the densification gradient threshold is scaled
    # by support_densify_grad_scale for that Gaussian.
    support_densify_score_thresh: float = 0.1

    # Multiplies the densification gradient threshold for highly supported
    # Gaussians. Values <1.0 make them easier to duplicate/split.
    support_densify_grad_scale: float = 1.0

    # Before this many steps, use default opacity pruning.
    # This avoids protecting random early Gaussians before the field has settled.
    support_warmup_steps: int = 1000

    # Usually False
    support_protect_big: bool = False

    # If True, prints pruning diagnostics each refinement step.
    support_verbose: bool = True

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """Initialize DefaultStrategy state plus persistent support state."""
        state = super().initialize_state(scene_scale=scene_scale)

        # Persistent support buffers. These are initialized lazily on first update
        # because the device and number of Gaussians are known then.
        state["support_count_ema"] = None
        state["support_grad_ema"] = None
        state["support_radii_ema"] = None

        # Optional diagnostic buffer, useful for debug visualization/logging.
        state["support_score"] = None

        # Optional pruning diagnostics.
        state["support_num_low_opacity"] = 0
        state["support_num_protected"] = 0
        state["support_num_weak_pruned"] = 0
        state["support_num_big_pruned"] = 0
        state["support_num_outside_extent_pruned"] = 0
        state["support_num_total_pruned"] = 0

        # Per-refinement densification diagnostics.
        state["support_num_duplicated"] = 0
        state["support_num_split"] = 0
        state["support_num_densified"] = 0
        state["support_num_support_eased_duplicated"] = 0
        state["support_num_support_eased_split"] = 0
        state["support_num_support_eased_densified"] = 0
        state["support_last_densify_step"] = -1
        state["support_last_cull_step"] = -1

        return state

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

    @staticmethod
    def _safe_normalize(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Robustly normalize a nonnegative per-Gaussian statistic to roughly [0, 1].

        A fixed absolute gradient threshold can be brittle across scenes and
        rasterization settings. Quantile normalization makes the support score
        less scene-scale dependent.

        Uses the 95th percentile instead of max so a few extreme Gaussians do
        not collapse the rest of the scores toward zero.
        """
        if values.numel() == 0:
            return values

        values = values.clamp_min(0.0)
        denom = torch.quantile(values.detach(), 0.95).clamp_min(eps)
        return (values / denom).clamp(0.0, 1.0)

    def _ensure_support_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        device: torch.device,
    ) -> None:
        """Create or resize persistent support buffers."""
        n_gaussian = len(params["means"])

        for key in ["support_count_ema", "support_grad_ema", "support_radii_ema"]:
            if state[key] is None:
                state[key] = torch.zeros(n_gaussian, device=device)
            else:
                state[key] = self._align_length(state[key], n_gaussian, fill_value=0.0)

    def _extract_visible_stats(
        self,
        info: Dict[str, Any],
        packed: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract visible Gaussian ids, gradient norms, and radii.

        This mirrors DefaultStrategy._update_state's packed/unpacked handling.
        It intentionally uses the same gradient normalization convention as the
        default strategy so that support_grad_ema is comparable to grad2d.
        """
        for key in ["width", "height", "n_cameras", "radii", "gaussian_ids", self.key_for_gradient]:
            assert key in info, f"{key} is required but missing."

        if self.absgrad:
            grads = info[self.key_for_gradient].absgrad.clone()
        else:
            grads = info[self.key_for_gradient].grad.clone()

        # Same screen-space normalization used by DefaultStrategy.
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        if packed:
            # grads: [nnz, 2]
            gs_ids = info["gaussian_ids"]
            radii = info["radii"]
            grad_norm = grads.norm(dim=-1)
        else:
            # grads: [C, N, 2], radii: [C, N]
            sel = info["radii"] > 0.0
            gs_ids = torch.where(sel)[1]
            grad_norm = grads[sel].norm(dim=-1)
            radii = info["radii"][sel]

        return gs_ids, grad_norm, radii

    @torch.no_grad()
    def _update_persistent_support(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ) -> None:
        """Update long-horizon multi-view support statistics.

        This is the main addition relative to DefaultStrategy.

        support_count_ema:
            Whether this Gaussian has repeatedly appeared in the rasterizer.

        support_grad_ema:
            Whether this Gaussian repeatedly receives image-plane gradients,
            which is a proxy for usefulness to reconstruction/optimization.

        support_radii_ema:
            Whether this Gaussian repeatedly occupies nontrivial screen space.

        These values decay every training step, then visible Gaussians get an
        increment. They should survive the short-window stat reset after each
        refine event.
        """
        device = params["means"].device
        self._ensure_support_state(params, state, device)

        gs_ids, grad_norm, radii = self._extract_visible_stats(info, packed=packed)
        if gs_ids.numel() == 0:
            # Still decay the support state slightly if nothing is visible.
            d = float(self.support_ema_decay)
            state["support_count_ema"].mul_(d)
            state["support_grad_ema"].mul_(d)
            state["support_radii_ema"].mul_(d)
            return

        d = float(self.support_ema_decay)
        one_minus_d = 1.0 - d

        # Decay all Gaussians.
        state["support_count_ema"].mul_(d)
        state["support_grad_ema"].mul_(d)
        state["support_radii_ema"].mul_(d)

        # Add support to visible Gaussians.
        state["support_count_ema"].index_add_(
            0,
            gs_ids,
            torch.ones_like(grad_norm, dtype=torch.float32) * one_minus_d,
        )

        state["support_grad_ema"].index_add_(
            0,
            gs_ids,
            grad_norm.detach().to(torch.float32) * one_minus_d,
        )

        # Normalize radii to approximately screen fraction before accumulation.
        radius_norm = radii.detach().to(torch.float32) / float(max(info["width"], info["height"]))
        state["support_radii_ema"].index_add_(
            0,
            gs_ids,
            radius_norm * one_minus_d,
        )

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ) -> None:
        """Update default short-window stats and persistent support stats."""
        # DefaultStrategy updates grad2d/count/radii for densification.
        super()._update_state(params, state, info, packed=packed)

        # New: persistent support stats for pruning protection.
        self._update_persistent_support(params, state, info, packed=packed)

        # Keep support_score fresh for metrics logging.
        # Otherwise it only appears after _prune_gs runs, and only after warmup.
        if state.get("support_count_ema", None) is not None:
            state["support_score"] = self._compute_support_score(params, state).detach()

    def _compute_support_score(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
    ) -> torch.Tensor:
        """Compute the normalized support score S_i for each Gaussian."""
        device = params["means"].device
        n_gaussian = len(params["means"])
        self._ensure_support_state(params, state, device)

        count = self._align_length(state["support_count_ema"], n_gaussian, fill_value=0.0)
        grad = self._align_length(state["support_grad_ema"], n_gaussian, fill_value=0.0)
        radii = self._align_length(state["support_radii_ema"], n_gaussian, fill_value=0.0)

        count_n = self._safe_normalize(count)
        grad_n = self._safe_normalize(grad)
        radii_n = self._safe_normalize(radii)

        total_weight = (
            float(self.support_count_weight) + float(self.support_grad_weight) + float(self.support_radii_weight)
        )
        if total_weight <= 0.0:
            # Degenerate setting: no support channels enabled.
            return torch.zeros(n_gaussian, device=device)

        score = (
            float(self.support_count_weight) * count_n
            + float(self.support_grad_weight) * grad_n
            + float(self.support_radii_weight) * radii_n
        ) / total_weight

        return score.clamp(0.0, 1.0)

    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> Tuple[int, int]:
        """DefaultStrategy growth copied here only to keep this file self-contained.

        You could omit this method and inherit DefaultStrategy._grow_gs directly.
        I am including it so the support-aware strategy remains easy to customize
        later if you want support-aware growth.
        """
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        support_eased_mask = torch.zeros_like(grads, dtype=torch.bool)
        grow_thresh = torch.full_like(grads, self.grow_grad2d)
        if (
            step >= self.support_warmup_steps
            and self.support_densify_grad_scale != 1.0
            and state.get("support_count_ema", None) is not None
        ):
            support_score = self._compute_support_score(params, state)
            state["support_score"] = support_score.detach()
            support_eased_mask = support_score >= self.support_densify_score_thresh
            grow_thresh = torch.where(
                support_eased_mask,
                grow_thresh * self.support_densify_grad_scale,
                grow_thresh,
            )

        is_grad_high = grads > grow_thresh
        is_small = torch.exp(params["scales"]).max(dim=-1).values <= self.grow_scale3d * state["scene_scale"]
        is_dupli = is_grad_high & is_small
        is_large = ~is_small
        is_split = is_grad_high & is_large

        if step < self.refine_scale2d_stop_iter:
            is_split |= state["radii"] > self.grow_scale2d

        remaining = None if self.cap_max is None else int(self.cap_max) - int(len(params["means"]))
        if remaining is not None:
            if remaining <= 0:
                return 0, 0

            dupli_idx = torch.where(is_dupli)[0]
            split_idx = torch.where(is_split)[0]

            if dupli_idx.numel() + split_idx.numel() > remaining:
                scores = grads

                keep_dupli = min(int(dupli_idx.numel()), remaining)
                if dupli_idx.numel() > keep_dupli:
                    top_dupli = torch.topk(scores[dupli_idx], k=keep_dupli, sorted=False).indices
                    kept_dupli_idx = dupli_idx[top_dupli]
                    new_is_dupli = torch.zeros_like(is_dupli)
                    new_is_dupli[kept_dupli_idx] = True
                    is_dupli = new_is_dupli
                    dupli_idx = kept_dupli_idx

                remaining -= int(dupli_idx.numel())
                if remaining <= 0:
                    is_split = torch.zeros_like(is_split)
                elif split_idx.numel() > remaining:
                    top_split = torch.topk(scores[split_idx], k=remaining, sorted=False).indices
                    kept_split_idx = split_idx[top_split]
                    new_is_split = torch.zeros_like(is_split)
                    new_is_split[kept_split_idx] = True
                    is_split = new_is_split

        n_dupli = int(is_dupli.sum().item())
        n_split = int(is_split.sum().item())
        n_support_eased_dupli = int((is_dupli & support_eased_mask).sum().item())
        n_support_eased_split = int((is_split & support_eased_mask).sum().item())

        state["support_num_duplicated"] = n_dupli
        state["support_num_split"] = n_split
        state["support_num_densified"] = n_dupli + n_split
        state["support_num_support_eased_duplicated"] = n_support_eased_dupli
        state["support_num_support_eased_split"] = n_support_eased_split
        state["support_num_support_eased_densified"] = n_support_eased_dupli + n_support_eased_split
        state["support_last_densify_step"] = int(step)

        if n_dupli > 0:
            duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)

        # New duplicated Gaussians did not exist when is_split was formed.
        # They should not also be split in the same step.
        is_split = torch.cat(
            [
                is_split,
                torch.zeros(n_dupli, dtype=torch.bool, device=device),
            ]
        )

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
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> int:
        """Support-aware pruning.

        This is the main behavioral change.

        Low opacity does not automatically mean prune. A low-opacity Gaussian is
        pruned only if it also has weak persistent multi-view support.

        Scale-based pruning and outside-extent pruning are preserved, because
        they target different failure modes than shadowed/dark useful geometry.
        """
        opac = torch.sigmoid(params["opacities"].flatten())
        n_gaussian = opac.shape[0]

        low_opacity = opac < self.prune_opa

        if step < self.support_warmup_steps:
            # Early training can contain many random Gaussians with accidental
            # visibility. During warmup, keep default weak pruning.
            support_score = torch.zeros(n_gaussian, device=opac.device)
            state["support_score"] = support_score.detach()
            support_protected = torch.zeros_like(low_opacity)
            weak_prune = low_opacity
        else:
            support_score = self._compute_support_score(params, state)
            state["support_score"] = support_score.detach()

            support_protected = support_score >= self.support_score_thresh
            weak_prune = low_opacity & ~support_protected

        # Diagnostics for understanding whether the method is protecting anything.
        state["support_num_low_opacity"] = int(low_opacity.sum().item())
        state["support_num_protected"] = int((low_opacity & support_protected).sum().item())
        state["support_num_weak_pruned"] = int(weak_prune.sum().item())

        outside_extent_mask = self._prune_outside_extent_mask(params)
        if outside_extent_mask is None:
            outside_extent_mask = torch.zeros_like(low_opacity)

        big_prune = torch.zeros_like(low_opacity)

        # Preserve default big-Gaussian pruning behavior.
        if step > self.reset_every:
            big_prune = torch.exp(params["scales"]).max(dim=-1).values > self.prune_scale3d * state["scene_scale"]

            if step < self.refine_scale2d_stop_iter:
                big_prune |= state["radii"] > self.prune_scale2d

            if self.support_protect_big and step >= self.support_warmup_steps:
                # Optional. Usually leave this False at first.
                big_prune = big_prune & ~support_protected

        # Keep reason buckets exclusive so they add up cleanly to total_pruned.
        outside_prune = outside_extent_mask
        weak_only_prune = weak_prune & ~outside_prune
        big_only_prune = big_prune & ~outside_prune & ~weak_prune
        is_prune = outside_prune | weak_only_prune | big_only_prune

        n_prune = int(is_prune.sum().item())
        state["support_num_weak_pruned"] = int(weak_only_prune.sum().item())
        state["support_num_big_pruned"] = int(big_only_prune.sum().item())
        state["support_num_outside_extent_pruned"] = int(outside_prune.sum().item())
        state["support_num_total_pruned"] = n_prune
        state["support_last_cull_step"] = int(step)

        if self.support_verbose and self.verbose:
            print(
                f"SupportAware prune diagnostics: "
                f"low_opacity={state['support_num_low_opacity']}, "
                f"support_protected={state['support_num_protected']}, "
                f"weak_pruned={state['support_num_weak_pruned']}, "
                f"big_pruned={state['support_num_big_pruned']}, "
                f"outside_extent_pruned={state['support_num_outside_extent_pruned']}, "
                f"total_pruned={n_prune}"
            )

        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune

    def _prune_outside_extent_mask(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    ) -> Optional[torch.Tensor]:
        """Same helper as your modified DefaultStrategy.

        This preserves your optional random_scale-box pruning behavior.
        """
        if self.prune_outside_extent is None:
            return None

        half_extent = 0.5 * float(self.prune_outside_extent)
        return (params["means"].abs() > half_extent).any(dim=-1)
