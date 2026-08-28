from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple, Union

import torch

from .default import DefaultStrategy
from .ops import duplicate, duplicate_selected, remove, reset_opa, split


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

    The min-Gaussian extension in this file is the support-aware analog of a
    mesh-aware minimum-count safeguard. Instead of selecting parents based on
    mesh SDF depth, it selects parents from a localized, support-backed object
    core inferred from the Gaussian cloud itself.
    """

    # -------------------------------------------------------------------------
    # Base persistent support-aware pruning settings.
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # RANDOM-BOX WARM-START ADDITION.
    # -------------------------------------------------------------------------
    # When Gaussians begin from a random box instead of COLMAP/SfM support,
    # early support statistics are unreliable. This staged warm-start keeps
    # Gaussians alive long enough for photometric gradients and a foreground
    # opacity prior to pull them toward the object before normal pruning starts.
    # -------------------------------------------------------------------------

    use_support_warmstart: bool = False
    support_warmstart_steps: int = 1000
    support_warmstart_decay_steps: int = 1000
    support_warmstart_disable_culling: bool = True
    support_warmstart_disable_densification: bool = True
    support_warmstart_means_only_optimization: bool = False
    support_warmstart_conservative_culling_steps: int = 1000
    support_warmstart_conservative_cull_alpha_scale: float = 0.5

    # Usually False. Big-scale pruning targets a different failure mode than
    # weak opacity pruning, so support should not protect big splats by default.
    support_protect_big: bool = False

    # If True, prints pruning diagnostics each refinement step.
    support_verbose: bool = True

    # -------------------------------------------------------------------------
    # MIN-GAUSSIANS ADDITION: configuration.
    # -------------------------------------------------------------------------
    # This is the support-aware analog of mesh-aware min_gaussians.
    #
    # The strategy prevents catastrophic collapse by duplicating high-quality
    # parents if pruning would leave fewer than min_gaussians Gaussians.
    #
    # Unlike a naive minimum-count clamp, it does NOT pick arbitrary survivors.
    # It builds a localized candidate pool from:
    #   - high support_score,
    #   - nontrivial opacity,
    #   - not near the random/pruning box edge,
    #   - near the robust center of the supported Gaussian cloud.
    # -------------------------------------------------------------------------

    min_gaussians: int = 0
    """Minimum number of Gaussians to preserve by support-aware backfilling.

    If <= 0, this feature is disabled.
    """

    min_gaussians_mode: Literal["weak_only", "weak_or_big", "always"] = "weak_only"
    """Which pruning events can trigger support-aware backfilling.

    weak_only:
        Backfill only if low-opacity weak pruning would drop below min_gaussians.

    weak_or_big:
        Backfill if weak pruning or big-scale pruning would drop below min_gaussians.

    always:
        Backfill after any prune reason, including outside-extent pruning.
    """

    min_gaussians_start_step: int = 1000
    """Do not run min-Gaussian backfill before this step.

    This avoids locking in random initialization artifacts before the scene has
    developed a meaningful support distribution.
    """

    min_backfill_support_score: float = 0.05
    """Minimum support score for a Gaussian to be eligible as a backfill parent."""

    min_backfill_opacity: float = 0.01
    """Minimum post-sigmoid opacity for a Gaussian to be eligible as a backfill parent."""

    backfill_extent: Optional[float] = None
    """Optional full random/pruning-box extent used to reject edge-of-box parents.

    If None, this falls back to prune_outside_extent. Passing the model's
    random_scale here is recommended if you want min-Gaussian backfill to avoid
    the random box boundary even when prune_outside_random_scale_box is disabled.
    """

    backfill_edge_margin_frac: float = 0.15
    """Fractional margin removed from the backfill candidate box.

    Example:
        If backfill_extent = 0.2, the full box is roughly [-0.1, 0.1].
        With backfill_edge_margin_frac = 0.15, candidates must lie within
        [-0.085, 0.085] along every axis.
    """

    backfill_use_supported_cloud: bool = True
    """If True, localize backfill parents to the robust supported Gaussian cloud."""

    backfill_cloud_quantile: float = 0.80
    """Quantile of supported-cloud distances used as the robust object-core radius."""

    backfill_cloud_radius_scale: float = 1.25
    """Scale applied to the robust supported-cloud radius for candidate filtering."""

    backfill_min_cloud_points: int = 64
    """Minimum number of high-support Gaussians required to estimate the supported cloud."""

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
        state["support_num_pruned_this_step"] = 0

        # Per-refinement densification diagnostics.
        state["support_num_duplicated"] = 0
        state["support_num_split"] = 0
        state["support_num_densified"] = 0
        state["support_num_support_eased_duplicated"] = 0
        state["support_num_support_eased_split"] = 0
        state["support_num_support_eased_densified"] = 0
        state["support_last_densify_step"] = -1
        state["support_last_cull_step"] = -1

        # ---------------------------------------------------------------------
        # MIN-GAUSSIANS ADDITION: diagnostics.
        # ---------------------------------------------------------------------
        # These are scalar diagnostics updated whenever _prune_gs runs.
        # They let you verify whether the min-Gaussian safeguard is actually
        # doing anything and whether it had a usable localized parent pool.
        # ---------------------------------------------------------------------
        state["support_num_backfilled"] = 0
        state["support_backfill_candidate_count"] = 0
        state["support_backfill_needed"] = 0
        state["support_last_backfill_step"] = -1

        return state

    # added for the warm-start
    def _in_support_warmstart(self, step: int) -> bool:
        return bool(self.use_support_warmstart) and step < int(self.support_warmstart_steps)

    def _in_conservative_culling(self, step: int) -> bool:
        if not self.use_support_warmstart:
            return False
        start = int(self.support_warmstart_steps)
        stop = start + int(self.support_warmstart_conservative_culling_steps)
        return start <= step < stop

    def _support_ready_step(self) -> int:
        if not self.use_support_warmstart:
            return int(self.support_warmup_steps)
        return min(int(self.support_warmup_steps), int(self.support_warmstart_steps))

    def _means_only_warmstart_active(self, step: int) -> bool:
        return self._in_support_warmstart(step) and bool(self.support_warmstart_means_only_optimization)

    def _warmstart_decay_support_score_scale(self, step: int) -> float:
        if not self.use_support_warmstart:
            return 1.0
        start = int(self.support_warmstart_steps)
        decay_steps = max(1, int(self.support_warmstart_decay_steps))
        if step < start:
            return 1.0
        if step >= start + decay_steps:
            return 1.0
        frac = float(step - start) / float(decay_steps)
        return 0.1 + 0.9 * frac

    @staticmethod
    def _reset_densify_diagnostics(state: Dict[str, Any]) -> None:
        state["support_num_duplicated"] = 0
        state["support_num_split"] = 0
        state["support_num_densified"] = 0
        state["support_num_support_eased_duplicated"] = 0
        state["support_num_support_eased_split"] = 0
        state["support_num_support_eased_densified"] = 0

    @staticmethod
    def _reset_cull_diagnostics(state: Dict[str, Any]) -> None:
        state["support_num_low_opacity"] = 0
        state["support_num_protected"] = 0
        state["support_num_weak_pruned"] = 0
        state["support_num_big_pruned"] = 0
        state["support_num_outside_extent_pruned"] = 0
        state["support_num_total_pruned"] = 0
        state["support_num_backfilled"] = 0
        state["support_backfill_candidate_count"] = 0
        state["support_backfill_needed"] = 0

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """Refinement step with optional support warm-start gating for random-box init."""
        if self._refinement_stopped(step):
            self._maybe_print_refinement_stopped(state, step)
            return

        state["support_num_pruned_this_step"] = 0
        self._update_state(params, state, info, packed=packed)

        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        ):
            self._print_refinement_status(step)
            in_warmstart = self._in_support_warmstart(step)
            means_only_warmstart = self._means_only_warmstart_active(step)
            skip_densify = in_warmstart and (self.support_warmstart_disable_densification or means_only_warmstart)
            skip_cull = in_warmstart and (self.support_warmstart_disable_culling or means_only_warmstart)

            if skip_densify or self._densification_stopped(step):
                self._reset_densify_diagnostics(state)
                n_dupli, n_split = 0, 0
            else:
                n_dupli, n_split = self._grow_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                    f"Now having {len(params['means'])} GSs."
                )

            if skip_cull:
                self._reset_cull_diagnostics(state)
                n_prune = 0
            else:
                n_prune = self._prune_gs(params, optimizers, state, step)
                if self.verbose:
                    print(f"Step {step}: {n_prune} GSs pruned. Now having {len(params['means'])} GSs.")
            state["support_num_pruned_this_step"] = int(n_prune)

            state["grad2d"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            torch.cuda.empty_cache()

        if step % self.reset_every == 0 and not self._means_only_warmstart_active(step):
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=self.prune_opa * 2.0,
            )

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
        """Robustly normalize a nonnegative per-Gaussian statistic to roughly [0, 1]."""
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
            if state.get(key, None) is None:
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
        """Update long-horizon multi-view support statistics."""
        device = params["means"].device
        self._ensure_support_state(params, state, device)

        d = float(self.support_ema_decay)
        one_minus_d = 1.0 - d

        gs_ids, grad_norm, radii = self._extract_visible_stats(info, packed=packed)
        if gs_ids.numel() == 0:
            # Still decay the support state slightly if nothing is visible.
            state["support_count_ema"].mul_(d)
            state["support_grad_ema"].mul_(d)
            state["support_radii_ema"].mul_(d)
            return

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

    # -------------------------------------------------------------------------
    # MIN-GAUSSIANS ADDITION: extent helper.
    # -------------------------------------------------------------------------
    # Backfill parent selection needs to know the random/pruning box extent so it
    # can avoid selecting edge-of-box Gaussians.
    #
    # This intentionally separates "backfill_extent" from "prune_outside_extent":
    #   - prune_outside_extent controls hard outside-box pruning.
    #   - backfill_extent controls candidate selection for min-Gaussian backfill.
    #
    # If you pass random_scale as backfill_extent from Splatfacto, the backfill
    # logic can avoid random-box edges even if prune_outside_random_scale_box is
    # disabled.
    # -------------------------------------------------------------------------
    def _effective_backfill_extent(self) -> Optional[float]:
        if self.backfill_extent is not None:
            return float(self.backfill_extent)
        if self.prune_outside_extent is not None:
            return float(self.prune_outside_extent)
        return None

    # -------------------------------------------------------------------------
    # MIN-GAUSSIANS ADDITION: estimate supported object cloud.
    # -------------------------------------------------------------------------
    # This replaces the mesh-aware notion of "inside the mesh" with an inferred
    # object-support prior. High-support, non-transparent Gaussians define a
    # weighted center and robust radius. Backfill parents can then be restricted
    # to the region around this supported cloud.
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _supported_cloud_geometry(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        support_score: torch.Tensor,
        opac: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Estimate a mesh-free object core from supported Gaussian locations."""
        means = params["means"].detach()

        support_mask = support_score >= self.min_backfill_support_score
        support_mask = support_mask & (opac >= self.min_backfill_opacity)

        if int(support_mask.sum().item()) < int(self.backfill_min_cloud_points):
            return None, None

        support_means = means[support_mask]
        weights = (support_score[support_mask] * opac[support_mask]).clamp_min(1e-6)

        center = (support_means * weights[:, None]).sum(dim=0) / weights.sum().clamp_min(1e-6)

        dists = torch.linalg.norm(support_means - center[None, :], dim=-1)
        radius = torch.quantile(dists, float(self.backfill_cloud_quantile)).clamp_min(1e-6)

        return center, radius

    # -------------------------------------------------------------------------
    # MIN-GAUSSIANS ADDITION: localized candidate mask.
    # -------------------------------------------------------------------------
    # This builds the parent pool used for backfill. It deliberately excludes:
    #   - Gaussians already marked for pruning,
    #   - weakly supported Gaussians,
    #   - nearly transparent Gaussians,
    #   - Gaussians near the random/pruning box edge,
    #   - Gaussians far from the robust supported cloud.
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _support_backfill_candidate_mask(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        prune_mask: torch.Tensor,
        support_score: torch.Tensor,
        opac: torch.Tensor,
    ) -> torch.Tensor:
        """Build a localized candidate pool for min-Gaussian backfilling."""
        means = params["means"].detach()

        candidate_mask = ~prune_mask
        candidate_mask &= support_score >= self.min_backfill_support_score
        candidate_mask &= opac >= self.min_backfill_opacity

        # Reject candidates near the random/pruning box edge.
        extent = self._effective_backfill_extent()
        if extent is not None and self.backfill_edge_margin_frac > 0.0:
            half_extent = 0.5 * float(extent)
            core_half_extent = half_extent * (1.0 - float(self.backfill_edge_margin_frac))
            inside_core_extent = (means.abs() <= core_half_extent).all(dim=-1)
            candidate_mask &= inside_core_extent

        # Localize to the supported Gaussian cloud.
        if self.backfill_use_supported_cloud:
            center, radius = self._supported_cloud_geometry(params, support_score, opac)
            if center is not None and radius is not None:
                dists = torch.linalg.norm(means - center[None, :], dim=-1)
                candidate_mask &= dists <= float(self.backfill_cloud_radius_scale) * radius

        return candidate_mask

    # -------------------------------------------------------------------------
    # MIN-GAUSSIANS ADDITION: choose backfill parents.
    # -------------------------------------------------------------------------
    # Parents are selected from the localized candidate pool. The score prefers:
    #   - high support score,
    #   - high opacity,
    #   - high persistent count support,
    #   - high persistent gradient support,
    #   - small distance from supported-cloud center,
    #   - small distance from random/pruning box edge.
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _select_support_backfill_parents(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        candidate_mask: torch.Tensor,
        support_score: torch.Tensor,
        opac: torch.Tensor,
        n_needed: int,
    ) -> torch.Tensor:
        """Select parent Gaussians for min-Gaussian backfill."""
        if n_needed <= 0:
            return torch.empty(0, dtype=torch.long, device=candidate_mask.device)

        if not torch.any(candidate_mask):
            return torch.empty(0, dtype=torch.long, device=candidate_mask.device)

        means = params["means"].detach()
        pool = torch.where(candidate_mask)[0]

        count = state.get("support_count_ema", None)
        grad = state.get("support_grad_ema", None)

        if isinstance(count, torch.Tensor):
            count = self._align_length(count, means.shape[0], fill_value=0.0)
            count_n = self._safe_normalize(count)
        else:
            count_n = torch.zeros_like(support_score)

        if isinstance(grad, torch.Tensor):
            grad = self._align_length(grad, means.shape[0], fill_value=0.0)
            grad_n = self._safe_normalize(grad)
        else:
            grad_n = torch.zeros_like(support_score)

        # Centrality penalty relative to the supported cloud.
        dist_penalty = torch.zeros_like(support_score)
        center, radius = self._supported_cloud_geometry(params, support_score, opac)
        if center is not None and radius is not None:
            denom = (float(self.backfill_cloud_radius_scale) * radius).clamp_min(1e-6)
            dists = torch.linalg.norm(means - center[None, :], dim=-1)
            dist_penalty = (dists / denom).clamp(0.0, 1.0)

        scores = (
            1.5 * support_score[pool]
            + 0.75 * opac[pool]
            + 0.25 * count_n[pool]
            + 0.25 * grad_n[pool]
            - 0.5 * dist_penalty[pool]
        ).clamp_min(1e-6)

        # Prefer top unique parents first.
        unique_take = min(int(n_needed), int(pool.numel()))
        selected = torch.empty(0, dtype=torch.long, device=pool.device)

        if unique_take > 0:
            selected = pool[torch.topk(scores, k=unique_take, sorted=False).indices]

        remaining = int(n_needed) - int(selected.numel())
        if remaining <= 0:
            return selected

        # If more Gaussians are needed than unique candidates, sample with replacement.
        sampled_local = torch.multinomial(scores, remaining, replacement=True)
        return torch.cat([selected, pool[sampled_local]], dim=0)

    # -------------------------------------------------------------------------
    # MIN-GAUSSIANS ADDITION: backfill implementation.
    # -------------------------------------------------------------------------
    # If pruning would leave fewer than min_gaussians, duplicate selected parents
    # from the localized object-core candidate pool before removing the marked
    # Gaussians.
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _backfill_min_gaussians(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        prune_mask: torch.Tensor,
        weak_prune: torch.Tensor,
        big_prune: torch.Tensor,
        outside_prune: torch.Tensor,
        support_score: torch.Tensor,
        opac: torch.Tensor,
    ) -> int:
        """Backfill if pruning would drop the model below min_gaussians."""
        state["support_num_backfilled"] = 0
        state["support_backfill_candidate_count"] = 0
        state["support_backfill_needed"] = 0

        if self.min_gaussians <= 0:
            return 0

        if step < self.min_gaussians_start_step:
            return 0

        any_pruned = bool(torch.any(prune_mask))
        weak_pruned = bool(torch.any(prune_mask & weak_prune))
        big_pruned = bool(torch.any(prune_mask & big_prune))
        outside_pruned = bool(torch.any(prune_mask & outside_prune))

        if self.min_gaussians_mode == "weak_only":
            should_backfill = weak_pruned
        elif self.min_gaussians_mode == "weak_or_big":
            should_backfill = weak_pruned or big_pruned
        elif self.min_gaussians_mode == "always":
            should_backfill = any_pruned
        else:
            raise ValueError(f"Unknown min_gaussians_mode: {self.min_gaussians_mode}")

        if not should_backfill:
            return 0

        # If outside pruning is the only reason and the mode is not "always",
        # do not refill the box just because out-of-extent junk was removed.
        if outside_pruned and not (weak_pruned or big_pruned) and self.min_gaussians_mode != "always":
            return 0

        n_survivors = int(prune_mask.numel() - prune_mask.sum().item())
        n_needed = int(self.min_gaussians - n_survivors)
        state["support_backfill_needed"] = max(n_needed, 0)

        if n_needed <= 0:
            return 0

        candidate_mask = self._support_backfill_candidate_mask(
            params=params,
            prune_mask=prune_mask,
            support_score=support_score,
            opac=opac,
        )
        state["support_backfill_candidate_count"] = int(candidate_mask.sum().item())

        selected = self._select_support_backfill_parents(
            params=params,
            state=state,
            candidate_mask=candidate_mask,
            support_score=support_score,
            opac=opac,
            n_needed=n_needed,
        )

        if selected.numel() == 0:
            return 0

        duplicate_selected(params=params, optimizers=optimizers, state=state, sel=selected)

        n_backfilled = int(selected.numel())
        state["support_num_backfilled"] = n_backfilled
        state["support_last_backfill_step"] = int(step)

        return n_backfilled

    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> Tuple[int, int]:
        """DefaultStrategy growth with optional support-eased densification."""
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        support_eased_mask = torch.zeros_like(grads, dtype=torch.bool)
        grow_thresh = torch.full_like(grads, self.grow_grad2d)
        if (
            step >= self._support_ready_step()
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
        """Support-aware pruning with optional support-localized min-Gaussian backfill."""
        opac = torch.sigmoid(params["opacities"].flatten())
        n_gaussian = opac.shape[0]

        effective_prune_opa = float(self.prune_opa)
        effective_support_score_thresh = float(self.support_score_thresh)
        if self._in_conservative_culling(step):
            scale = max(float(self.support_warmstart_conservative_cull_alpha_scale), 0.0)
            effective_prune_opa *= scale
        effective_support_score_thresh *= self._warmstart_decay_support_score_scale(step)

        low_opacity = opac < effective_prune_opa

        if step < self._support_ready_step():
            # Early training can contain many random Gaussians with accidental
            # visibility. During warmup, keep default weak pruning.
            support_score = torch.zeros(n_gaussian, device=opac.device)
            state["support_score"] = support_score.detach()
            support_protected = torch.zeros_like(low_opacity)
            weak_prune = low_opacity
        else:
            support_score = self._compute_support_score(params, state)
            state["support_score"] = support_score.detach()

            support_protected = support_score >= effective_support_score_thresh
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

            if self.support_protect_big and step >= self._support_ready_step():
                # Optional. Usually leave this False at first.
                big_prune = big_prune & ~support_protected

        # Keep reason buckets exclusive so they add up cleanly to total_pruned.
        outside_prune = outside_extent_mask
        weak_only_prune = weak_prune & ~outside_prune
        big_only_prune = big_prune & ~outside_prune & ~weak_prune
        is_prune = outside_prune | weak_only_prune | big_only_prune

        # ---------------------------------------------------------------------
        # MIN-GAUSSIANS ADDITION: backfill before remove().
        # ---------------------------------------------------------------------
        # Backfilling must happen before remove() because the parents still exist.
        # duplicate_selected() appends new Gaussians to params/state. Because
        # is_prune was computed before appending, it must be padded with False
        # so newly backfilled Gaussians are not immediately removed.
        # ---------------------------------------------------------------------
        n_backfill = self._backfill_min_gaussians(
            params=params,
            optimizers=optimizers,
            state=state,
            step=step,
            prune_mask=is_prune,
            weak_prune=weak_only_prune,
            big_prune=big_only_prune,
            outside_prune=outside_prune,
            support_score=support_score,
            opac=opac,
        )
        if n_backfill > 0:
            is_prune = self._align_length(is_prune, len(params["means"]), fill_value=False)

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
                f"backfilled={state['support_num_backfilled']}, "
                f"backfill_needed={state['support_backfill_needed']}, "
                f"backfill_candidates={state['support_backfill_candidate_count']}, "
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
