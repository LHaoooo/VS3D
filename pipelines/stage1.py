"""Stage 1 occupancy editing with RASI and partial mean guidance (PMG)."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import VS3DConfig

from .base import FlowEditSamplerBase, coords_to_voxel, ply_to_coords, resolve_device


ConditionPack = Dict[str, Any]
PMGStatisticsCollector = Callable[[Dict[str, torch.Tensor]], None]


def resolve_rasi_probe_weights(
    variant: str,
    cfg_src: float,
    cfg_tgt: float,
) -> Tuple[float, float]:
    """Return ``(target_probe_cfg, source_probe_cfg)`` for a RASI ablation."""
    normalized_variant = str(variant).upper()
    if normalized_variant == "A1":
        return cfg_tgt, cfg_src
    if normalized_variant == "A0":
        return cfg_src, cfg_src
    raise ValueError(f"RASI probe variant must be 'A1' or 'A0', got {variant!r}")


def build_rasi_condition_pair(
    cond_src: ConditionPack,
    cond_tgt: ConditionPack,
    phi_t: torch.Tensor,
) -> Tuple[ConditionPack, ConditionPack]:
    """Build source/target packs sharing the exact cached RASI embedding."""
    return (
        {**cond_src, "neg_cond": phi_t},
        {**cond_tgt, "neg_cond": phi_t},
    )


def partial_mean_guidance(
    v_deltas: Sequence[torch.Tensor],
    partial_samples: int,
    weight: float,
) -> torch.Tensor:
    """Apply ``mu_S + weight * (mu_S - mu_L)`` to velocity deltas."""
    _validate_pmg_parameters(len(v_deltas), partial_samples, weight)
    full = torch.stack(v_deltas).mean(0)
    partial = torch.stack(v_deltas[:partial_samples]).mean(0)
    return full + weight * (full - partial)


def _validate_pmg_parameters(
    num_samples: int,
    partial_samples: int,
    weight: float,
) -> None:
    if num_samples < 2:
        raise ValueError("pmg_num_samples must be at least 2")
    if partial_samples <= 0 or partial_samples >= num_samples:
        raise ValueError(
            "pmg_partial_samples must satisfy "
            "1 <= pmg_partial_samples < pmg_num_samples"
        )
    if not math.isfinite(float(weight)):
        raise ValueError("pmg_weight must be finite")


def _validate_pmg_epsilon(eps: float) -> float:
    resolved = float(eps)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError("PMG epsilon must be finite and positive")
    return resolved


def _validate_pmg_amplification_weight(weight: float) -> None:
    if float(weight) < 0.0:
        raise ValueError("gated PMG weight must be non-negative")


def _stack_dense_velocity_deltas(
    v_deltas: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.dtype]:
    if not v_deltas:
        raise ValueError("PMG requires at least one velocity delta")
    reference = v_deltas[0]
    if reference.ndim != 5:
        raise ValueError("PMG velocity deltas must have shape [B, C, R, R, R]")
    if not reference.is_floating_point():
        raise ValueError("PMG velocity deltas must be floating-point tensors")
    for delta in v_deltas[1:]:
        if delta.shape != reference.shape:
            raise ValueError("PMG velocity deltas must have identical shapes")
        if delta.dtype != reference.dtype or delta.device != reference.device:
            raise ValueError("PMG velocity deltas must share dtype and device")
    samples = torch.stack(tuple(v_deltas), dim=0).float()
    if not bool(torch.isfinite(samples).all().item()):
        raise ValueError("PMG velocity deltas must contain only finite values")
    return samples, reference.dtype


def _complementary_cosine(
    samples: torch.Tensor,
    partial_samples: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    partial_mean = samples[:partial_samples].mean(dim=0)
    complement_mean = samples[partial_samples:].mean(dim=0)
    partial_norm_sq = partial_mean.square().sum(dim=1, keepdim=True)
    complement_norm_sq = complement_mean.square().sum(dim=1, keepdim=True)
    denominator = (
        (partial_norm_sq + eps) * (complement_norm_sq + eps)
    ).sqrt()
    cosine = (
        (partial_mean * complement_mean).sum(dim=1, keepdim=True) / denominator
    ).clamp(-1.0, 1.0)
    valid = (
        torch.isfinite(cosine)
        & torch.isfinite(partial_norm_sq)
        & torch.isfinite(complement_norm_sq)
        & (partial_norm_sq > eps)
        & (complement_norm_sq > eps)
    )
    return cosine, valid


def consistency_gated_guidance(
    v_deltas: Sequence[torch.Tensor],
    partial_samples: int,
    weight: float,
    tau: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Amplify the full mean where complementary sample means agree."""
    _validate_pmg_parameters(len(v_deltas), partial_samples, weight)
    _validate_pmg_amplification_weight(weight)
    epsilon = _validate_pmg_epsilon(eps)
    threshold = float(tau)
    if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
        raise ValueError("pmg_consistency_tau must be finite and in [-1, 1]")

    samples, output_dtype = _stack_dense_velocity_deltas(v_deltas)
    full_mean = samples.mean(dim=0)
    cosine, valid = _complementary_cosine(
        samples,
        partial_samples,
        epsilon,
    )
    gate = ((cosine - threshold) / (1.0 - threshold + epsilon)).clamp(0.0, 1.0)
    gate = torch.where(valid, gate, torch.zeros_like(gate))
    return ((1.0 + float(weight) * gate) * full_mean).to(dtype=output_dtype)


def _fold_mean_and_se_energy(
    samples: torch.Tensor, indices: Sequence[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    fold = samples[list(indices)]
    sample_count = int(fold.shape[0])
    if sample_count < 2:
        raise ValueError("Cross-Fit PMG folds must contain at least two samples")
    mean = fold.mean(dim=0)
    residual = fold - mean.unsqueeze(0)
    energy = residual.square().sum(dim=2, keepdim=True).sum(dim=0)
    return mean, energy / float(sample_count * (sample_count - 1))


def _snr_gate(
    mean: torch.Tensor,
    se_energy: torch.Tensor,
    kappa: float,
    eps: float,
) -> torch.Tensor:
    snr, valid = _fold_snr(mean, se_energy, eps)
    gate = (1.0 - float(kappa) / (snr + eps)).clamp(0.0, 1.0)
    return torch.where(valid, gate, torch.zeros_like(gate))


def _fold_snr(
    mean: torch.Tensor,
    se_energy: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mean_norm_sq = mean.square().sum(dim=1, keepdim=True)
    snr = mean_norm_sq / (se_energy + eps)
    valid = (
        torch.isfinite(mean_norm_sq)
        & torch.isfinite(se_energy)
        & torch.isfinite(snr)
        & (mean_norm_sq > eps)
    )
    return snr, valid


def pmg_calibration_statistics(
    v_deltas: Sequence[torch.Tensor],
    partial_samples: int,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Return the valid FP32 scores consumed by the gated PMG formulas."""
    _validate_pmg_parameters(len(v_deltas), partial_samples, 0.0)
    if len(v_deltas) != 5 or int(partial_samples) != 2:
        raise ValueError(
            "PMG calibration requires pmg_num_samples=5 and pmg_partial_samples=2"
        )
    epsilon = _validate_pmg_epsilon(eps)
    samples, _ = _stack_dense_velocity_deltas(v_deltas)
    cosine, cosine_valid = _complementary_cosine(
        samples,
        int(partial_samples),
        epsilon,
    )

    fold_2_scores: List[torch.Tensor] = []
    fold_3_scores: List[torch.Tensor] = []
    all_indices = tuple(range(len(v_deltas)))
    for partial_indices in combinations(all_indices, int(partial_samples)):
        partial_set = set(partial_indices)
        complement_indices = tuple(
            index for index in all_indices if index not in partial_set
        )
        partial_mean, partial_energy = _fold_mean_and_se_energy(
            samples, partial_indices
        )
        complement_mean, complement_energy = _fold_mean_and_se_energy(
            samples, complement_indices
        )
        partial_snr, partial_valid = _fold_snr(
            partial_mean, partial_energy, epsilon
        )
        complement_snr, complement_valid = _fold_snr(
            complement_mean, complement_energy, epsilon
        )
        fold_2_scores.append(partial_snr[partial_valid].reshape(-1))
        fold_3_scores.append(complement_snr[complement_valid].reshape(-1))

    return {
        "consistency": cosine[cosine_valid].reshape(-1),
        "crossfit_fold_2": torch.cat(fold_2_scores, dim=0),
        "crossfit_fold_3": torch.cat(fold_3_scores, dim=0),
    }


def crossfit_snr_guidance(
    v_deltas: Sequence[torch.Tensor],
    partial_samples: int,
    weight: float,
    kappa_partial: float,
    kappa_complement: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Average all 2/3 splits while applying each opposite fold's SNR gate."""
    _validate_pmg_parameters(len(v_deltas), partial_samples, weight)
    _validate_pmg_amplification_weight(weight)
    if len(v_deltas) != 5 or int(partial_samples) != 2:
        raise ValueError("Cross-Fit PMG requires pmg_num_samples=5 and pmg_partial_samples=2")
    epsilon = _validate_pmg_epsilon(eps)
    resolved_kappas = (float(kappa_partial), float(kappa_complement))
    if any(not math.isfinite(value) or value < 0.0 for value in resolved_kappas):
        raise ValueError("Cross-Fit PMG kappas must be finite and non-negative")

    samples, output_dtype = _stack_dense_velocity_deltas(v_deltas)
    all_indices = tuple(range(len(v_deltas)))
    outputs: List[torch.Tensor] = []
    for partial_indices in combinations(all_indices, int(partial_samples)):
        partial_set = set(partial_indices)
        complement_indices = tuple(index for index in all_indices if index not in partial_set)
        partial_mean, partial_energy = _fold_mean_and_se_energy(
            samples, partial_indices
        )
        complement_mean, complement_energy = _fold_mean_and_se_energy(
            samples, complement_indices
        )
        partial_gate = _snr_gate(
            partial_mean, partial_energy, resolved_kappas[0], epsilon
        )
        complement_gate = _snr_gate(
            complement_mean, complement_energy, resolved_kappas[1], epsilon
        )
        outputs.append(
            (2.0 / 5.0)
            * (1.0 + float(weight) * complement_gate)
            * partial_mean
            + (3.0 / 5.0)
            * (1.0 + float(weight) * partial_gate)
            * complement_mean
        )
    return torch.stack(outputs, dim=0).mean(dim=0).to(dtype=output_dtype)


def dispatch_pmg_guidance(
    v_deltas: Sequence[torch.Tensor],
    mode: str,
    partial_samples: int,
    weight: float,
    *,
    consistency_tau: Optional[float] = None,
    crossfit_kappa_partial: Optional[float] = None,
    crossfit_kappa_complement: Optional[float] = None,
) -> torch.Tensor:
    """Select one PMG formula without changing the sampled velocity deltas."""
    normalized_mode = str(mode).lower()
    if normalized_mode == "original":
        return partial_mean_guidance(v_deltas, partial_samples, weight)
    if normalized_mode == "consistency":
        if consistency_tau is None:
            raise ValueError("pmg_consistency_tau is required for consistency PMG")
        return consistency_gated_guidance(
            v_deltas, partial_samples, weight, consistency_tau
        )
    if normalized_mode == "crossfit":
        if crossfit_kappa_partial is None or crossfit_kappa_complement is None:
            raise ValueError("both Cross-Fit PMG kappas are required")
        return crossfit_snr_guidance(
            v_deltas,
            partial_samples,
            weight,
            crossfit_kappa_partial,
            crossfit_kappa_complement,
        )
    raise ValueError(
        "pmg_mode must be one of 'original', 'consistency', or 'crossfit'"
    )


def _validate_pmg_mode_parameters(
    mode: str,
    num_samples: int,
    partial_samples: int,
    weight: float,
    consistency_tau: Optional[float],
    crossfit_kappa_2: Optional[float],
    crossfit_kappa_3: Optional[float],
) -> str:
    _validate_pmg_parameters(num_samples, partial_samples, weight)
    normalized_mode = str(mode).lower()
    if normalized_mode == "original":
        return normalized_mode
    if float(weight) < 0.0:
        raise ValueError("gated PMG weight must be non-negative")
    if normalized_mode == "consistency":
        if consistency_tau is None:
            raise ValueError("pmg_consistency_tau is required for consistency PMG")
        threshold = float(consistency_tau)
        if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
            raise ValueError("pmg_consistency_tau must be finite and in [-1, 1]")
        return normalized_mode
    if normalized_mode == "crossfit":
        if int(num_samples) != 5 or int(partial_samples) != 2:
            raise ValueError(
                "Cross-Fit PMG requires pmg_num_samples=5 and "
                "pmg_partial_samples=2"
            )
        if crossfit_kappa_2 is None or crossfit_kappa_3 is None:
            raise ValueError(
                "pmg_crossfit_kappa_2 and pmg_crossfit_kappa_3 are required"
            )
        kappas = (float(crossfit_kappa_2), float(crossfit_kappa_3))
        if any(not math.isfinite(value) or value < 0.0 for value in kappas):
            raise ValueError("Cross-Fit PMG kappas must be finite and non-negative")
        return normalized_mode
    raise ValueError(
        "pmg_mode must be one of 'original', 'consistency', or 'crossfit'"
    )


def _downsample_occupancy(
    occupancy: torch.Tensor,
    ss_out_res: int,
) -> torch.Tensor:
    """Downsample a cubic occupancy grid after validating the requested size."""
    output_resolution = int(ss_out_res)
    input_shape = tuple(int(size) for size in occupancy.shape[2:])
    if output_resolution <= 0:
        raise ValueError("ss_out_res must be positive")
    if len(input_shape) != 3 or len(set(input_shape)) != 1:
        raise ValueError("occupancy must have equal spatial dimensions")

    input_resolution = input_shape[0]
    if output_resolution > input_resolution:
        raise ValueError("ss_out_res cannot exceed the occupancy resolution")
    if input_resolution % output_resolution != 0:
        raise ValueError("ss_out_res must evenly divide the occupancy resolution")
    if output_resolution == input_resolution:
        return occupancy

    ratio = input_resolution // output_resolution
    return F.max_pool3d(occupancy.float(), ratio, ratio, 0) > 0.5


class RASIFlowEditSampler(FlowEditSamplerBase):
    """FlowEdit sampler using RASI reconstruction and optional PMG."""

    def optimize_rasi(
        self,
        model,
        x_src: torch.Tensor,
        cond_src: ConditionPack,
        cfg_src: float,
        cfg_tgt: float,
        steps: int,
        n_max: int,
        n_min: int,
        probe_variant: str,
        rescale_t: float = 5.0,
        cfg_interval: Tuple[float, float] = (0.6, 1.0),
        num_inner_steps: int = 6,
        lr: float = 1e-2,
        early_stop_eps: float = 1e-5,
        desc: str = "Phase 1: RASI optimize phi",
    ) -> List[torch.Tensor]:
        """Optimize a per-step shared negative embedding for reconstruction.

        Both reconstruction probes use the source image condition. A1 applies
        the editing CFG strengths to the target/source probes respectively;
        A0 retains the legacy source-CFG ablation on both probes.
        """
        probe_cfg_tgt, probe_cfg_src = resolve_rasi_probe_weights(
            probe_variant,
            cfg_src=cfg_src,
            cfg_tgt=cfg_tgt,
        )
        n_max = min(int(n_max), int(steps))
        n_min = max(0, min(int(n_min), n_max))

        t_seq = self.get_timesteps(int(steps), rescale_t=rescale_t)
        t_pairs = [
            (float(t_seq[i]), float(t_seq[i + 1])) for i in range(int(steps))
        ]

        phi_init = cond_src["neg_cond"].detach().clone()
        cached_phi: List[torch.Tensor] = []
        zt_edit = x_src.clone()

        model_params = list(model.parameters())
        original_requires_grad = [parameter.requires_grad for parameter in model_params]
        for parameter in model_params:
            parameter.requires_grad_(False)

        try:
            active_idx = 0
            for i, (t_curr, t_prev) in enumerate(tqdm(t_pairs, desc=desc)):
                if steps - i > n_max:
                    continue
                if steps - i <= n_min:
                    cached_phi.append(phi_init.detach().clone().cpu())
                    continue

                in_cfg = cfg_interval[0] <= t_curr <= cfg_interval[1]
                if not in_cfg:
                    cached_phi.append(phi_init.detach().clone().cpu())
                    with torch.no_grad():
                        fwd_noise = torch.randn_like(x_src)
                        zt_src = (1.0 - t_curr) * x_src + t_curr * fwd_noise
                        zt_tgt = zt_edit + zt_src - x_src
                        c_src = self.infer(model, zt_src, t_curr, cond_src["cond"])
                        c_tgt = self.infer(model, zt_tgt, t_curr, cond_src["cond"])
                        zt_edit = zt_edit + (t_prev - t_curr) * (c_tgt - c_src)
                    active_idx += 1
                    continue

                phi_t = phi_init.clone().detach().requires_grad_(True)
                lr_step = lr * (1.0 - active_idx / max(1, n_max - n_min))
                optimizer = torch.optim.Adam([phi_t], lr=lr_step)

                for _ in range(num_inner_steps):
                    optimizer.zero_grad(set_to_none=True)
                    fwd_noise = torch.randn_like(x_src)
                    zt_src = (1.0 - t_curr) * x_src + t_curr * fwd_noise
                    zt_tgt = zt_edit + zt_src - x_src

                    with torch.no_grad():
                        c_src = self.infer(model, zt_src, t_curr, cond_src["cond"])
                        c_tgt = self.infer(model, zt_tgt, t_curr, cond_src["cond"])
                    u_src = self.infer(model, zt_src, t_curr, phi_t)
                    u_tgt = self.infer(model, zt_tgt, t_curr, phi_t)

                    v_src = self.cfg(c_src, u_src, probe_cfg_src)
                    v_tgt = self.cfg(c_tgt, u_tgt, probe_cfg_tgt)
                    prediction = zt_edit + (t_prev - t_curr) * (v_tgt - v_src)
                    loss = F.mse_loss(prediction, x_src)
                    loss.backward()
                    optimizer.step()
                    if loss.item() < early_stop_eps:
                        break

                cached_phi.append(phi_t.detach().clone().cpu())

                with torch.no_grad():
                    fwd_noise = torch.randn_like(x_src)
                    zt_src = (1.0 - t_curr) * x_src + t_curr * fwd_noise
                    zt_tgt = zt_edit + zt_src - x_src
                    c_src = self.infer(model, zt_src, t_curr, cond_src["cond"])
                    c_tgt = self.infer(model, zt_tgt, t_curr, cond_src["cond"])
                    u_src = self.infer(model, zt_src, t_curr, phi_t)
                    u_tgt = self.infer(model, zt_tgt, t_curr, phi_t)
                    v_src = self.cfg(c_src, u_src, probe_cfg_src)
                    v_tgt = self.cfg(c_tgt, u_tgt, probe_cfg_tgt)
                    zt_edit = zt_edit + (t_prev - t_curr) * (v_tgt - v_src)

                del optimizer, phi_t
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                active_idx += 1

            return cached_phi
        finally:
            for parameter, requires_grad in zip(
                model_params, original_requires_grad
            ):
                parameter.requires_grad_(requires_grad)

    @torch.no_grad()
    def sample_flowedit(
        self,
        model,
        x_src: torch.Tensor,
        cond_src: ConditionPack,
        cond_tgt: ConditionPack,
        cached_phi: Optional[Sequence[torch.Tensor]],
        cfg_src: float,
        cfg_tgt: float,
        steps: int,
        n_max: int,
        n_min: int,
        pmg_enabled: bool,
        pmg_num_samples: int,
        pmg_partial_samples: int,
        pmg_weight: float,
        pmg_mode: str = "original",
        pmg_consistency_tau: Optional[float] = None,
        pmg_crossfit_kappa_2: Optional[float] = None,
        pmg_crossfit_kappa_3: Optional[float] = None,
        pmg_statistics_collector: Optional[PMGStatisticsCollector] = None,
        rescale_t: float = 5.0,
        cfg_interval: Tuple[float, float] = (0.6, 1.0),
        desc: str = "Phase 2: FlowEdit with PMG",
    ) -> torch.Tensor:
        """Edit the occupancy latent, sharing each cached ``phi_t`` exactly."""
        n_max = min(int(n_max), int(steps))
        n_min = max(0, min(int(n_min), n_max))
        if cached_phi is not None and len(cached_phi) != n_max:
            raise ValueError(
                f"cached_phi length mismatch: expected {n_max}, "
                f"got {len(cached_phi)}"
            )

        if pmg_enabled:
            num_samples = int(pmg_num_samples)
            partial_samples = int(pmg_partial_samples)
            guidance_weight = float(pmg_weight)
            resolved_pmg_mode = _validate_pmg_mode_parameters(
                pmg_mode,
                num_samples,
                partial_samples,
                guidance_weight,
                pmg_consistency_tau,
                pmg_crossfit_kappa_2,
                pmg_crossfit_kappa_3,
            )
        else:
            num_samples = 1
            partial_samples = 0
            guidance_weight = 0.0
            resolved_pmg_mode = "original"
        if pmg_statistics_collector is not None and (
            not pmg_enabled or num_samples != 5 or partial_samples != 2
        ):
            raise ValueError(
                "PMG statistics collection requires enabled PMG with "
                "pmg_num_samples=5 and pmg_partial_samples=2"
            )

        t_seq = self.get_timesteps(int(steps), rescale_t=rescale_t)
        t_pairs = [
            (float(t_seq[i]), float(t_seq[i + 1])) for i in range(int(steps))
        ]

        zt_edit = x_src.clone()
        xt_tgt: Optional[torch.Tensor] = None
        cached_idx = 0

        for i, (t_curr, t_prev) in enumerate(tqdm(t_pairs, desc=desc)):
            if steps - i > n_max:
                continue

            if steps - i > n_min:
                if cached_phi is None:
                    cond_src_step = cond_src
                    cond_tgt_step = cond_tgt
                else:
                    phi_t = cached_phi[cached_idx].to(x_src.device)
                    cond_src_step, cond_tgt_step = build_rasi_condition_pair(
                        cond_src,
                        cond_tgt,
                        phi_t,
                    )
                    cached_idx += 1

                v_deltas: List[torch.Tensor] = []
                for _ in range(num_samples):
                    fwd_noise = torch.randn_like(x_src)
                    zt_src = (1.0 - t_curr) * x_src + t_curr * fwd_noise
                    zt_tgt = zt_edit + zt_src - x_src

                    c_src, u_src = self.infer_cond_uncond(
                        model, zt_src, t_curr, cond_src_step, cfg_interval
                    )
                    c_tgt, u_tgt = self.infer_cond_uncond(
                        model, zt_tgt, t_curr, cond_tgt_step, cfg_interval
                    )
                    v_src = self.cfg(c_src, u_src, cfg_src)
                    v_tgt = self.cfg(c_tgt, u_tgt, cfg_tgt)
                    v_deltas.append(v_tgt - v_src)

                if pmg_statistics_collector is not None:
                    pmg_statistics_collector(
                        pmg_calibration_statistics(
                            v_deltas,
                            partial_samples=partial_samples,
                        )
                    )
                if pmg_enabled:
                    update = dispatch_pmg_guidance(
                        v_deltas,
                        mode=resolved_pmg_mode,
                        partial_samples=partial_samples,
                        weight=guidance_weight,
                        consistency_tau=pmg_consistency_tau,
                        crossfit_kappa_partial=pmg_crossfit_kappa_2,
                        crossfit_kappa_complement=pmg_crossfit_kappa_3,
                    )
                else:
                    update = v_deltas[0]
                zt_edit = zt_edit + (t_prev - t_curr) * update
                del cond_src_step, cond_tgt_step, v_deltas
                if cached_phi is not None:
                    del phi_t
            else:
                if cached_phi is None:
                    cond_tgt_step = cond_tgt
                else:
                    phi_t = cached_phi[cached_idx].to(x_src.device)
                    cond_tgt_step = {**cond_tgt, "neg_cond": phi_t}
                    cached_idx += 1

                if xt_tgt is None:
                    fwd_noise = torch.randn_like(x_src)
                    xt_src = (1.0 - t_curr) * x_src + t_curr * fwd_noise
                    xt_tgt = zt_edit + xt_src - x_src

                c_tgt, u_tgt = self.infer_cond_uncond(
                    model, xt_tgt, t_curr, cond_tgt_step, cfg_interval
                )
                v_tgt = self.cfg(c_tgt, u_tgt, cfg_tgt)
                xt_tgt = xt_tgt + (t_prev - t_curr) * v_tgt
                del cond_tgt_step
                if cached_phi is not None:
                    del phi_t

        return xt_tgt if n_min > 0 and xt_tgt is not None else zt_edit


def stage1_rasi_pmg(
    pipeline,
    voxel_ply_path: str,
    cond_src: ConditionPack,
    cond_tgt: ConditionPack,
    cfg_src: float,
    cfg_tgt: float,
    n_max: int,
    n_min: int,
    *,
    config: Optional[VS3DConfig] = None,
    steps: Optional[int] = None,
    rasi_enabled: Optional[bool] = None,
    rasi_probe_variant: Optional[str] = None,
    pmg_enabled: Optional[bool] = None,
    pmg_num_samples: Optional[int] = None,
    pmg_partial_samples: Optional[int] = None,
    pmg_weight: Optional[float] = None,
    pmg_mode: Optional[str] = None,
    pmg_consistency_tau: Optional[float] = None,
    pmg_crossfit_kappa_2: Optional[float] = None,
    pmg_crossfit_kappa_3: Optional[float] = None,
    pmg_statistics_collector: Optional[PMGStatisticsCollector] = None,
    cfg_interval: Tuple[float, float] = (0.6, 1.0),
    ss_out_res: Optional[int] = None,
    voxel_in_res: int = 64,
    rasi_inner_steps: int = 6,
    rasi_lr: float = 1e-2,
    rasi_early_stop_eps: float = 1e-5,
    occ_threshold_logit: float = 0.0,
) -> torch.Tensor:
    """Run Stage 1 with defaults sourced from :class:`VS3DConfig`."""
    runtime_config = config if config is not None else VS3DConfig()
    steps = runtime_config.stage1_steps if steps is None else int(steps)
    rasi_enabled = (
        runtime_config.rasi_enabled if rasi_enabled is None else bool(rasi_enabled)
    )
    probe_variant = (
        runtime_config.rasi_probe_variant
        if rasi_probe_variant is None
        else rasi_probe_variant
    )
    pmg_enabled = (
        runtime_config.pmg_enabled if pmg_enabled is None else bool(pmg_enabled)
    )
    pmg_num_samples = (
        runtime_config.pmg_num_samples
        if pmg_num_samples is None
        else int(pmg_num_samples)
    )
    pmg_partial_samples = (
        runtime_config.pmg_partial_samples
        if pmg_partial_samples is None
        else int(pmg_partial_samples)
    )
    pmg_weight = (
        runtime_config.pmg_weight if pmg_weight is None else float(pmg_weight)
    )
    pmg_mode = runtime_config.pmg_mode if pmg_mode is None else str(pmg_mode)
    pmg_consistency_tau = (
        runtime_config.pmg_consistency_tau
        if pmg_consistency_tau is None
        else float(pmg_consistency_tau)
    )
    pmg_crossfit_kappa_2 = (
        runtime_config.pmg_crossfit_kappa_2
        if pmg_crossfit_kappa_2 is None
        else float(pmg_crossfit_kappa_2)
    )
    pmg_crossfit_kappa_3 = (
        runtime_config.pmg_crossfit_kappa_3
        if pmg_crossfit_kappa_3 is None
        else float(pmg_crossfit_kappa_3)
    )
    if pmg_enabled:
        pmg_mode = _validate_pmg_mode_parameters(
            pmg_mode,
            pmg_num_samples,
            pmg_partial_samples,
            pmg_weight,
            pmg_consistency_tau,
            pmg_crossfit_kappa_2,
            pmg_crossfit_kappa_3,
        )

    device = resolve_device(pipeline)
    flow = pipeline.models["sparse_structure_flow_model"]
    encoder = pipeline.models["sparse_structure_encoder"]
    decoder = pipeline.models["sparse_structure_decoder"]
    low_vram = bool(getattr(pipeline, "low_vram", False))

    try:
        if low_vram:
            encoder.to(device)

        coords_xyz = ply_to_coords(
            voxel_ply_path,
            res=voxel_in_res,
            device=device,
        )
        voxel_src = coords_to_voxel(coords_xyz, res=voxel_in_res)
        with torch.no_grad():
            x_src = encoder(voxel_src).detach()
        del coords_xyz, voxel_src

        if low_vram:
            encoder.cpu()
            flow.to(device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        sampler = RASIFlowEditSampler(device=device)
        cached_phi: Optional[Sequence[torch.Tensor]]
        if rasi_enabled:
            cached_phi = sampler.optimize_rasi(
                model=flow,
                x_src=x_src,
                cond_src=cond_src,
                cfg_src=cfg_src,
                cfg_tgt=cfg_tgt,
                steps=steps,
                n_max=n_max,
                n_min=n_min,
                probe_variant=probe_variant,
                cfg_interval=cfg_interval,
                num_inner_steps=rasi_inner_steps,
                lr=rasi_lr,
                early_stop_eps=rasi_early_stop_eps,
            )
        else:
            cached_phi = None

        x_edit = sampler.sample_flowedit(
            model=flow,
            x_src=x_src,
            cond_src=cond_src,
            cond_tgt=cond_tgt,
            cached_phi=cached_phi,
            cfg_src=cfg_src,
            cfg_tgt=cfg_tgt,
            steps=steps,
            n_max=n_max,
            n_min=n_min,
            cfg_interval=cfg_interval,
            pmg_enabled=pmg_enabled,
            pmg_num_samples=pmg_num_samples,
            pmg_partial_samples=pmg_partial_samples,
            pmg_weight=pmg_weight,
            pmg_mode=pmg_mode,
            pmg_consistency_tau=pmg_consistency_tau,
            pmg_crossfit_kappa_2=pmg_crossfit_kappa_2,
            pmg_crossfit_kappa_3=pmg_crossfit_kappa_3,
            pmg_statistics_collector=pmg_statistics_collector,
        )
        cached_phi = None
        x_src = None
        sampler = None

        if low_vram:
            flow.cpu()
            decoder.to(device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        with torch.no_grad():
            decoded = decoder(x_edit)
        x_edit = None
        if decoded.dim() == 4:
            decoded = decoded[:, None]
        occupancy = decoded > float(occ_threshold_logit)
        decoded = None

        if ss_out_res is not None:
            occupancy = _downsample_occupancy(occupancy, ss_out_res)
        return torch.argwhere(occupancy)[:, [0, 2, 3, 4]].int()
    finally:
        cached_phi = None
        x_src = None
        sampler = None
        x_edit = None
        decoded = None
        if low_vram:
            encoder.cpu()
            flow.cpu()
            decoder.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
