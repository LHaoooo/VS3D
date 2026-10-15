"""Twin-aligned residual (TAR) sampling for shape and texture SLATs."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import torch

from config import VS3DConfig
from .native import sample_shape_slat_1024, sample_texture_slat_1024


def sample_stage_twins(
    sample_fn: Callable[..., Any],
    *,
    noise: torch.Tensor,
    scaffold: Any,
    source_condition: Any,
    target_condition: Any,
    sampler_kwargs: Mapping[str, Any],
) -> Tuple[Any, Any]:
    """Sample source and target twins with identical runtime inputs."""
    shared_kwargs = dict(sampler_kwargs)
    source = sample_fn(
        noise=noise,
        scaffold=scaffold,
        condition=source_condition,
        **shared_kwargs,
    )
    target = sample_fn(
        noise=noise,
        scaffold=scaffold,
        condition=target_condition,
        **shared_kwargs,
    )
    return source, target


def robust_quantile_keep_probability(
    disagreement_norm: torch.Tensor,
    *,
    quantiles: Sequence[float] = (0.05, 0.95),
) -> torch.Tensor:
    """Map small twin disagreement to one and large disagreement to zero."""
    if len(quantiles) != 2:
        raise ValueError("quantiles must contain exactly two values")
    lower, upper = (float(quantiles[0]), float(quantiles[1]))
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("quantiles must be ordered values in [0, 1]")
    if disagreement_norm.ndim != 1 or disagreement_norm.numel() == 0:
        raise ValueError("disagreement_norm must be a non-empty 1-D tensor")
    if not torch.isfinite(disagreement_norm).all():
        raise ValueError("disagreement_norm must contain only finite values")

    values = disagreement_norm.float()
    lo = torch.quantile(values, lower)
    hi = torch.quantile(values, upper)
    if torch.isclose(lo, hi, rtol=0.0, atol=1e-9):
        return torch.ones_like(disagreement_norm)
    scaled = ((values - lo) / (hi - lo)).clamp(0.0, 1.0)
    return (1.0 - scaled).to(dtype=disagreement_norm.dtype)


def exact_coordinate_intersection(
    target_coords: torch.Tensor,
    source_coords: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return index pairs whose integer (batch, x, y, z) rows match exactly."""
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    if (
        target_coords.ndim != 2
        or source_coords.ndim != 2
        or target_coords.shape[1] != 4
        or source_coords.shape[1] != 4
    ):
        raise ValueError("coordinates must have shape [N, 4]")
    if target_coords.dtype not in integer_dtypes or source_coords.dtype not in integer_dtypes:
        raise TypeError("coordinates must use integer dtypes")

    target_cpu = target_coords.detach().cpu()
    source_cpu = source_coords.detach().cpu()
    target_rows = [tuple(row.tolist()) for row in target_cpu]
    source_rows_list = [tuple(row.tolist()) for row in source_cpu]
    if len(set(target_rows)) != len(target_rows):
        raise ValueError("target coordinates must be unique; duplicate row found")
    if len(set(source_rows_list)) != len(source_rows_list):
        raise ValueError("source coordinates must be unique; duplicate row found")
    source_rows = {tuple(row.tolist()): index for index, row in enumerate(source_cpu)}
    target_indices = []
    source_indices = []
    for index, row in enumerate(target_cpu):
        source_index = source_rows.get(tuple(row.tolist()))
        if source_index is not None:
            target_indices.append(index)
            source_indices.append(source_index)
    return (
        torch.tensor(target_indices, dtype=torch.long),
        torch.tensor(source_indices, dtype=torch.long),
    )


def clip_row_norm(value: torch.Tensor, *, tau: float) -> torch.Tensor:
    """L2-clip each feature row while preserving its direction."""
    tau = float(tau)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be finite and positive")
    if value.ndim != 2:
        raise ValueError("value must have shape [N, D]")
    norm = value.norm(dim=1, keepdim=True)
    scale = torch.clamp_max(tau / norm.clamp_min(torch.finfo(value.dtype).eps), 1.0)
    return value * scale


def inject_tar_residual(
    target: torch.Tensor,
    source: torch.Tensor,
    p_keep: torch.Tensor,
    target_indices: torch.Tensor,
    source_indices: torch.Tensor,
    *,
    lam: float,
    tau: float,
    threshold: float,
) -> torch.Tensor:
    """Inject a clipped source-minus-target residual on eligible intersections."""
    lam = float(lam)
    threshold = float(threshold)
    if not math.isfinite(lam) or lam < 0.0:
        raise ValueError("lam must be finite and non-negative")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and in [0, 1]")
    if target.ndim != 2 or source.ndim != 2 or target.shape[1] != source.shape[1]:
        raise ValueError("target and source must be [N, D] tensors with equal D")
    if p_keep.ndim != 1 or p_keep.shape[0] != target.shape[0]:
        raise ValueError("p_keep must contain one value per target token")
    if target_indices.ndim != 1 or source_indices.ndim != 1:
        raise ValueError("intersection indices must be 1-D")
    if target_indices.shape != source_indices.shape:
        raise ValueError("target_indices and source_indices must have equal length")
    if target_indices.numel() == 0:
        return target.clone()

    target_indices = target_indices.to(target.device)
    source_indices = source_indices.to(source.device)
    weights = p_keep.to(device=target.device, dtype=target.dtype)[target_indices]
    eligible = weights >= threshold
    if not eligible.any():
        return target.clone()

    selected_target = target_indices[eligible]
    selected_source = source_indices[eligible.to(source_indices.device)]
    residual = source[selected_source].to(target) - target[selected_target]
    residual = clip_row_norm(residual, tau=tau)
    result = target.clone()
    result[selected_target] = (
        target[selected_target]
        + lam * weights[eligible, None] * residual
    )
    return result


def _features(value: Any) -> torch.Tensor:
    return value.feats if hasattr(value, "feats") else value


def _coords(value: Any) -> torch.Tensor:
    if not hasattr(value, "coords"):
        raise TypeError("TAR runtime values must expose integer coords")
    return value.coords


def _run_tar_stage(
    sample_fn: Callable[..., Any],
    pipeline: Any,
    *,
    noise: torch.Tensor,
    scaffold: Any,
    source_condition: Any,
    target_condition: Any,
    source_slat: Any,
    sampler_kwargs: Optional[Mapping[str, Any]],
    config: Optional[VS3DConfig],
) -> Tuple[Any, dict]:
    resolved_config = config or VS3DConfig()

    def bound_sample(**kwargs: Any) -> Any:
        return sample_fn(pipeline, **kwargs)

    source_twin, target_generated = sample_stage_twins(
        bound_sample,
        noise=noise,
        scaffold=scaffold,
        source_condition=source_condition,
        target_condition=target_condition,
        sampler_kwargs=dict(sampler_kwargs or {}),
    )
    target_features = _features(target_generated)
    twin_features = _features(source_twin).to(target_features)
    if twin_features.shape != target_features.shape:
        raise ValueError("source and target twins must have the same feature shape")
    if not torch.equal(
        _coords(source_twin).detach().cpu(),
        _coords(target_generated).detach().cpu(),
    ):
        raise ValueError("source and target twins must have identical coordinates")
    delta_norm = (target_features.float() - twin_features.float()).norm(dim=1)
    p_keep = robust_quantile_keep_probability(
        delta_norm, quantiles=resolved_config.tar_quantiles
    )
    target_indices, source_indices = exact_coordinate_intersection(
        _coords(target_generated), _coords(source_slat)
    )
    injected_features = inject_tar_residual(
        target_features,
        _features(source_slat),
        p_keep,
        target_indices,
        source_indices,
        lam=resolved_config.tar_lambda,
        tau=resolved_config.tar_clip_tau,
        threshold=resolved_config.tar_keep_threshold,
    )
    result = target_generated.replace(injected_features)
    diagnostics = {
        "p_keep": p_keep,
        "delta_norm": delta_norm,
        "target_indices": target_indices,
        "source_indices": source_indices,
        "forward_count": 2,
        "intersection_count": int(target_indices.numel()),
    }
    return result, diagnostics


def run_shape_tar(
    pipeline: Any,
    *,
    noise: torch.Tensor,
    scaffold: torch.Tensor,
    source_condition: Any,
    target_condition: Any,
    source_slat: Any,
    sampler_kwargs: Optional[Mapping[str, Any]] = None,
    config: Optional[VS3DConfig] = None,
) -> Tuple[Any, dict]:
    """Run exactly two shape forwards and inject source-encoded shape detail."""
    return _run_tar_stage(
        sample_shape_slat_1024,
        pipeline,
        noise=noise,
        scaffold=scaffold,
        source_condition=source_condition,
        target_condition=target_condition,
        source_slat=source_slat,
        sampler_kwargs=sampler_kwargs,
        config=config,
    )


def run_texture_tar(
    pipeline: Any,
    *,
    noise: torch.Tensor,
    scaffold: Any,
    source_condition: Any,
    target_condition: Any,
    source_slat: Any,
    sampler_kwargs: Optional[Mapping[str, Any]] = None,
    config: Optional[VS3DConfig] = None,
) -> Tuple[Any, dict]:
    """Run exactly two texture forwards and inject source-encoded texture detail."""
    return _run_tar_stage(
        sample_texture_slat_1024,
        pipeline,
        noise=noise,
        scaffold=scaffold,
        source_condition=source_condition,
        target_condition=target_condition,
        source_slat=source_slat,
        sampler_kwargs=sampler_kwargs,
        config=config,
    )


__all__ = [
    "clip_row_norm",
    "exact_coordinate_intersection",
    "inject_tar_residual",
    "robust_quantile_keep_probability",
    "run_shape_tar",
    "run_texture_tar",
    "sample_stage_twins",
]
