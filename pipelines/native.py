"""Native TRELLIS2 stage-2/3 sampling with caller-supplied noise."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch


def _device(pipeline: Any) -> torch.device:
    device = getattr(pipeline, "device", None)
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device if isinstance(device, torch.device) else torch.device(str(device))


def _model_in_channels(model: Any) -> int:
    return int(model.in_channels if hasattr(model, "in_channels") else model[0].in_channels)


def _sampler_arguments(defaults: Mapping[str, Any], supplied: Mapping[str, Any]) -> dict:
    arguments = {**defaults, **supplied}
    arguments.setdefault("verbose", True)
    return arguments


def _denormalize(sample: Any, normalization: Mapping[str, Any]) -> Any:
    std = torch.as_tensor(
        normalization["std"], device=sample.device, dtype=sample.feats.dtype
    )[None]
    mean = torch.as_tensor(
        normalization["mean"], device=sample.device, dtype=sample.feats.dtype
    )[None]
    return sample.replace(sample.feats * std + mean)


@torch.no_grad()
def sample_shape_slat_1024(
    pipeline: Any,
    *,
    noise: torch.Tensor,
    scaffold: torch.Tensor,
    condition: Mapping[str, Any],
    **sampler_kwargs: Any,
) -> Any:
    """Run one 1024 shape-SLAT forward from supplied feature noise and coords."""
    from trellis2.modules.sparse import SparseTensor

    device = _device(pipeline)
    model = pipeline.models["shape_slat_flow_model_1024"]
    if noise.ndim != 2 or scaffold.ndim != 2 or scaffold.shape[1] != 4:
        raise ValueError("shape noise must be [N,C] and scaffold coords must be [N,4]")
    if noise.shape[0] != scaffold.shape[0]:
        raise ValueError("shape noise and scaffold must contain the same token count")
    if noise.shape[1] != _model_in_channels(model):
        raise ValueError("shape noise channel count does not match the flow model")

    sparse_noise = SparseTensor(
        feats=noise.to(device),
        coords=scaffold.to(device),
    )
    arguments = _sampler_arguments(
        pipeline.shape_slat_sampler_params, sampler_kwargs
    )
    arguments.setdefault("tqdm_desc", "Sampling shape SLat")
    low_vram = bool(getattr(pipeline, "low_vram", False))
    try:
        if low_vram:
            model.to(device)
        sample = pipeline.shape_slat_sampler.sample(
            model,
            sparse_noise,
            **condition,
            **arguments,
        ).samples
    finally:
        if low_vram:
            model.cpu()
    return _denormalize(sample, pipeline.shape_slat_normalization)


@torch.no_grad()
def sample_texture_slat_1024(
    pipeline: Any,
    *,
    noise: torch.Tensor,
    scaffold: Any,
    condition: Mapping[str, Any],
    **sampler_kwargs: Any,
) -> Any:
    """Run one 1024 texture-SLAT forward on a normalized shape scaffold."""
    device = _device(pipeline)
    model = pipeline.models["tex_slat_flow_model_1024"]
    expected_channels = _model_in_channels(model) - int(scaffold.feats.shape[1])
    if noise.ndim != 2 or noise.shape[0] != scaffold.coords.shape[0]:
        raise ValueError("texture noise and scaffold must contain the same token count")
    if noise.shape[1] != expected_channels:
        raise ValueError("texture noise channel count does not match the flow model")

    normalized_scaffold = scaffold
    if hasattr(scaffold, "to") and torch.device(scaffold.device) != device:
        normalized_scaffold = scaffold.to(device)
    sparse_noise = normalized_scaffold.replace(feats=noise.to(device))
    arguments = _sampler_arguments(pipeline.tex_slat_sampler_params, sampler_kwargs)
    arguments.setdefault("tqdm_desc", "Sampling texture SLat")
    low_vram = bool(getattr(pipeline, "low_vram", False))
    try:
        if low_vram:
            model.to(device)
        sample = pipeline.tex_slat_sampler.sample(
            model,
            sparse_noise,
            concat_cond=normalized_scaffold,
            **condition,
            **arguments,
        ).samples
    finally:
        if low_vram:
            model.cpu()
    return _denormalize(sample, pipeline.tex_slat_normalization)


def run_native_shape(
    pipeline: Any,
    *,
    noise: torch.Tensor,
    scaffold: torch.Tensor,
    condition: Mapping[str, Any],
    sampler_kwargs: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Native shape route: exactly one target-conditioned forward."""
    return sample_shape_slat_1024(
        pipeline,
        noise=noise,
        scaffold=scaffold,
        condition=condition,
        **dict(sampler_kwargs or {}),
    )


def run_native_texture(
    pipeline: Any,
    *,
    noise: torch.Tensor,
    scaffold: Any,
    condition: Mapping[str, Any],
    sampler_kwargs: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Native texture route: exactly one target-conditioned forward."""
    return sample_texture_slat_1024(
        pipeline,
        noise=noise,
        scaffold=scaffold,
        condition=condition,
        **dict(sampler_kwargs or {}),
    )


__all__ = [
    "run_native_shape",
    "run_native_texture",
    "sample_shape_slat_1024",
    "sample_texture_slat_1024",
]
