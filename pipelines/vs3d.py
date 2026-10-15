"""Canonical VS3D inference pipeline."""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from config import VS3DConfig

from .base import preprocess_image_pair, resolve_device
from .native import run_native_shape, run_native_texture
from .source_encoding import (
    encode_shape_slat_from_glb,
    encode_texture_slat_from_glb,
)
from .stage1 import stage1_rasi_pmg
from .tar import run_shape_tar, run_texture_tar


TEXTURE_SEED_OFFSET = 1000

DEFAULT_SHAPE_SAMPLER_PARAMS: Dict[str, Any] = {
    "steps": 25,
    "guidance_strength": 7.5,
    "guidance_interval": (0.6, 1.0),
    "guidance_rescale": 0.5,
}

DEFAULT_TEXTURE_SAMPLER_PARAMS: Dict[str, Any] = {
    "steps": 25,
    "guidance_strength": 1.0,
    "guidance_interval": (0.6, 0.9),
    "guidance_rescale": 0.0,
}


def derive_stage_seeds(seed: int) -> Tuple[int, int]:
    """Return deterministic, distinct shape and texture seeds."""
    shape_seed = int(seed)
    return shape_seed, shape_seed + TEXTURE_SEED_OFFSET


def _model_in_channels(model: Any) -> int:
    return int(model.in_channels if hasattr(model, "in_channels") else model[0].in_channels)


def _sampler_params(
    defaults: Mapping[str, Any], overrides: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    return {**defaults, **dict(overrides or {})}


def _noise(
    token_count: int,
    channel_count: int,
    *,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if token_count <= 0:
        raise ValueError("sparse stages require at least one occupancy token")
    if channel_count <= 0:
        raise ValueError("sparse stage noise channel count must be positive")
    generator = torch.Generator(device=device).manual_seed(int(seed))
    return torch.randn(
        token_count,
        channel_count,
        generator=generator,
        device=device,
    )


def _normalize_shape_slat(pipeline: Any, shape_slat: Any) -> Any:
    std = torch.as_tensor(
        pipeline.shape_slat_normalization["std"],
        device=shape_slat.device,
        dtype=shape_slat.feats.dtype,
    )[None]
    mean = torch.as_tensor(
        pipeline.shape_slat_normalization["mean"],
        device=shape_slat.device,
        dtype=shape_slat.feats.dtype,
    )[None]
    return shape_slat.replace((shape_slat.feats - mean) / std)


def _prepare_decode_memory(pipeline: Any) -> None:
    if not bool(getattr(pipeline, "low_vram", False)):
        return
    decoder_names = {"shape_slat_decoder", "tex_slat_decoder"}
    for name, model in getattr(pipeline, "models", {}).items():
        if name in decoder_names:
            continue
        cpu = getattr(model, "cpu", None)
        if callable(cpu):
            cpu()
    for name in decoder_names:
        decoder = getattr(pipeline, "models", {}).get(name)
        if decoder is None or not hasattr(decoder, "modules"):
            continue
        for module in decoder.modules():
            if hasattr(module, "use_checkpoint"):
                module.use_checkpoint = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def _decode_and_export_glb_impl(
    pipeline: Any,
    shape_slat: Any,
    texture_slat: Any,
    *,
    output_path: str,
    resolution: int = 1024,
) -> Dict[str, Any]:
    """Decode final SLATs and export one non-empty GLB."""
    import o_voxel

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    decoded = pipeline.decode_latent(shape_slat, texture_slat, int(resolution))
    if not decoded:
        raise RuntimeError("VS3D decode returned no meshes")
    mesh = decoded[0]
    vertex_count = int(mesh.vertices.shape[0])
    face_count = int(mesh.faces.shape[0])
    if vertex_count == 0 or face_count == 0:
        raise RuntimeError("VS3D decode returned an empty mesh")

    try:
        mesh.simplify(16777216)
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    glb = None
    attempts = (
        (1_000_000, True, 4096),
        (500_000, True, 2048),
        (500_000, False, 2048),
    )
    for decimation_target, remesh, texture_size in attempts:
        try:
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=getattr(mesh, "attrs", getattr(mesh, "attr_volume", None)),
                coords=mesh.coords,
                attr_layout=getattr(mesh, "layout", pipeline.pbr_attr_layout),
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=decimation_target,
                texture_size=texture_size,
                remesh=remesh,
                remesh_band=1,
                remesh_project=0,
                verbose=False,
            )
            break
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if glb is None:
        raise RuntimeError("all VS3D GLB postprocess attempts ran out of memory")

    try:
        glb.export(output_path, extension_webp=True)
    except TypeError:
        glb.export(output_path)
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(f"VS3D export did not create a non-empty GLB: {output_path}")
    return {
        "export_ok": True,
        "output_path": output_path,
        "vertex_count": vertex_count,
        "face_count": face_count,
    }


def decode_and_export_glb(
    pipeline: Any,
    shape_slat: Any,
    texture_slat: Any,
    *,
    output_path: str,
    resolution: int = 1024,
) -> Dict[str, Any]:
    """Decode/export and return low-VRAM decoders to CPU on every exit."""
    try:
        return _decode_and_export_glb_impl(
            pipeline,
            shape_slat,
            texture_slat,
            output_path=output_path,
            resolution=resolution,
        )
    finally:
        if bool(getattr(pipeline, "low_vram", False)):
            for name in ("shape_slat_decoder", "tex_slat_decoder"):
                decoder = getattr(pipeline, "models", {}).get(name)
                cpu = getattr(decoder, "cpu", None)
                if callable(cpu):
                    cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _diagnostics_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _diagnostics_to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_diagnostics_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_diagnostics_to_cpu(item) for item in value]
    return value


def run_vs3d(
    pipeline: Any,
    *,
    src_glb_path: str,
    mesh_ply_path: Optional[str],
    voxel_ply_path: str,
    xsrc_path: str,
    xedit_path: str,
    output_path: str,
    config: Optional[VS3DConfig] = None,
    seed: int = 42,
    encoder_ckpt_dir: Optional[str] = None,
    shape_sampler_params: Optional[Mapping[str, Any]] = None,
    texture_sampler_params: Optional[Mapping[str, Any]] = None,
    cfg_src: float = 1.5,
    cfg_tgt: float = 9.0,
    n_max: int = 12,
    n_min: int = 0,
    cfg_interval: Tuple[float, float] = (0.6, 1.0),
    rasi_inner_steps: int = 3,
    rasi_lr: float = 1e-5,
    rasi_early_stop_eps: float = 1e-5,
) -> Dict[str, Any]:
    """Run FlowEdit + optional RASI/PMG + native/TAR sparse stages."""
    runtime_config = config or VS3DConfig()
    device = resolve_device(pipeline)
    source_image, target_image = preprocess_image_pair(
        pipeline, xsrc_path, xedit_path
    )
    cond_512_source = pipeline.get_cond([source_image], 512)
    cond_512_target = pipeline.get_cond([target_image], 512)
    cond_1024_source = pipeline.get_cond([source_image], 1024)
    cond_1024_target = pipeline.get_cond([target_image], 1024)

    with torch.enable_grad():
        coords = stage1_rasi_pmg(
            pipeline,
            voxel_ply_path=voxel_ply_path,
            cond_src=cond_512_source,
            cond_tgt=cond_512_target,
            config=runtime_config,
            cfg_src=float(cfg_src),
            cfg_tgt=float(cfg_tgt),
            n_max=int(n_max),
            n_min=int(n_min),
            cfg_interval=cfg_interval,
            ss_out_res=64,
            voxel_in_res=64,
            rasi_inner_steps=int(rasi_inner_steps),
            rasi_lr=float(rasi_lr),
            rasi_early_stop_eps=float(rasi_early_stop_eps),
        )
    if coords.ndim != 2 or coords.shape[1] != 4 or coords.shape[0] == 0:
        raise RuntimeError("Stage 1 returned invalid or empty occupancy coordinates")
    coords = coords.to(device=device, dtype=torch.int32)

    shape_seed, texture_seed = derive_stage_seeds(seed)
    shape_model = pipeline.models["shape_slat_flow_model_1024"]
    shape_noise = _noise(
        coords.shape[0],
        _model_in_channels(shape_model),
        seed=shape_seed,
        device=device,
    )
    resolved_shape_params = _sampler_params(
        DEFAULT_SHAPE_SAMPLER_PARAMS, shape_sampler_params
    )

    if runtime_config.shape_mode == "tar":
        source_shape = encode_shape_slat_from_glb(
            pipeline,
            src_glb_path,
            resolution=1024,
            device=device,
            ckpt_dir=encoder_ckpt_dir,
            mesh_ply_path=mesh_ply_path,
        )
        shape_slat, shape_diagnostics = run_shape_tar(
            pipeline,
            noise=shape_noise,
            scaffold=coords,
            source_condition=cond_1024_source,
            target_condition=cond_1024_target,
            source_slat=source_shape,
            sampler_kwargs=resolved_shape_params,
            config=runtime_config,
        )
        shape_forward_count = int(shape_diagnostics["forward_count"])
        del source_shape
    else:
        shape_slat = run_native_shape(
            pipeline,
            noise=shape_noise,
            scaffold=coords,
            condition=cond_1024_target,
            sampler_kwargs=resolved_shape_params,
        )
        shape_diagnostics = {"forward_count": 1}
        shape_forward_count = 1

    normalized_shape = _normalize_shape_slat(pipeline, shape_slat)
    texture_model = pipeline.models["tex_slat_flow_model_1024"]
    texture_channels = _model_in_channels(texture_model) - int(
        normalized_shape.feats.shape[1]
    )
    texture_noise = _noise(
        normalized_shape.coords.shape[0],
        texture_channels,
        seed=texture_seed,
        device=device,
    )
    resolved_texture_params = _sampler_params(
        DEFAULT_TEXTURE_SAMPLER_PARAMS, texture_sampler_params
    )

    if runtime_config.texture_mode == "tar":
        source_texture = encode_texture_slat_from_glb(
            pipeline,
            src_glb_path,
            resolution=1024,
            device=device,
            ckpt_dir=encoder_ckpt_dir,
            mesh_ply_path=mesh_ply_path,
        )
        texture_slat, texture_diagnostics = run_texture_tar(
            pipeline,
            noise=texture_noise,
            scaffold=normalized_shape,
            source_condition=cond_1024_source,
            target_condition=cond_1024_target,
            source_slat=source_texture,
            sampler_kwargs=resolved_texture_params,
            config=runtime_config,
        )
        texture_forward_count = int(texture_diagnostics["forward_count"])
        del source_texture
    else:
        texture_slat = run_native_texture(
            pipeline,
            noise=texture_noise,
            scaffold=normalized_shape,
            condition=cond_1024_target,
            sampler_kwargs=resolved_texture_params,
        )
        texture_diagnostics = {"forward_count": 1}
        texture_forward_count = 1

    shape_diagnostics = _diagnostics_to_cpu(shape_diagnostics)
    texture_diagnostics = _diagnostics_to_cpu(texture_diagnostics)
    del (
        shape_noise,
        texture_noise,
        normalized_shape,
        coords,
        source_image,
        target_image,
        cond_512_source,
        cond_512_target,
        cond_1024_source,
        cond_1024_target,
    )
    _prepare_decode_memory(pipeline)
    decode_info = decode_and_export_glb(
        pipeline,
        shape_slat,
        texture_slat,
        output_path=output_path,
        resolution=1024,
    )
    return {
        **decode_info,
        "config": runtime_config,
        "shape_seed": shape_seed,
        "texture_seed": texture_seed,
        "shape_forward_count": shape_forward_count,
        "texture_forward_count": texture_forward_count,
        "shape_diagnostics": shape_diagnostics,
        "texture_diagnostics": texture_diagnostics,
    }


__all__ = [
    "DEFAULT_SHAPE_SAMPLER_PARAMS",
    "DEFAULT_TEXTURE_SAMPLER_PARAMS",
    "TEXTURE_SEED_OFFSET",
    "decode_and_export_glb",
    "derive_stage_seeds",
    "run_vs3d",
]
