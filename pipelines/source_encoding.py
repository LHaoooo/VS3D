"""Source GLB/PLY normalization and SC-VAE SLAT encoding for TAR."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch


_SC_VAE_ENCODERS: Dict[str, Dict[str, Any]] = {}

_PBR_TEXTURE_ATTRIBUTES = (
    "baseColorTexture",
    "emissiveTexture",
    "metallicRoughnessTexture",
    "normalTexture",
    "occlusionTexture",
    "image",
)

_HISTORICAL_TEXTURE_SIZE = 1024


def normalize_pbr_textures_for_voxelization(
    asset: Any,
) -> list[tuple[str, tuple[int, int], tuple[int, int]]]:
    """Apply the historical selected-assets texture repair in memory."""
    from PIL import Image

    geometry = getattr(asset, "geometry", None)
    geometries = geometry.values() if hasattr(geometry, "values") else (asset,)
    textures = []
    seen = set()
    has_invalid_texture = False
    for mesh in geometries:
        material = getattr(getattr(mesh, "visual", None), "material", None)
        if material is None:
            continue
        for attribute in _PBR_TEXTURE_ATTRIBUTES:
            key = (id(material), attribute)
            if key in seen:
                continue
            seen.add(key)
            texture = getattr(material, attribute, None)
            if not isinstance(texture, Image.Image):
                continue
            width, height = map(int, texture.size)
            textures.append((material, attribute, texture))
            if width != height or width <= 0 or width & (width - 1) != 0:
                has_invalid_texture = True

    if not has_invalid_texture:
        return []

    resized = []
    target_size = (_HISTORICAL_TEXTURE_SIZE, _HISTORICAL_TEXTURE_SIZE)
    for material, attribute, texture in textures:
        original_size = tuple(map(int, texture.size))
        if original_size == target_size:
            continue
        if texture.mode not in ("RGB", "RGBA"):
            texture = texture.convert("RGBA" if "A" in texture.mode else "RGB")
        normalized = texture.resize(target_size, Image.Resampling.LANCZOS)
        setattr(material, attribute, normalized)
        resized.append((attribute, original_size, target_size))
    return resized


def _device(pipeline: Any, device: Optional[torch.device]) -> torch.device:
    if device is not None:
        return device
    pipeline_device = getattr(pipeline, "device", None)
    if pipeline_device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (
        pipeline_device
        if isinstance(pipeline_device, torch.device)
        else torch.device(str(pipeline_device))
    )


def _default_checkpoint_directory() -> str:
    return os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "hub/models--microsoft--TRELLIS.2-4B/snapshots/"
        "05a74714e786a12aac1a02c3037771fa77bbad96/ckpts",
    )


def load_sc_vae_encoders(
    device: torch.device,
    ckpt_dir: Optional[str] = None,
    encoder_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Load requested SC-VAE encoders into a device-independent CPU cache."""
    import trellis2.models as trellis2_models

    checkpoint_directory = ckpt_dir or _default_checkpoint_directory()
    key = os.path.abspath(checkpoint_directory)
    encoders = _SC_VAE_ENCODERS.setdefault(key, {})
    requested = tuple(encoder_names or ("shape_encoder", "tex_encoder"))
    checkpoint_names = {
        "shape_encoder": "shape_enc_next_dc_f16c32_fp16",
        "tex_encoder": "tex_enc_next_dc_f16c32_fp16",
    }
    unknown = set(requested) - set(checkpoint_names)
    if unknown:
        raise ValueError(f"unknown SC-VAE encoder names: {sorted(unknown)}")
    for name in requested:
        if name not in encoders:
            path = os.path.join(checkpoint_directory, checkpoint_names[name])
            encoders[name] = trellis2_models.from_pretrained(path).eval().cpu()
    return encoders


@contextmanager
def _encoder_on_device(
    encoders: Dict[str, Any],
    name: str,
    device: torch.device,
    *,
    low_vram: bool,
):
    encoder = encoders[name]
    try:
        encoder.to(device)
        yield encoder
    finally:
        if low_vram:
            for loaded_encoder in encoders.values():
                loaded_encoder.cpu()


def load_and_normalize_glb(
    glb_path: str,
    *,
    mesh_ply_path: Optional[str] = None,
    need_texture: bool = False,
) -> Any:
    """Normalize a source mesh with the established GLB-to-PLY bbox mapping."""
    import trimesh

    has_ply = mesh_ply_path is not None and os.path.isfile(mesh_ply_path)
    if has_ply and not need_texture:
        asset = trimesh.load(mesh_ply_path, force="mesh")
        vertex_min = float(asset.vertices.min())
        vertex_max = float(asset.vertices.max())
        if vertex_min < -0.6 or vertex_max > 0.6:
            bounds = np.asarray(asset.bounds, dtype=np.float64)
            center = (bounds[0] + bounds[1]) / 2.0
            extent = (bounds[1] - bounds[0]).max()
            asset.apply_translation(-center)
            asset.apply_scale(0.99999 / extent)
        return asset

    asset = trimesh.load(glb_path)
    if need_texture:
        resized_textures = normalize_pbr_textures_for_voxelization(asset)
        if resized_textures:
            details = ", ".join(
                f"{attribute} {before[0]}x{before[1]}->{after[0]}x{after[1]}"
                for attribute, before, after in resized_textures
            )
            print(
                f">>> Resized {len(resized_textures)} source PBR texture(s) "
                f"for o_voxel: {details}"
            )
    if has_ply and need_texture:
        glb_bounds = np.asarray(asset.bounds, dtype=np.float64)
        glb_center = (glb_bounds[0] + glb_bounds[1]) / 2.0
        glb_half = (glb_bounds[1] - glb_bounds[0]).max() / 2.0
        ply_mesh = trimesh.load(mesh_ply_path, force="mesh")
        ply_bounds = np.asarray(ply_mesh.bounds, dtype=np.float64)
        ply_center = (ply_bounds[0] + ply_bounds[1]) / 2.0
        ply_half = (ply_bounds[1] - ply_bounds[0]).max() / 2.0
        asset.apply_translation(-glb_center)
        asset.apply_scale(float(ply_half) / (float(glb_half) + 1e-12))
        asset.apply_translation(ply_center)
        return asset

    bounds = np.asarray(asset.bounds, dtype=np.float64)
    center = (bounds[0] + bounds[1]) / 2.0
    extent = (bounds[1] - bounds[0]).max()
    asset.apply_translation(-center)
    asset.apply_scale(0.99999 / extent)
    if isinstance(asset, trimesh.Scene):
        try:
            asset = asset.to_geometry()
        except Exception:
            asset = asset.dump(concatenate=True)
    return asset


@torch.no_grad()
def encode_shape_slat_from_glb(
    pipeline: Any,
    glb_path: str,
    *,
    resolution: int = 1024,
    device: Optional[torch.device] = None,
    ckpt_dir: Optional[str] = None,
    mesh_ply_path: Optional[str] = None,
) -> Any:
    """Encode source geometry into a raw shape SLAT with TRELLIS2 SC-VAE."""
    import o_voxel
    from trellis2.modules.sparse import SparseTensor

    resolved_device = _device(pipeline, device)
    encoders = load_sc_vae_encoders(
        resolved_device, ckpt_dir, encoder_names=("shape_encoder",)
    )
    with _encoder_on_device(
        encoders,
        "shape_encoder",
        resolved_device,
        low_vram=bool(getattr(pipeline, "low_vram", False)),
    ) as encoder:
        mesh = load_and_normalize_glb(glb_path, mesh_ply_path=mesh_ply_path)
        vertices = torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32)).float()
        faces = torch.from_numpy(np.asarray(mesh.faces, dtype=np.int64)).long()
        voxel_indices, dual_vertices, intersected = (
            o_voxel.convert.mesh_to_flexible_dual_grid(
                vertices,
                faces,
                grid_size=int(resolution),
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                face_weight=1.0,
                boundary_weight=0.2,
                regularization_weight=1e-2,
            )
        )
        mapping = torch.argsort(o_voxel.serialize.encode_seq(voxel_indices))
        voxel_indices = voxel_indices[mapping]
        dual_vertices = dual_vertices[mapping]
        intersected = intersected[mapping]
        packed_vertices = (
            dual_vertices * resolution - voxel_indices.float()
        ).clamp(0, 1)
        coords = torch.cat(
            [
                torch.zeros(voxel_indices.shape[0], 1, dtype=voxel_indices.dtype),
                voxel_indices,
            ],
            dim=-1,
        ).to(resolved_device)
        vertices_sparse = SparseTensor(
            feats=packed_vertices.float().to(resolved_device), coords=coords
        )
        intersected_sparse = vertices_sparse.replace(
            intersected.bool().float().to(resolved_device)
        )
        return encoder(
            vertices_sparse, intersected_sparse, sample_posterior=False
        )


@torch.no_grad()
def encode_texture_slat_from_glb(
    pipeline: Any,
    glb_path: str,
    *,
    resolution: int = 1024,
    device: Optional[torch.device] = None,
    ckpt_dir: Optional[str] = None,
    mesh_ply_path: Optional[str] = None,
) -> Any:
    """Encode source PBR attributes into a raw texture SLAT with SC-VAE."""
    import o_voxel
    from trellis2.modules.sparse import SparseTensor

    resolved_device = _device(pipeline, device)
    encoders = load_sc_vae_encoders(
        resolved_device, ckpt_dir, encoder_names=("tex_encoder",)
    )
    with _encoder_on_device(
        encoders,
        "tex_encoder",
        resolved_device,
        low_vram=bool(getattr(pipeline, "low_vram", False)),
    ) as encoder:
        asset = load_and_normalize_glb(
            glb_path,
            mesh_ply_path=mesh_ply_path,
            need_texture=True,
        )
        voxel_indices, attributes = (
            o_voxel.convert.textured_mesh_to_volumetric_attr(
                asset,
                grid_size=int(resolution),
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            )
        )
        mapping = torch.argsort(o_voxel.serialize.encode_seq(voxel_indices))
        voxel_indices = voxel_indices[mapping]
        attributes = {name: value[mapping] for name, value in attributes.items()}
        features = []
        for name in ("base_color", "metallic", "roughness", "alpha"):
            value = attributes[name]
            if value.dtype == torch.uint8:
                value = value.float() / 255.0
            else:
                value = value.float()
            features.append(value * 2.0 - 1.0)
        coords = torch.cat(
            [
                torch.zeros(voxel_indices.shape[0], 1, dtype=voxel_indices.dtype),
                voxel_indices,
            ],
            dim=-1,
        ).to(resolved_device)
        source = SparseTensor(
            feats=torch.cat(features, dim=-1).to(resolved_device),
            coords=coords,
        )
        return encoder(source, sample_posterior=False)


# Backward-compatible alias for callers using the shorter name.
encode_tex_slat_from_glb = encode_texture_slat_from_glb


__all__ = [
    "encode_shape_slat_from_glb",
    "encode_tex_slat_from_glb",
    "encode_texture_slat_from_glb",
    "load_and_normalize_glb",
    "load_sc_vae_encoders",
    "normalize_pbr_textures_for_voxelization",
]
