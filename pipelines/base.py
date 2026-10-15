#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared base classes and utilities for FlowEdit pipelines.

Shared utilities for the VS3D editing stages.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import utils3d


# ---------------------------------------------------------------------------
# device / image helpers
# ---------------------------------------------------------------------------
def resolve_device(pipeline: Any) -> torch.device:
    """Resolve pipeline runtime device as a torch.device."""
    dev = getattr(pipeline, "device", None)
    if dev is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(dev, torch.device):
        return dev
    return torch.device(str(dev))


def load_image_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode == "RGB":
        return img
    img = img.convert("RGBA")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    return bg


def preprocess_image_pair(
    pipeline: Any, xsrc_path: str, xedit_path: str
) -> Tuple[Image.Image, Image.Image]:
    images = [load_image_rgb(xsrc_path), load_image_rgb(xedit_path)]
    if hasattr(pipeline, "preprocess_image"):
        for index, path in enumerate((xsrc_path, xedit_path)):
            try:
                images[index] = pipeline.preprocess_image(images[index])
            except Exception as error:
                raise RuntimeError(
                    f"VS3D image preprocessing failed for {path}: {error}"
                ) from error
    return images[0], images[1]


# ---------------------------------------------------------------------------
# voxel helpers
# ---------------------------------------------------------------------------
def ply_to_coords(ply_path: str, res: int, device: torch.device) -> torch.Tensor:
    position = utils3d.io.read_ply(ply_path)[0]
    coords = (
        (torch.tensor(position, device=device, dtype=torch.float32) + 0.5)
        * float(res)
    ).int().contiguous()
    coords = torch.clamp(coords, 0, res - 1)
    return coords


def coords_to_voxel(coords_xyz: torch.Tensor, res: int) -> torch.Tensor:
    voxel = torch.zeros(
        (1, 1, res, res, res), dtype=torch.float32, device=coords_xyz.device
    )
    voxel[:, 0, coords_xyz[:, 0], coords_xyz[:, 1], coords_xyz[:, 2]] = 1.0
    return voxel


def infer_coords_res(coords4: torch.Tensor) -> int:
    return int(coords4[:, 1:4].detach().max().item()) + 1


def ss_out_res_for_pipeline_type(pipeline_type: str, fallback: int = 32) -> int:
    mapping = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32}
    return int(mapping.get(str(pipeline_type), fallback))


# ---------------------------------------------------------------------------
# Flow sampler base
# ---------------------------------------------------------------------------
class FlowEditSamplerBase:
    """Shared low-level inference helpers for flow-based editing."""

    def __init__(self, device: torch.device):
        self.device = device

    def get_timesteps(
        self, steps: int, rescale_t: float = 5.0
    ) -> torch.Tensor:
        t = torch.linspace(1.0, 0.0, steps + 1, device=self.device, dtype=torch.float32)
        t = rescale_t * t / (1.0 + (rescale_t - 1.0) * t)
        return t

    def infer(
        self, model, x_t: torch.Tensor, t: float, cond: torch.Tensor
    ) -> torch.Tensor:
        t_in = torch.tensor(
            [1000.0 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32
        )
        return model(x_t, t_in, cond)

    def infer_cond_uncond(
        self,
        model,
        x_t: torch.Tensor,
        t: float,
        cond_pack: Dict[str, Any],
        cfg_interval: Tuple[float, float] = (0.6, 1.0),
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pred_cond = self.infer(model, x_t, t, cond_pack["cond"])
        pred_uncond: Optional[torch.Tensor] = None
        if (
            cfg_interval[0] <= t <= cfg_interval[1]
            and cond_pack.get("neg_cond", None) is not None
        ):
            pred_uncond = self.infer(model, x_t, t, cond_pack["neg_cond"])
        return pred_cond, pred_uncond

    @staticmethod
    def cfg(
        pred_cond: torch.Tensor,
        pred_uncond: Optional[torch.Tensor],
        cfg_strength: float,
    ) -> torch.Tensor:
        if pred_uncond is None:
            return pred_cond
        return pred_uncond + float(cfg_strength) * (pred_cond - pred_uncond)


# ---------------------------------------------------------------------------
# Stage 2/3: standard Trellis2 forward (shared by all pipelines)
# ---------------------------------------------------------------------------
@torch.no_grad()
def stage2_forward_shape_slat(
    pipeline: Any,
    coords: torch.Tensor,
    cond_512_tgt: Dict[str, Any],
    cond_1024_tgt: Optional[Dict[str, Any]],
    pipeline_type: str,
    max_num_tokens: int = 49152,
    shape_slat_sampler_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, int]:
    pipeline_type = pipeline_type or getattr(
        pipeline, "default_pipeline_type", "1024_cascade"
    )
    shape_slat_sampler_params = shape_slat_sampler_params or {}

    if pipeline_type == "512":
        flow_model = pipeline.models["shape_slat_flow_model_512"]
        shape_slat = pipeline.sample_shape_slat(
            cond_512_tgt, flow_model, coords, sampler_params=shape_slat_sampler_params
        )
        return shape_slat, 512

    if pipeline_type == "1024":
        assert cond_1024_tgt is not None
        flow_model = pipeline.models["shape_slat_flow_model_1024"]
        shape_slat = pipeline.sample_shape_slat(
            cond_1024_tgt, flow_model, coords, sampler_params=shape_slat_sampler_params
        )
        return shape_slat, 1024

    if pipeline_type == "1024_cascade":
        assert cond_1024_tgt is not None
        shape_slat, res = pipeline.sample_shape_slat_cascade(
            cond_512_tgt,
            cond_1024_tgt,
            pipeline.models["shape_slat_flow_model_512"],
            pipeline.models["shape_slat_flow_model_1024"],
            512,
            1024,
            coords,
            sampler_params=shape_slat_sampler_params,
            max_num_tokens=int(max_num_tokens),
        )
        return shape_slat, int(res)

    if pipeline_type == "1536_cascade":
        assert cond_1024_tgt is not None
        shape_slat, res = pipeline.sample_shape_slat_cascade(
            cond_512_tgt,
            cond_1024_tgt,
            pipeline.models["shape_slat_flow_model_512"],
            pipeline.models["shape_slat_flow_model_1024"],
            512,
            1536,
            coords,
            sampler_params=shape_slat_sampler_params,
            max_num_tokens=int(max_num_tokens),
        )
        return shape_slat, int(res)

    raise ValueError(f"Invalid pipeline_type: {pipeline_type}")


@torch.no_grad()
def stage3_forward_tex_slat(
    pipeline: Any,
    shape_slat: Any,
    cond_512_tgt: Dict[str, Any],
    cond_1024_tgt: Optional[Dict[str, Any]],
    pipeline_type: str,
    tex_slat_sampler_params: Optional[Dict[str, Any]] = None,
) -> Any:
    pipeline_type = pipeline_type or getattr(
        pipeline, "default_pipeline_type", "1024_cascade"
    )
    tex_slat_sampler_params = tex_slat_sampler_params or {}

    if pipeline_type == "512":
        flow_model = pipeline.models["tex_slat_flow_model_512"]
        return pipeline.sample_tex_slat(
            cond_512_tgt, flow_model, shape_slat, sampler_params=tex_slat_sampler_params
        )

    assert cond_1024_tgt is not None
    flow_model = pipeline.models["tex_slat_flow_model_1024"]
    return pipeline.sample_tex_slat(
        cond_1024_tgt, flow_model, shape_slat, sampler_params=tex_slat_sampler_params
    )
