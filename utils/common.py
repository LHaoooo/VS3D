#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公共工具函数 — 被所有 runner 脚本共享。

包含：
  - set_seed: 确定性随机种子设置
  - ensure_file: 文件存在性检查
  - glb_to_ply: GLB 场景展开、归一化并转为 PLY
  - voxelize_ply_mesh: 三角网格体素化
  - patch_decode_latent: Monkey-patch decode_latent 以安全跳过 fill_holes
"""

from __future__ import annotations

import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    """尽力设置确定性随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_file(path: str, name: str) -> None:
    """检查文件是否存在，不存在则抛出 FileNotFoundError。"""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")


def glb_to_ply(
    glb_path: str,
    ply_path: str,
    blender_python: Optional[str] = None,
) -> str:
    """Run Blender GLB import, normalization, and PLY export."""
    ensure_file(glb_path, "glb_path")
    blender_python = blender_python or sys.executable
    ensure_file(blender_python, "blender_python")
    converter = PROJECT_ROOT / "tools" / "blender_glb_to_ply.py"
    ensure_file(str(converter), "blender_converter")

    os.makedirs(os.path.dirname(os.path.abspath(ply_path)) or ".", exist_ok=True)
    environment = os.environ.copy()
    runtime_lib = Path(blender_python).resolve().parent.parent / "lib"
    existing_library_path = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = (
        str(runtime_lib)
        if not existing_library_path
        else os.pathsep.join((str(runtime_lib), existing_library_path))
    )
    subprocess.run(
        [blender_python, str(converter), glb_path, ply_path],
        env=environment,
        check=True,
    )
    ensure_file(ply_path, "ply_path")
    return ply_path


def voxelize_ply_mesh(
    mesh_ply_path: str,
    voxels_path: str,
    voxel_res: int = 64,
) -> str:
    """将三角网格 (.ply) 在 [-0.5, 0.5]^3 范围内体素化，输出体素中心点云 .ply。"""
    import open3d as o3d

    ensure_file(mesh_ply_path, "mesh_ply_path")
    if voxel_res <= 0:
        raise ValueError(f"voxel_res must be positive, got {voxel_res}")

    mesh = o3d.io.read_triangle_mesh(mesh_ply_path)
    if mesh is None or mesh.is_empty() or len(mesh.triangles) == 0:
        raise ValueError(f"Failed to read triangle mesh from: {mesh_ply_path}")

    # 将顶点裁剪到 [-0.5, 0.5] 范围内
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh,
        voxel_size=1.0 / float(voxel_res),
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )

    idx = np.array([v.grid_index for v in voxel_grid.get_voxels()], dtype=np.int32)
    if idx.size == 0:
        raise ValueError(
            "Voxelization produced zero voxels. "
            "Check whether the mesh is within [-0.5, 0.5]^3."
        )

    # 将体素网格索引转换为归一化坐标中的体素中心
    pts = (idx.astype(np.float64) + 0.5) / float(voxel_res) - 0.5

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    ok = o3d.io.write_point_cloud(voxels_path, pcd, write_ascii=True)
    if not ok:
        raise IOError(f"Failed to write voxel ply: {voxels_path}")
    return voxels_path


def patch_decode_latent(pipeline) -> None:
    """Monkey-patch pipeline.decode_latent，安全跳过 OOM 的 fill_holes。

    避免修改上游 trellis2 源码，通过包装原始方法使 CuMesh.fill_holes()
    在 CUDA OOM 时优雅降级。
    """
    import types
    from trellis2.representations.mesh import MeshWithVoxel

    def _patched_decode_latent(self, shape_slat, tex_slat, resolution=256):
        meshes, subs = self.decode_shape_slat(shape_slat, resolution)
        # 在 tex 解码之前释放 shape decoder 的中间显存，缓解 24GB 卡 OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        tex_voxels = self.decode_tex_slat(tex_slat, subs)
        out_mesh = []
        for m, v in zip(meshes, tex_voxels):
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                m.fill_holes()
            except RuntimeError as e:
                msg = str(e).lower()
                if ("out of memory" in msg) or ("cuda error" in msg):
                    print("[WARN] decode_latent: CuMesh fill_holes OOM, "
                          "skipping hole filling for this mesh.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise
            # Detach tex voxel tensors: when activation checkpointing is
            # enabled on slat decoders, outputs carry requires_grad=True
            # even inside @torch.no_grad, which crashes downstream
            # `.numpy()` calls in o_voxel.postprocess.to_glb.
            try:
                _feats = v.feats.detach() if hasattr(v.feats, "detach") else v.feats
            except Exception:
                _feats = v.feats
            try:
                _coords = v.coords.detach() if hasattr(v.coords, "detach") else v.coords
            except Exception:
                _coords = v.coords
            out_mesh.append(
                MeshWithVoxel(
                    m.vertices, m.faces,
                    origin=[-0.5, -0.5, -0.5],
                    voxel_size=1 / resolution,
                    coords=_coords[:, 1:],
                    attrs=_feats,
                    voxel_shape=torch.Size([*v.shape, *v.spatial_shape]),
                    layout=self.pbr_attr_layout,
                )
            )
        return out_mesh

    pipeline.decode_latent = types.MethodType(_patched_decode_latent, pipeline)


def preprocess_3d_asset(
    glb_path: str,
    voxel_res: int = 64,
    blender_python: Optional[str] = None,
    *,
    output_dir: str,
) -> tuple:
    """预处理 3D 资产：glb -> ply -> voxel.ply。

    Returns:
        (mesh_ply_path, voxel_ply_path) 元组
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    glb_stem = os.path.splitext(os.path.basename(glb_path))[0]

    mesh_ply_path = os.path.join(output_dir, f"{glb_stem}.ply")
    print(">>> Converting input 3D asset (.glb) to .ply:", mesh_ply_path)
    mesh_ply_path = glb_to_ply(
        glb_path,
        mesh_ply_path,
        blender_python=blender_python,
    )

    print(">>> Voxelizing .ply to voxel centers (.ply)...")
    voxel_ply_path = voxelize_ply_mesh(
        mesh_ply_path,
        voxels_path=os.path.join(output_dir, f"voxels_{glb_stem}.ply"),
        voxel_res=voxel_res,
    )
    ensure_file(voxel_ply_path, "voxel_ply_path")

    return mesh_ply_path, voxel_ply_path
