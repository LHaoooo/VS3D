#!/usr/bin/env python3
"""Command-line entry point for canonical VS3D inference."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from config import VS3DConfig

REQUIRED_MODEL_NAMES = (
    "sparse_structure_decoder",
    "sparse_structure_encoder",
    "sparse_structure_flow_model",
    "shape_slat_decoder",
    "shape_slat_flow_model_1024",
    "tex_slat_decoder",
    "tex_slat_flow_model_1024",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("VS3D: RASI + PMG + TAR 3D editing")
    parser.add_argument("--glb-asset")
    parser.add_argument("--source-image")
    parser.add_argument("--target-image")
    parser.add_argument("--output")

    parser.add_argument(
        "--hf-cache",
        default=os.path.join(PROJECT_ROOT, "weights"),
        help="Hugging Face cache root (default: weights/)",
    )
    parser.add_argument(
        "--model-path",
        default=os.path.join(PROJECT_ROOT, "weights", "TRELLIS.2-4B"),
        help="Local TRELLIS.2 checkpoint directory (default: weights/TRELLIS.2-4B)",
    )
    parser.add_argument("--encoder-ckpt-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--low-vram",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--skip-fill-holes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--rasi",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--rasi-probe-variant", choices=("A1", "A0"), default="A1"
    )
    parser.add_argument(
        "--pmg",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--pmg-num-samples", type=int, default=5)
    parser.add_argument("--pmg-partial-samples", type=int, default=2)
    parser.add_argument("--pmg-weight", type=float, default=1.2)
    parser.add_argument(
        "--pmg-mode",
        choices=("original", "consistency", "crossfit"),
        default="consistency",
    )
    parser.add_argument("--pmg-consistency-tau", type=float, default=0.9767255187034607)
    parser.add_argument("--pmg-crossfit-kappa-2", type=float, default=None)
    parser.add_argument("--pmg-crossfit-kappa-3", type=float, default=None)
    parser.add_argument("--shape-mode", choices=("tar", "native"), default="tar")
    parser.add_argument("--texture-mode", choices=("tar", "native"), default="tar")
    parser.add_argument("--tar-lambda", type=float, default=0.5)
    parser.add_argument("--tar-clip-tau", type=float, default=10.0)
    parser.add_argument("--tar-keep-threshold", type=float, default=0.7)
    parser.add_argument("--tar-quantile-low", type=float, default=0.05)
    parser.add_argument("--tar-quantile-high", type=float, default=0.95)

    parser.add_argument("--stage1-steps", type=int, default=25)
    parser.add_argument("--cfg-src", type=float, default=1.5)
    parser.add_argument("--cfg-tgt", type=float, default=9.0)
    parser.add_argument("--n-max", type=int, default=12)
    parser.add_argument("--n-min", type=int, default=0)
    parser.add_argument("--cfg-interval-low", type=float, default=0.6)
    parser.add_argument("--cfg-interval-high", type=float, default=1.0)
    parser.add_argument("--rasi-inner-steps", type=int, default=3)
    parser.add_argument("--rasi-lr", type=float, default=1e-5)
    parser.add_argument("--rasi-early-stop-eps", type=float, default=1e-5)
    parser.add_argument(
        "--load-rembg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove image backgrounds before preprocessing (default: enabled)",
    )

    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render a 360-degree preview of the edited asset after inference",
    )
    parser.add_argument(
        "--render-output-dir",
        default=None,
        help="Directory for render outputs; defaults to the inference output directory",
    )
    parser.add_argument("--render-frames", type=int, default=120)
    parser.add_argument("--render-fps", type=int, default=24)
    parser.add_argument("--render-resolution", type=int, default=512)
    parser.add_argument("--render-samples", type=int, default=8)
    parser.add_argument(
        "--render-python",
        default=sys.executable,
        help="Python interpreter for GLB-to-PLY and rendering (default: current Python)",
    )

    parser.add_argument("--stage23-steps", type=int, default=25)
    parser.add_argument("--shape-guidance", type=float, default=7.5)
    parser.add_argument("--shape-guidance-rescale", type=float, default=0.5)
    parser.add_argument("--texture-guidance", type=float, default=1.0)
    parser.add_argument("--texture-guidance-rescale", type=float, default=0.0)
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def config_from_args(args: argparse.Namespace) -> VS3DConfig:
    return VS3DConfig(
        stage1_steps=args.stage1_steps,
        rasi_enabled=args.rasi,
        rasi_probe_variant=args.rasi_probe_variant,
        pmg_enabled=args.pmg,
        pmg_num_samples=args.pmg_num_samples,
        pmg_partial_samples=args.pmg_partial_samples,
        pmg_weight=args.pmg_weight,
        pmg_mode=args.pmg_mode,
        pmg_consistency_tau=args.pmg_consistency_tau,
        pmg_crossfit_kappa_2=args.pmg_crossfit_kappa_2,
        pmg_crossfit_kappa_3=args.pmg_crossfit_kappa_3,
        shape_mode=args.shape_mode,
        texture_mode=args.texture_mode,
        tar_lambda=args.tar_lambda,
        tar_clip_tau=args.tar_clip_tau,
        tar_keep_threshold=args.tar_keep_threshold,
        tar_quantiles=(args.tar_quantile_low, args.tar_quantile_high),
    )


def _validate_inputs(args: argparse.Namespace) -> None:
    required = {
        "--glb-asset": args.glb_asset,
        "--source-image": args.source_image,
        "--target-image": args.target_image,
        "--output": args.output,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"missing required VS3D input arguments: {', '.join(missing)}")
    for name in ("glb_asset", "source_image", "target_image"):
        path = getattr(args, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"VS3D input {name} not found: {path}")
    if not os.path.isfile(os.path.join(args.model_path, "pipeline.json")):
        raise FileNotFoundError(
            f"TRELLIS2 model snapshot not found: {args.model_path}"
        )
    if args.encoder_ckpt_dir is not None and not os.path.isdir(args.encoder_ckpt_dir):
        raise FileNotFoundError(
            f"SC-VAE checkpoint directory not found: {args.encoder_ckpt_dir}"
        )


def load_pipeline(
    model_path: str, *, device: str, low_vram: bool, load_rembg: bool = True
):
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
        model_path,
        model_names=REQUIRED_MODEL_NAMES,
        load_rembg=load_rembg,
    )
    pipeline.low_vram = bool(low_vram)
    pipeline.to(device)
    return pipeline


def _shape_sampler_params(args: argparse.Namespace) -> dict:
    return {
        "steps": args.stage23_steps,
        "guidance_strength": args.shape_guidance,
        "guidance_interval": (0.6, 1.0),
        "guidance_rescale": args.shape_guidance_rescale,
    }


def _texture_sampler_params(args: argparse.Namespace) -> dict:
    return {
        "steps": args.stage23_steps,
        "guidance_strength": args.texture_guidance,
        "guidance_interval": (0.6, 0.9),
        "guidance_rescale": args.texture_guidance_rescale,
    }


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    config = config_from_args(args)
    _validate_inputs(args)

    # Configure the cache before importing any model-loading dependencies.
    args.hf_cache = os.path.abspath(os.path.expanduser(args.hf_cache))
    os.environ["HF_HOME"] = args.hf_cache
    os.environ["HF_HUB_CACHE"] = os.path.join(args.hf_cache, "hub")

    import torch
    from pipelines.vs3d import run_vs3d
    from tools.render_outputs import render_preview
    from utils.common import patch_decode_latent, preprocess_3d_asset, set_seed

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    print("VS3D config:", json.dumps(asdict(config), sort_keys=True))
    print(f"VS3D HF cache: {args.hf_cache}")
    print(f"VS3D model: {args.model_path}")
    print(f"VS3D stage seeds: shape={args.seed}, texture={args.seed + 1000}")

    pipeline = load_pipeline(
        args.model_path,
        device=device,
        low_vram=args.low_vram,
        load_rembg=args.load_rembg,
    )
    if args.skip_fill_holes:
        patch_decode_latent(pipeline)
    mesh_ply_path, voxel_ply_path = preprocess_3d_asset(
        args.glb_asset,
        blender_python=args.render_python,
        output_dir=os.path.dirname(os.path.abspath(args.output)),
    )
    encoder_ckpt_dir = args.encoder_ckpt_dir or os.path.join(
        args.model_path, "ckpts"
    )
    result = run_vs3d(
        pipeline,
        src_glb_path=args.glb_asset,
        mesh_ply_path=mesh_ply_path,
        voxel_ply_path=voxel_ply_path,
        xsrc_path=args.source_image,
        xedit_path=args.target_image,
        output_path=args.output,
        config=config,
        seed=args.seed,
        encoder_ckpt_dir=encoder_ckpt_dir,
        shape_sampler_params=_shape_sampler_params(args),
        texture_sampler_params=_texture_sampler_params(args),
        cfg_src=args.cfg_src,
        cfg_tgt=args.cfg_tgt,
        n_max=args.n_max,
        n_min=args.n_min,
        cfg_interval=(args.cfg_interval_low, args.cfg_interval_high),
        rasi_inner_steps=args.rasi_inner_steps,
        rasi_lr=args.rasi_lr,
        rasi_early_stop_eps=args.rasi_early_stop_eps,
    )
    print(
        "VS3D forwards: "
        f"shape={result['shape_forward_count']}, "
        f"texture={result['texture_forward_count']}"
    )
    print(f"VS3D output: {result.get('output_path', args.output)}")

    if args.render:
        output_glb = result.get("output_path", args.output)
        render_dir = args.render_output_dir or os.path.dirname(
            os.path.abspath(output_glb)
        )
        try:
            preview = render_preview(
                output_glb,
                output_dir=render_dir,
                render_python=args.render_python,
                frames=args.render_frames,
                fps=args.render_fps,
                resolution=args.render_resolution,
                samples=args.render_samples,
            )
            print(f"VS3D preview video: {preview['video']}")
        except Exception as error:
            print(f"VS3D preview render failed ({error}); the edited GLB is kept.")
    return result


if __name__ == "__main__":
    main()
