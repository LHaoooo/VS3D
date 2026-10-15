#!/usr/bin/env python3
"""Render a 360-degree video or 16 images of the edited asset using Blender.

Run from the inference environment. --views writes PNGs at elevations 15 and
30 degrees, with eight evenly spaced directions at each elevation. No external
renderer or metric computation is used. Pass --render-python to reuse an
existing Blender Python runtime; the default is the current Python interpreter.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable

# This file is also executed as a subprocess by the render runtime, which may not
# inherit the caller's working directory, so the repository root is added here.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

def _renderer_environment(python: Path) -> dict[str, str]:
    environment = os.environ.copy()
    runtime_lib = python.resolve().parent.parent / "lib"
    existing = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = str(runtime_lib) + (os.pathsep + existing if existing else "")
    return environment



def normalize_glb(source: Path, destination: Path, *, max_edge: float = 2.0) -> dict[str, object]:
    """Center the prediction and scale its longest axis to ``max_edge``.

    The raw ``edited.glb`` is never modified. Video and still images read
    the same normalized display copy.
    """
    import numpy as np
    import trimesh

    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"prediction GLB not found or empty: {source}")
    mesh = trimesh.load(str(source), force="mesh", process=False)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError(f"prediction GLB has invalid bounds: {source}")
    center = (bounds[0] + bounds[1]) / 2.0
    longest = float((bounds[1] - bounds[0]).max())
    if not math.isfinite(longest) or longest <= 0.0:
        raise ValueError(f"prediction GLB has degenerate bounds: {source}")
    scale = longest / max_edge
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.eye(3, dtype=np.float64) / scale
    transform[:3, 3] = -center / scale

    asset = trimesh.load(str(source), force=None, process=False)
    asset.apply_transform(transform)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.glb")
    try:
        asset.export(str(temporary), file_type="glb")
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise IOError(f"normalization produced no GLB: {destination}")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "source": str(source),
        "normalized": str(destination),
        "max_edge": max_edge,
        "center": center.tolist(),
        "half_extent": longest / 2.0,
    }


def camera_radius_for_bounds(
    bounds_min: Iterable[float],
    bounds_max: Iterable[float],
    *,
    fov_deg: float = 49.134,
    fill: float = 0.88,
    samples: int = 360,
) -> float:
    """Estimate an orbit radius from box corners and sampled horizontal views.

    The calculation does not account for camera elevation or prove a
    globally minimal framing distance.
    """
    lo = tuple(float(v) for v in bounds_min)
    hi = tuple(float(v) for v in bounds_max)
    center = [(a + b) * 0.5 for a, b in zip(lo, hi)]
    corners = [
        (hi[0] if mask & 1 else lo[0],
         hi[1] if mask & 2 else lo[1],
         hi[2] if mask & 4 else lo[2])
        for mask in range(8)
    ]
    half = math.tan(math.radians(fov_deg) * 0.5) * max(fill, 1e-3)
    worst = 0.0
    for step in range(max(samples, 4)):
        azimuth = 2.0 * math.pi * step / max(samples, 4)
        # Measure corner offsets along and perpendicular to each horizontal
        # viewing direction; camera elevation is not sampled here.
        ca, sa = math.cos(azimuth), math.sin(azimuth)
        direction = (sa, -ca, 0.0)
        for corner in corners:
            rel = [corner[i] - center[i] for i in range(3)]
            along = sum(rel[i] * direction[i] for i in range(3))
            lateral = math.sqrt(sum(value * value for value in rel) - along * along)
            required = abs(along) + lateral / half
            worst = max(worst, required)
    return max(worst, 1e-6)


def _mesh_bounds(model: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    import numpy as np
    import trimesh

    mesh = trimesh.load(str(model), force="mesh", process=False)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    return tuple(bounds[0].tolist()), tuple(bounds[1].tolist())


def _setup_scene(model: Path, *, resolution: int, samples: int, fov: float,
                 elevation: float, distance: float):
    """Set up the existing Blender lighting and camera for both outputs."""
    import bpy
    from mathutils import Vector

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.1
    scene.cycles.max_bounces = 3
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transmission_bounces = 2
    scene.cycles.transparent_max_bounces = 2
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for device_type in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = device_type
            break
        except TypeError:
            continue
    prefs.get_devices()
    gpu_types = {"OPTIX", "CUDA", "HIP", "METAL"}
    for device in prefs.devices:
        device.use = device.type in gpu_types
    scene.cycles.device = "GPU"

    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.frame_start = 0

    world = scene.world or bpy.data.worlds.new("TurntableWorld")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.18, 0.18, 0.18, 1.0)
    background.inputs["Strength"].default_value = 0.8
    try:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass

    bpy.ops.import_scene.gltf(filepath=str(model))
    meshes = [obj for obj in scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"model contains no mesh objects: {model}")
    corners = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    lo = [min(c[i] for c in corners) for i in range(3)]
    hi = [max(c[i] for c in corners) for i in range(3)]
    center = [(a + b) * 0.5 for a, b in zip(lo, hi)]

    target = bpy.data.objects.new("TurntableTarget", None)
    target.location = center
    scene.collection.objects.link(target)
    orbit = bpy.data.objects.new("TurntableOrbit", None)
    orbit.location = center
    scene.collection.objects.link(orbit)

    cam_data = bpy.data.cameras.new("TurntableCamera")
    cam_data.type = "PERSP"
    cam_data.sensor_fit = "VERTICAL"
    cam_data.angle = math.radians(fov)
    cam_data.clip_start = 1e-4
    cam_data.clip_end = distance * 20.0 + 10.0
    cam = bpy.data.objects.new("TurntableCamera", cam_data)
    scene.collection.objects.link(cam)
    cam.parent = orbit
    pitch = math.radians(elevation)
    cam.location = (distance * math.cos(pitch), 0.0, distance * math.sin(pitch))
    track = cam.constraints.new(type="TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    scene.camera = cam

    for name, energy, rotation in (
        ("Key", 2.2, (math.radians(28), 0.0, math.radians(-35))),
        ("Fill", 1.2, (math.radians(58), 0.0, math.radians(120))),
        ("Rim", 1.5, (math.radians(-35), 0.0, math.radians(210))),
    ):
        data = bpy.data.lights.new(name, type="SUN")
        data.energy = energy
        light = bpy.data.objects.new(name, data)
        light.rotation_euler = rotation
        scene.collection.objects.link(light)

    return scene, orbit, cam


def _render_views(model: Path, output: Path, *, resolution: int, samples: int,
                  fov: float, distance: float) -> dict[str, object]:
    import bpy

    scene, orbit, camera = _setup_scene(
        model, resolution=resolution, samples=samples, fov=fov,
        elevation=15.0, distance=distance,
    )
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = "RGBA"
    output.mkdir(parents=True, exist_ok=True)
    index = 0
    for elevation in (15.0, 30.0):
        pitch = math.radians(elevation)
        camera.location = (distance * math.cos(pitch), 0.0, distance * math.sin(pitch))
        for azimuth in range(0, 360, 45):
            orbit.rotation_euler.z = math.radians(azimuth)
            scene.render.filepath = str(output / f"render_{index:04d}.png")
            bpy.context.view_layer.update()
            bpy.ops.render.render(write_still=True)
            index += 1
    return {"views": index, "directory": str(output)}


def _turntable_video(
    model: Path,
    output: Path,
    *,
    resolution: int,
    frames: int,
    fps: int,
    samples: int,
    fov: float,
    fill: float,
    elevation: float,
    distance: float,
    keep_frames: bool,
) -> dict[str, object]:
    import bpy

    scene, orbit, _ = _setup_scene(
        model, resolution=resolution, samples=samples, fov=fov,
        elevation=elevation, distance=distance,
    )
    scene.render.fps = fps
    scene.frame_end = frames - 1
    orbit.rotation_mode = "XYZ"
    orbit.keyframe_insert(data_path="rotation_euler", index=2, frame=0)
    orbit.rotation_euler.z = 2.0 * math.pi
    orbit.keyframe_insert(data_path="rotation_euler", index=2, frame=frames)
    for fcurve in orbit.animation_data.action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "LINEAR"

    frames_dir = Path(tempfile.mkdtemp(prefix="vs3d_turntable_"))
    try:
        scene.render.filepath = str(frames_dir/"frame_")
        scene.frame_set(0)
        bpy.context.view_layer.update()
        bpy.ops.render.render(animation=True)
        rendered = sorted(frames_dir.glob("frame_*.png"))
        if len(rendered) != frames:
            raise RuntimeError(f"expected {frames} frames, found {len(rendered)}")

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            from importlib.metadata import distribution

            binaries = distribution("imageio-ffmpeg").locate_file("imageio_ffmpeg/binaries")
            ffmpeg = next((str(p) for p in binaries.glob("ffmpeg*") if os.access(p, os.X_OK)), None)
        if ffmpeg is None:
            raise RuntimeError("Install ffmpeg or imageio-ffmpeg in the render environment")
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_suffix(".partial.mp4")
        encoded = subprocess.run(
            [
                str(ffmpeg), "-hide_banner", "-loglevel", "warning", "-y",
                "-framerate", str(fps), "-start_number", "0",
                "-i", str(frames_dir/"frame_%04d.png"),
                "-frames:v", str(frames), "-an", "-c:v", "libx264",
                "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(partial),
            ],
            capture_output=True,
            text=True,
        )
        if encoded.returncode or not partial.is_file() or partial.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg failed:\n{encoded.stderr[-1500:]}")
        partial.replace(output)
    finally:
        if keep_frames:
            print(f"frames kept at {frames_dir}")
        else:
            shutil.rmtree(frames_dir, ignore_errors=True)

    return {
        "video": str(output),
        "bytes": output.stat().st_size,
        "frames": frames,
        "fps": fps,
        "resolution": [resolution, resolution],
        "camera_distance": distance,
    }


def render_preview(
    glb_path,
    *,
    output_dir,
    render_python=None,
    frames: int = 120,
    fps: int = 24,
    resolution: int = 512,
    samples: int = 8,
) -> dict[str, object]:
    """Normalize a prediction and render the 360-degree preview video.

    Called by the inference runner so that a user always gets a video of the
    edited asset. The Blender runtime interpreter is invoked as a subprocess and
    receives the renderer library path through its environment.
    """
    glb = Path(glb_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    interpreter = Path(render_python) if render_python else Path(sys.executable)
    normalized = destination/"normalized.glb"
    summary = {
        "normalization": normalize_glb(glb, normalized, max_edge=2.0),
    }
    bounds_min, bounds_max = _mesh_bounds(normalized)
    summary["bounds"] = {"min": bounds_min, "max": bounds_max}
    distance = camera_radius_for_bounds(
        bounds_min, bounds_max, fov_deg=50.0, fill=0.86
    )
    command = [
        str(interpreter),
        str(Path(__file__).resolve()),
        "--glb", str(normalized),
        "--output-dir", str(destination),
        "--video",
        "--turntable-only",
        "--camera-distance", repr(distance),
        "--video-resolution", str(resolution),
        "--video-samples", str(samples),
        "--frames", str(frames),
        "--fps", str(fps),
    ]
    completed = subprocess.run(
        command,
        env=_renderer_environment(interpreter),
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            "turntable render failed:\n" + (completed.stdout + completed.stderr)[-2000:]
        )
    summary["video"] = {
        "video": str(destination/"turntable.mp4"),
        "camera_distance": distance,
        "frames": frames,
        "fps": fps,
        "resolution": [resolution, resolution],
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glb", required=True, help="Raw prediction GLB (edited.glb)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--max-edge",
        type=float,
        default=2.0,
        help="Longest axis of the normalized copy used by both outputs",
    )
    parser.add_argument("--video", action="store_true", help="Render the 360-degree turntable")
    parser.add_argument("--views", action="store_true", help="Render 16 images of the edited asset")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--video-resolution", type=int, default=512)
    parser.add_argument("--video-samples", type=int, default=8)
    parser.add_argument("--view-resolution", type=int, default=1024)
    parser.add_argument("--view-samples", type=int, default=8)
    parser.add_argument("--fov", type=float, default=50.0, help="Video camera field of view")
    parser.add_argument("--video-fill", type=float, default=0.86)
    parser.add_argument("--elevation", type=float, default=12.0)
    parser.add_argument(
        "--view-fill",
        type=float,
        default=0.88,
        help="Target object-to-frame ratio used to choose the display camera distance",
    )
    parser.add_argument(
        "--render-python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter for rendering (default: current Python)",
    )
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument(
        "--turntable-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--views-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--camera-distance", type=float, help=argparse.SUPPRESS)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.views_only:
        if not args.views or args.camera_distance is None:
            raise SystemExit("--views-only requires --views and --camera-distance")
        summary = _render_views(
            Path(args.glb), Path(args.output_dir) / "views",
            resolution=args.view_resolution, samples=args.view_samples,
            fov=args.fov, distance=args.camera_distance,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.turntable_only:
        # Rendering runs in a child process, using the current Python by default.
        # The parent has already normalized the asset; render only the turntable.
        if not args.video or args.camera_distance is None:
            raise SystemExit("--turntable-only requires --video and --camera-distance")
        model = Path(args.glb).expanduser().resolve()
        output_dir = Path(args.output_dir).expanduser().resolve()
        summary = {
            "video": _turntable_video(
                model,
                output_dir/"turntable.mp4",
                resolution=args.video_resolution,
                frames=args.frames,
                fps=args.fps,
                samples=args.video_samples,
                fov=args.fov,
                fill=args.video_fill,
                elevation=args.elevation,
                distance=args.camera_distance,
                keep_frames=args.keep_frames,
            )
        }
        print(json.dumps(summary, indent=2))
        return 0

    if not args.video and not args.views:
        raise SystemExit("choose at least one of --video or --views")
    for name in ("frames", "fps", "video_resolution", "video_samples", "view_resolution", "view_samples"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 < args.view_fill <= 1.0:
        raise SystemExit("--view-fill must be in (0, 1]")

    render_python = Path(args.render_python).expanduser().absolute()
    source = Path(args.glb).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    normalized = output_dir/"normalized.glb"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "normalization": normalize_glb(source, normalized, max_edge=args.max_edge)
    }
    bounds_min, bounds_max = _mesh_bounds(normalized)
    summary["bounds"] = {"min": bounds_min, "max": bounds_max}

    if args.video:
        distance = camera_radius_for_bounds(
            bounds_min,
            bounds_max,
            fov_deg=args.fov,
            fill=args.video_fill,
        )
        command = [
            str(render_python),
            str(Path(__file__).resolve()),
            "--glb", str(normalized),
            "--output-dir", str(output_dir),
            "--video",
            "--turntable-only",
            "--camera-distance", repr(distance),
            "--video-resolution", str(args.video_resolution),
            "--video-samples", str(args.video_samples),
            "--frames", str(args.frames),
            "--fps", str(args.fps),
            "--fov", repr(args.fov),
            "--video-fill", repr(args.video_fill),
            "--elevation", repr(args.elevation),
        ]
        if args.keep_frames:
            command.append("--keep-frames")
        completed = subprocess.run(
            command,
            env=_renderer_environment(render_python),
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            sys.stderr.write(completed.stdout + completed.stderr)
            raise SystemExit(f"turntable render failed: exit {completed.returncode}")
        summary["video"] = {
            "video": str(output_dir/"turntable.mp4"),
            "camera_distance": distance,
        }

    if args.views:
        radius = camera_radius_for_bounds(
            bounds_min, bounds_max, fov_deg=args.fov, fill=args.view_fill,
        )
        command = [
            str(render_python), str(Path(__file__).resolve()),
            "--glb", str(normalized), "--output-dir", str(output_dir),
            "--views", "--views-only", "--camera-distance", repr(radius),
            "--view-resolution", str(args.view_resolution),
            "--view-samples", str(args.view_samples), "--fov", repr(args.fov),
        ]
        completed = subprocess.run(
            command, env=_renderer_environment(render_python),
            capture_output=True, text=True,
        )
        if completed.returncode:
            raise RuntimeError("16-view render failed:\n" + (completed.stdout + completed.stderr)[-2000:])
        files = [output_dir / "views" / f"render_{i:04d}.png" for i in range(16)]
        if not all(p.is_file() and p.stat().st_size for p in files):
            raise RuntimeError("16-view rendering did not produce all 16 PNGs")
        summary["views"] = {"views": len(files), "directory": str(output_dir / "views")}

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())