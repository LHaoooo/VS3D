#!/usr/bin/env python3
"""Run the Blender GLB-to-PLY conversion inside Blender Python."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.bpy_render import BpyRenderer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("glb_path")
    parser.add_argument("ply_path")
    args = parser.parse_args()

    renderer = BpyRenderer(
        resolution=1024,
        engine="CYCLES",
        geo_mode=False,
        split_normal=False,
    )
    result = renderer.glb_to_ply(
        glb_path=args.glb_path,
        ply_path=args.ply_path,
    )
    output = result.get("ply_path")
    if not result.get("exported") or not output or not Path(output).is_file():
        raise RuntimeError(f"Blender did not export the requested PLY: {args.ply_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
