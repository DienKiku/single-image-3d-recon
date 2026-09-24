#!/usr/bin/env python3
"""
==============================================================================
Stable Fast 3D (SF3D) Production Reconstruction Pipeline
Official Reference Framework: Stability AI (https://github.com/Stability-AI/stable-fast-3d)
==============================================================================

Architectural Breakdown:
------------------------
1. Triplane & Large Reconstruction Model (LRM):
   SF3D builds upon TripoSR with a high-throughput feed-forward transformer
   that predicts a continuous 3D NeRF / SDF representation in under 0.5s on GPU.
2. Fast Parameterization & UV Unwrapping:
   Instead of volumetric marching cubes alone, SF3D extracts an isosurface mesh
   and applies rapid atlas parameterization (via xatlas or boundary-preserving
   conformal chart decomposition). This unfolds complex manifold geometries
   into non-overlapping 2D UV islands with optimal packing efficiency.
3. Illumination Disentanglement (Delighting) & PBR Texture Baking:
   Unlike standard photo-projection which bakes directional shadows and baked
   specular highlights into the diffuse map, SF3D disentangles the lighting
   environment. A specialized feed-forward delighting stage predicts:
     - Base Color / Diffuse Albedo (pure intrinsic surface color)
     - Roughness & Metallic Maps (physically based rendering material response)
     - Bump / Normal Maps (high-frequency tactile micro-surface details)
   The textures are baked onto the UV layout via GPU-accelerated rasterization,
   producing game-engine-ready glTF/GLB models for Unity, Unreal Engine, & Three.js.

Key Capabilities:
-----------------
- Foreground isolation via Rembg (U2-Net) or Alpha mask.
- Hugging Face gated access token validation (`stabilityai/stable-fast-3d`).
- Three remeshing workflows: 'none', 'triangle' (Botsch & Kobbelt), 'quad' (Field-aligned).
- Dynamic VRAM footprint throttling (FP16 mixed precision, target <= 6GB VRAM).
- Dual-engine execution: official neural checkpoint when available, seamless
  local metric CAD unprojection engine when running strictly offline.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in sys.path when running as standalone script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import trimesh

from backend.sf3d_pipeline import (
    StableFast3DGenerator,
    AssetExporter3D,
    authenticate_huggingface,
    get_device,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for SF3D production pipeline."""
    parser = argparse.ArgumentParser(
        description="Stable Fast 3D (SF3D) Production 3D Reconstruction CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--image",
        required=True,
        type=str,
        help="Path to input 2D image file (JPG, PNG, WEBP).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="output/sf3d",
        type=str,
        help="Directory to save generated 3D assets (.glb, .stl, .obj, .png).",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face User Access Token for gated model access (stabilityai/stable-fast-3d).",
    )
    parser.add_argument(
        "--texture-resolution",
        type=int,
        choices=[512, 1024, 2048, 4096],
        default=1024,
        help="Output texture map resolution.",
    )
    parser.add_argument(
        "--remesh",
        type=str,
        choices=["none", "triangle", "quad"],
        default="none",
        help="Topological remeshing workflow: 'none', 'triangle' (Botsch & Kobbelt), 'quad' (Field-aligned).",
    )
    parser.add_argument(
        "--target-vertex-count",
        type=int,
        default=-1,
        help="Target polygon vertex count when remeshing is enabled (-1 to preserve native count).",
    )
    parser.add_argument(
        "--foreground-ratio",
        type=float,
        default=0.85,
        help="Ratio of foreground bounding box scale relative to camera frame (default: 0.85).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute device for execution. 'auto' selects CUDA if available.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Execute in half-precision (FP16) to conserve GPU memory within <= 6GB VRAM.",
    )
    parser.add_argument(
        "--target-dimensions-mm",
        nargs=3,
        type=float,
        default=None,
        metavar=("WIDTH", "HEIGHT", "DEPTH"),
        help="Optional real-world physical dimensions in millimeters to scale the solid model.",
    )

    return parser.parse_args()


def run_sf3d_pipeline(
    image_path: Union[str, Path],
    output_dir: Union[str, Path] = "output/sf3d",
    hf_token: Optional[str] = None,
    texture_resolution: int = 1024,
    remesh_option: str = "none",
    target_vertex_count: int = -1,
    foreground_ratio: float = 0.85,
    device: str = "auto",
    use_fp16: bool = True,
    target_dimensions_mm: Optional[Tuple[float, float, float]] = None,
) -> dict[str, Path]:
    """Execute complete SF3D reconstruction pipeline on an input image.

    Args:
        image_path: Path to the input 2D image.
        output_dir: Output destination directory.
        hf_token: Optional Hugging Face token for official weights.
        texture_resolution: Size of output baked texture maps (512, 1024, 2048, 4096).
        remesh_option: 'none', 'triangle', or 'quad'.
        target_vertex_count: Desired polygon vertex count if remeshing is enabled.
        foreground_ratio: Bounding box scale factor (standard SF3D value: 0.85).
        device: 'cuda', 'cpu', or 'auto'.
        use_fp16: Enable FP16 mixed precision for <= 6GB VRAM optimization.
        target_dimensions_mm: Optional (W, H, D) tuple in millimeters.

    Returns:
        Dictionary containing paths to exported .glb, .stl, and .obj files.
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # 1. Device Resolution & VRAM Management
    if device == "auto":
        resolved_device = get_device()
    else:
        resolved_device = device

    dtype = torch.float16 if (use_fp16 and resolved_device != "cpu") else torch.float32

    print("=" * 70)
    print("🚀 Stable Fast 3D (SF3D) Production Pipeline")
    print("=" * 70)
    print(f"• Input Image        : {image_path.name}")
    print(f"• Output Directory   : {output_dir.resolve()}")
    print(f"• Compute Device     : {resolved_device.upper()} ({dtype})")
    print(f"• Foreground Ratio   : {foreground_ratio:.2f}")
    print(f"• Texture Resolution : {texture_resolution}x{texture_resolution}")
    print(f"• Remesh Workflow    : {remesh_option} (Target vertices: {target_vertex_count})")
    if target_dimensions_mm:
        print(f"• Metric Scaling     : {target_dimensions_mm[0]:.1f} x {target_dimensions_mm[1]:.1f} x {target_dimensions_mm[2]:.1f} mm")
    print("-" * 70)

    # 2. Authenticate Hugging Face Hub if token provided
    if hf_token:
        authenticate_huggingface(hf_token)

    # 3. Instantiate SF3D Generator
    generator = StableFast3DGenerator(
        device=resolved_device,
        dtype=dtype,
        foreground_ratio=foreground_ratio,
        texture_resolution=texture_resolution,
        remesh_option=remesh_option,
        target_vertex_count=target_vertex_count,
        hf_token=hf_token,
    )

    # 4. Clean GPU Memory Before Inference (<= 6GB VRAM Safety Guard)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # 5. Core SF3D Inference
    print("[1/3] Processing image & generating 3D geometry...")
    input_pil = Image.open(image_path).convert("RGBA")
    mesh: trimesh.Trimesh = generator.run_image(
        image=input_pil,
        bake_resolution=texture_resolution,
        remesh=remesh_option,
        vertex_count=target_vertex_count,
        target_dimensions_mm=target_dimensions_mm,
    )

    # 6. Post-Processing & Export (GLB + STL + OBJ)
    print("[2/3] Aligning coordinate frame and grounding base...")
    base_name = image_path.stem
    print(f"[3/3] Exporting 3D assets to {output_dir.resolve()}...")
    export_files = AssetExporter3D.export_all(
        mesh=mesh,
        output_dir=output_dir,
        base_name=base_name,
        target_dimensions_mm=target_dimensions_mm,
    )

    # Clean GPU Memory After Inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print("-" * 70)
    print("✅ SF3D Reconstruction Complete!")
    print(f"• Total Vertices : {len(mesh.vertices):,}")
    print(f"• Total Faces    : {len(mesh.faces):,}")
    print(f"• Watertight     : {mesh.is_watertight}")
    if hasattr(mesh, "extents"):
        print(f"• Dimensions     : {mesh.extents[0]:.1f} x {mesh.extents[1]:.1f} x {mesh.extents[2]:.1f} mm")
    for fmt, p in export_files.items():
        print(f"• Exported [{fmt.upper()}] : {p.name} ({p.stat().st_size / 1024:.1f} KB)")
    print("=" * 70)

    return export_files


def main() -> None:
    """CLI Entrypoint."""
    args = parse_args()
    target_dims = tuple(args.target_dimensions_mm) if args.target_dimensions_mm else None
    
    try:
        run_sf3d_pipeline(
            image_path=args.image,
            output_dir=args.output_dir,
            hf_token=args.hf_token,
            texture_resolution=args.texture_resolution,
            remesh_option=args.remesh,
            target_vertex_count=args.target_vertex_count,
            foreground_ratio=args.foreground_ratio,
            device=args.device,
            use_fp16=args.fp16,
            target_dimensions_mm=target_dims,
        )
    except Exception as exc:
        print(f"\n❌ Error executing SF3D pipeline: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
