"""
Generate Figure 1: Salient Object Extraction & Canonical Canvas Normalization Pipeline.
Outputs: docs/images/fig1_input_preprocessing_pipeline.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from backend.depth_processor import estimate_depth_map

def generate_fig1():
    raw_path = Path("assets/samples/hero_battery_holder.png")
    if not raw_path.exists():
        raw_path = Path("output/test_cleaned_rgb.png")

    raw_bgr = cv2.imread(str(raw_path))
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

    clean_path = Path("output/test_cleaned_rgb.png")
    clean_img = Image.open(clean_path).convert("RGBA")

    # Depth map
    depth = estimate_depth_map(raw_bgr)
    depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

    # 512x512 Canvas
    canvas = Image.new("RGB", (512, 512), (127, 127, 127))
    scale = min((512 * 0.85) / clean_img.width, (512 * 0.85) / clean_img.height)
    nw, nh = int(clean_img.width * scale), int(clean_img.height * scale)
    res = clean_img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas.paste(res, ((512 - nw) // 2, (512 - nh) // 2), res)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), dpi=200, facecolor="#0b0f17")
    fig.suptitle(
        "Figure 1: Salient Object Extraction & Canonical Canvas Normalization Pipeline",
        color="#38bdf8",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )

    axes[0].imshow(raw_rgb)
    axes[0].set_title("(a) Raw Input 2D Photograph\n(Mechanical Dimensions: 76x41 mm)", color="#f1f5f9", fontsize=11, pad=10)

    axes[1].imshow(clean_img)
    axes[1].set_title("(b) High-Precision Alpha Matte\n(Isolated Chassis & Terminals)", color="#f1f5f9", fontsize=11, pad=10)

    axes[2].imshow(depth_colored)
    axes[2].set_title("(c) Metric Monocular Depth Map\n(Cavity Gradient & Spring Relief)", color="#f1f5f9", fontsize=11, pad=10)

    axes[3].imshow(canvas)
    axes[3].set_title("(d) Normalized Neural Canvas\n(512x512 Centered, Pad Gray 127)", color="#f1f5f9", fontsize=11, pad=10)

    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#0b0f17")

    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.92])
    out_file = Path("docs/images/fig1_input_preprocessing_pipeline.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, facecolor="#0b0f17", dpi=200)
    plt.close()
    print(f"Saved {out_file} ({out_file.stat().st_size} bytes)")

if __name__ == "__main__":
    generate_fig1()
