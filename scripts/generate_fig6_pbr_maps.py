"""
Script to generate Figure 6: 6-Channel PBR Material Baking & Physics Surface Maps Decomposition.
Outputs: docs/images/fig6_pbr_material_baking_suite.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from backend.pbr_baker import PBRBaker

def generate_fig6():
    # Use clean input image and center cleanly onto neutral studio gray canvas
    input_path = Path("output/test_cleaned_rgb.png")
    if not input_path.exists():
        input_path = Path("output/material_0.png")
    
    src = Image.open(input_path).convert("RGBA")
    # Clean black background into alpha if needed
    src_np = np.array(src)
    is_black = (src_np[:, :, 0] < 5) & (src_np[:, :, 1] < 5) & (src_np[:, :, 2] < 5)
    src_np[is_black, 3] = 0
    clean_src = Image.fromarray(src_np)

    # Compose onto 1024x1024 canvas
    canvas_size = 1024
    raw_img = Image.new("RGB", (canvas_size, canvas_size), (127, 127, 127))
    scale = min((canvas_size * 0.78) / clean_src.width, (canvas_size * 0.78) / clean_src.height)
    new_w, new_h = int(clean_src.width * scale), int(clean_src.height * scale)
    resized_src = clean_src.resize((new_w, new_h), Image.Resampling.LANCZOS)
    offset_x = (canvas_size - new_w) // 2
    offset_y = (canvas_size - new_h) // 2
    raw_img.paste(resized_src, (offset_x, offset_y), resized_src)

    # Bake all 6 channels
    maps = PBRBaker.bake_pbr_maps(raw_img, target_size=(canvas_size, canvas_size))

    fig_w, fig_h = 16.0, 10.0
    dpi = 200

    plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans", "Helvetica"]
    plt.rcParams["font.family"] = "sans-serif"

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="#0b0f17")

    # Title & Subtitle
    fig.text(
        0.5, 0.965,
        "Figure 6: 6-Channel PBR Material Decomposition & Texture Baking Suite",
        color="#38bdf8",
        fontsize=18,
        fontweight="bold",
        ha="center",
        va="center"
    )
    fig.text(
        0.5, 0.932,
        "100% local physics-based decomposition: Albedo Delighting, Tangent-Space Normals, Roughness, Metallic, AO, and glTF ORM",
        color="#94a3b8",
        fontsize=11,
        ha="center",
        va="center"
    )

    panels = [
        ("(a) Raw Input 2D Photograph", "RGB Camera Capture with Shadows & Highlights", raw_img, "rgb"),
        ("(b) Delighted Albedo (Base Color)", "Bilateral Luminance Delighting (Shadows Removed)", maps["albedo"], "rgb"),
        ("(c) Tangent-Space Normal Map", "Multi-Scale Sobel Curvature Gradients (3x3 & 5x5)", maps["normal"], "rgb"),
        ("(d) Perceptual Roughness Map", "Dielectric Casings (0.75) vs Specular Surfaces (0.20)", maps["roughness"], "gray"),
        ("(e) Metallic Mask", "Metal Screw Bosses & Terminals (0.90) vs Plastic (0.0)", maps["metallic"], "gray"),
        ("(f) Packed glTF 2.0 ORM Texture", "R = Ambient Occlusion | G = Roughness | B = Metallic", maps["orm"], "rgb")
    ]

    # 2 rows x 3 columns
    for idx, (title, subtitle, img_data, mode) in enumerate(panels):
        row = idx // 3
        col = idx % 3

        # Grid positioning with clean margins
        left = 0.05 + col * 0.315
        bottom = 0.50 - row * 0.43
        width = 0.27
        height = 0.36

        ax = fig.add_axes([left, bottom, width, height], facecolor="#101726")
        
        if mode == "gray":
            ax.imshow(img_data, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(img_data)
        
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#334155")
            spine.set_linewidth(1.5)

        # Panel title and subtitle
        ax.set_title(title, color="#f8fafc", fontsize=11, fontweight="bold", pad=14)
        ax.set_xlabel(subtitle, color="#94a3b8", fontsize=8.5, labelpad=8)

    output_path = Path("docs/images/fig6_pbr_material_baking_suite.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Figure 6 successfully saved to {output_path}")

if __name__ == "__main__":
    generate_fig6()
