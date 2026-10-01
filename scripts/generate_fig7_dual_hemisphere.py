"""
Script to generate Figure 7: Dual-Hemisphere UV Atlas & Industrial Rear Chassis Panel Synthesis.
Outputs: docs/images/fig7_dual_hemisphere_uv_chassis.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from backend.pbr_baker import PBRBaker

def generate_fig7():
    # 1. Synthesize industrial rear chassis suite
    chassis_suite = PBRBaker.generate_rear_chassis_textures(
        dominant_color=(45, 48, 54),
        size=(1024, 1024)
    )
    chassis_albedo = chassis_suite["albedo"]
    chassis_normal = chassis_suite["normal"]
    chassis_orm = chassis_suite["orm"]

    # 2. Load front texture from sample
    input_path = Path("output/test_cleaned_rgb.png")
    if not input_path.exists():
        input_path = Path("output/material_0.png")
    front_img = Image.open(input_path).convert("RGBA")
    
    # Clean black background into alpha
    src_np = np.array(front_img)
    is_black = (src_np[:, :, 0] < 5) & (src_np[:, :, 1] < 5) & (src_np[:, :, 2] < 5)
    src_np[is_black, 3] = 0
    clean_front = Image.fromarray(src_np)

    # Compose front onto 512x1024
    front_half = Image.new("RGB", (512, 1024), (127, 127, 127))
    scale = min((512 * 0.90) / clean_front.width, (1024 * 0.85) / clean_front.height)
    new_w, new_h = int(clean_front.width * scale), int(clean_front.height * scale)
    res_front = clean_front.resize((new_w, new_h), Image.Resampling.LANCZOS)
    front_half.paste(res_front, ((512 - new_w) // 2, (1024 - new_h) // 2), res_front)

    # Resize rear chassis to 512x1024
    rear_half = chassis_albedo.resize((512, 1024), Image.Resampling.LANCZOS)

    # Combine into Dual-Hemisphere UV Atlas (1024x1024)
    dual_atlas = Image.new("RGB", (1024, 1024))
    dual_atlas.paste(front_half, (0, 0))
    dual_atlas.paste(rear_half, (512, 0))

    # Setup matplotlib figure
    fig_w, fig_h = 16.0, 9.5
    dpi = 200

    plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans", "Helvetica"]
    plt.rcParams["font.family"] = "sans-serif"

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="#0b0f17")

    # Header Titles
    fig.text(
        0.5, 0.965,
        "Figure 7: Dual-Hemisphere UV Atlas & Industrial Rear Chassis Synthesis",
        color="#38bdf8",
        fontsize=18,
        fontweight="bold",
        ha="center",
        va="center"
    )
    fig.text(
        0.5, 0.932,
        "Front hemisphere optical camera projection (Nz >= 0) and synthesized rear chassis panel (Nz < 0) with ventilation slats and screw bosses",
        color="#94a3b8",
        fontsize=11,
        ha="center",
        va="center"
    )

    panels = [
        ("(a) Dual-Hemisphere UV Atlas (1024x1024)", "Left: Front Ray Projection (Nz >= 0) | Right: Rear Chassis (Nz < 0)", dual_atlas),
        ("(b) Industrial Rear Chassis Panel", "Synthesized Ventilation Slats, Perimeter Bevels & Technical Plate", chassis_albedo),
        ("(c) Rear Chassis Tangent Normal Map", "Multi-Scale Curvature Depth for Slats, Inset Bays & Screw Wells", chassis_normal),
        ("(d) Rear Chassis glTF ORM Map", "Packed R=AO Crevices, G=ABS Polymer Roughness, B=Metallic Screws", chassis_orm)
    ]

    for idx, (title, subtitle, img_data) in enumerate(panels):
        col = idx % 4
        left = 0.035 + col * 0.242
        bottom = 0.12
        width = 0.215
        height = 0.72

        ax = fig.add_axes([left, bottom, width, height], facecolor="#101726")
        ax.imshow(img_data)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#334155")
            spine.set_linewidth(1.5)

        # Highlight center split line on panel (a) at the top
        if idx == 0:
            ax.axvline(x=512, color="#38bdf8", linestyle="--", linewidth=1.5, alpha=0.85)
            ax.text(256, 80, "FRONT UV\n(Nz >= 0)", color="#38bdf8", fontsize=8.5, fontweight="bold", ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f172a", edgecolor="#38bdf8", alpha=0.9))
            ax.text(768, 80, "REAR UV\n(Nz < 0)", color="#f43f5e", fontsize=8.5, fontweight="bold", ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f172a", edgecolor="#f43f5e", alpha=0.9))

        ax.set_title(title, color="#f8fafc", fontsize=10.5, fontweight="bold", pad=12)
        ax.set_xlabel(subtitle, color="#94a3b8", fontsize=8.0, labelpad=8)

    output_path = Path("docs/images/fig7_dual_hemisphere_uv_chassis.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Figure 7 successfully saved to {output_path}")

if __name__ == "__main__":
    generate_fig7()
