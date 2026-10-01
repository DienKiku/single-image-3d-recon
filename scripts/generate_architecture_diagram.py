import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def generate_architecture_diagram(output_path: str = "docs/images/fig0_system_architecture.png"):
    # Target dimensions: 3180 x 1980 at 200 DPI -> 15.9 x 9.9 inches
    fig_w, fig_h = 15.9, 9.9
    dpi = 200

    # Prefer modern clean Windows sans-serif fonts
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans", "Helvetica"]
    plt.rcParams["font.family"] = "sans-serif"

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("#0b0f17")
    ax.set_facecolor("#0b0f17")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Title & Subtitle
    ax.text(
        50, 95.8,
        "End-to-End System Architecture: Single-Image to 3D Reconstruction",
        color="#38bdf8",
        fontsize=19,
        fontweight="bold",
        ha="center",
        va="center"
    )
    ax.text(
        50, 92.8,
        "A 100% offline modular pipeline transforming single 2D photographs into watertight metric CAD assets with full 6-channel PBR materials",
        color="#94a3b8",
        fontsize=10.5,
        ha="center",
        va="center"
    )

    # 5 Stages configuration - exactly 4 cards each for perfect geometric symmetry
    stages = [
        {
            "num": "STAGE 1",
            "title": "Image Acquisition &\nPreprocessing",
            "color": "#38bdf8",  # Sky blue
            "accent": "#0284c7",
            "cards": [
                ("2D Input Photograph", "Unconstrained Camera Capture (.png, .jpg, .webp)"),
                ("Salient Object Segmentation", "U2-Net Memory-Safe High-Precision Alpha Matte"),
                ("Canvas Normalization", "Canonical 512×512 Centering & 85% Scale Ratio"),
                ("Background Neutralization", "Neutral Gray (V=127) Padding & Bilateral Denoising")
            ]
        },
        {
            "num": "STAGE 2",
            "title": "360° Foundation Model\n(TripoSR / NeRF)",
            "color": "#c084fc",  # Light Purple
            "accent": "#9333ea",
            "cards": [
                ("Vision Transformer Tokenizer", "facebook/dino-vitb16 Feed-Forward Image Tokens"),
                ("Triplane NeRF Field Decoder", "Cross-Attention Continuous Density Field MLP"),
                ("Marching Cubes Isosurface", "256³ Resolution Solid Watertight Enclosure"),
                ("Quadric Mesh Simplification", "fast_simplification → Exactly ~20,000 Faces")
            ]
        },
        {
            "num": "STAGE 3",
            "title": "CAD Alignment &\nMetric Scaling",
            "color": "#fbbf24",  # Amber / Gold
            "accent": "#d97706",
            "cards": [
                ("CAD Frame Transformation", "X_cad=Y, Y_cad=X, Z_cad=Z (Upright Coordinate)"),
                ("Grounding & Base Alignment", "Object Base Grounded at Y=0, Centered on XZ"),
                ("Metric Scale Calibration", "Physical Dimensions in mm (ArUco Marker / ID Card)"),
                ("Ergonomic Rear CAD Beveling", "Volumetric Chamfering: dz ~ clamp(r - 0.65)^2")
            ]
        },
        {
            "num": "STAGE 4",
            "title": "Local 6-Channel PBR\nBaking Engine",
            "color": "#f43f5e",  # Rose / Pink
            "accent": "#e11d48",
            "cards": [
                ("Dual-Hemisphere UV Atlas", "Front Photo Projection & Rear Chassis Atlas"),
                ("Albedo Delighting & Normals", "Bilateral Luminance Filter & Multi-Scale Sobel"),
                ("Roughness & Metallic Engine", "Physics Classification: Glass vs Polymer vs Metal"),
                ("glTF ORM Texture Packing", "Packed R=AO, G=Roughness, B=Metal + Vent Slats")
            ]
        },
        {
            "num": "STAGE 5",
            "title": "Watertight Assembly &\nMulti-Format Delivery",
            "color": "#34d399",  # Emerald Green
            "accent": "#059669",
            "cards": [
                ("Analytical Planar Capping", "cap=True Exact Polygon Watertight Solid Slicing"),
                ("Area Gradient Seam Scanner", "Smart Joint Detection: min |dA/ds| (Y / Z Axis)"),
                ("Interactive Three.js Studio", "Clay, Photo, Realistic PBR, Wireframe, 1% Zoom HUD"),
                ("Multi-Format Industrial Export", ".glb (PBR) + .obj/.mtl + .stl + .zip Bundle")
            ]
        }
    ]

    num_stages = len(stages)
    col_width = 18.0
    gap = 1.3
    total_w = num_stages * col_width + (num_stages - 1) * gap
    start_x = (100 - total_w) / 2.0
    stage_bottom = 3.8
    stage_top = 89.2
    stage_height = stage_top - stage_bottom

    for i, stage in enumerate(stages):
        sx = start_x + i * (col_width + gap)
        scolor = stage["color"]

        # Outer Stage container box with subtle glow border
        stage_box = patches.FancyBboxPatch(
            (sx, stage_bottom), col_width, stage_height,
            boxstyle="round,pad=0.2,rounding_size=1.2",
            facecolor="#101726",
            edgecolor=scolor,
            linewidth=1.8,
            zorder=1
        )
        ax.add_patch(stage_box)

        # Stage Number Badge / Pill
        badge_w = 6.2
        badge_h = 2.4
        badge_x = sx + (col_width - badge_w) / 2.0
        badge_y = stage_top - 4.2
        badge_box = patches.FancyBboxPatch(
            (badge_x, badge_y), badge_w, badge_h,
            boxstyle="round,pad=0.1,rounding_size=0.8",
            facecolor="#1e293b",
            edgecolor=scolor,
            linewidth=1.2,
            zorder=2
        )
        ax.add_patch(badge_box)

        ax.text(
            sx + col_width / 2.0, badge_y + badge_h / 2.0,
            stage["num"],
            color=scolor,
            fontsize=9.5,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=3
        )

        # Stage Title
        ax.text(
            sx + col_width / 2.0, stage_top - 7.5,
            stage["title"],
            color="#f8fafc",
            fontsize=10.2,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=3
        )

        # Render cards inside stage (all 4 cards, identical dimensions)
        cards = stage["cards"]
        n_cards = len(cards)
        card_w = col_width - 1.8
        card_cx = sx + 0.9

        cards_top = stage_top - 12.2
        cards_bottom = stage_bottom + 2.2
        available_h = cards_top - cards_bottom
        card_h = 13.0
        c_gap = (available_h - n_cards * card_h) / (n_cards - 1)

        for ci, (ctitle, cdesc) in enumerate(cards):
            cy = cards_top - ci * (card_h + c_gap) - card_h

            # Card background
            card_box = patches.FancyBboxPatch(
                (card_cx, cy), card_w, card_h,
                boxstyle="round,pad=0.15,rounding_size=0.8",
                facecolor="#172237",
                edgecolor="#334155",
                linewidth=1.1,
                zorder=2
            )
            ax.add_patch(card_box)

            # Card title
            ax.text(
                card_cx + card_w / 2.0, cy + card_h * 0.65,
                ctitle,
                color="#ffffff",
                fontsize=8.6,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=3
            )

            # Card description
            ax.text(
                card_cx + card_w / 2.0, cy + card_h * 0.30,
                cdesc,
                color="#94a3b8",
                fontsize=7.1,
                ha="center",
                va="center",
                zorder=3
            )

            # Vertical arrow connecting to next card
            if ci < n_cards - 1:
                arrow_y1 = cy
                arrow_y2 = cy - c_gap
                ax.annotate(
                    "",
                    xy=(card_cx + card_w / 2.0, arrow_y2),
                    xytext=(card_cx + card_w / 2.0, arrow_y1),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=scolor,
                        lw=1.5,
                        mutation_scale=10
                    ),
                    zorder=3
                )

        # Horizontal arrow to next stage
        if i < num_stages - 1:
            arrow_x1 = sx + col_width + 0.08
            arrow_x2 = sx + col_width + gap - 0.08
            arrow_y = (stage_top + stage_bottom) / 2.0
            ax.annotate(
                "",
                xy=(arrow_x2, arrow_y),
                xytext=(arrow_x1, arrow_y),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#38bdf8",
                    lw=2.2,
                    mutation_scale=14
                ),
                zorder=4
            )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Diagram successfully saved to {output_path}")

if __name__ == "__main__":
    generate_architecture_diagram()

