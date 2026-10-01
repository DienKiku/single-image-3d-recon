"""
Generate Figure 2: Spatial Coordinate Distribution and Multi-Plane Projection Analysis.
Outputs: docs/images/fig2_spatial_coordinate_analysis.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import trimesh

def generate_fig2():
    mesh_path = Path("output/e2e_final_battery.obj")
    mesh = trimesh.load(mesh_path, force="mesh")
    pts = mesh.vertices

    # Subsample points for crisp rendering
    if len(pts) > 4500:
        idx = np.random.choice(len(pts), 4500, replace=False)
        pts = pts[idx]

    fig = plt.figure(figsize=(16, 10.5), dpi=200, facecolor="#0b0f17")
    fig.suptitle(
        "Figure 2: Spatial Coordinate Distribution and Multi-Plane Projection Analysis",
        color="#38bdf8",
        fontsize=18,
        fontweight="bold",
        y=0.97
    )

    colors = pts[:, 2]

    # Subplot 1: Front (X-Y)
    ax1 = fig.add_subplot(2, 3, 1, facecolor="#0f172a")
    ax1.scatter(pts[:, 0], pts[:, 1], c=colors, cmap="viridis", s=2.5, alpha=0.85)
    ax1.set_title("(a) Front View (X-Y Plane)", color="#f1f5f9", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("X (Length mm)", color="#94a3b8")
    ax1.set_ylabel("Y (Height mm)", color="#94a3b8")
    ax1.tick_params(colors="#94a3b8")
    ax1.grid(True, linestyle="--", alpha=0.2, color="#38bdf8")

    # Subplot 2: Side Profile (Z-Y)
    ax2 = fig.add_subplot(2, 3, 2, facecolor="#0f172a")
    ax2.scatter(pts[:, 2], pts[:, 1], c=colors, cmap="viridis", s=2.5, alpha=0.85)
    ax2.set_title("(b) Side Profile (Z-Y Plane)", color="#f1f5f9", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Z (Depth mm)", color="#94a3b8")
    ax2.set_ylabel("Y (Height mm)", color="#94a3b8")
    ax2.tick_params(colors="#94a3b8")
    ax2.grid(True, linestyle="--", alpha=0.2, color="#38bdf8")

    # Subplot 3: 3D Isometric
    ax3 = fig.add_subplot(2, 3, 3, projection="3d", facecolor="#0b0f17")
    ax3.scatter(pts[:, 0], pts[:, 2], pts[:, 1], c=colors, cmap="viridis", s=2.0, alpha=0.85)
    ax3.set_title("(c) Isometric 3D Spatial Field", color="#f1f5f9", fontsize=12, fontweight="bold", pad=10)
    ax3.set_xlabel("X (mm)", color="#94a3b8")
    ax3.set_ylabel("Z (mm)", color="#94a3b8")
    ax3.set_zlabel("Y (mm)", color="#94a3b8")
    ax3.tick_params(colors="#94a3b8")
    ax3.xaxis.pane.fill = False
    ax3.yaxis.pane.fill = False
    ax3.zaxis.pane.fill = False

    # Subplot 4: Rear Elevation (-X-Y)
    ax4 = fig.add_subplot(2, 3, 4, facecolor="#0f172a")
    ax4.scatter(-pts[:, 0], pts[:, 1], c=colors, cmap="viridis", s=2.5, alpha=0.85)
    ax4.set_title("(d) Rear Elevation (-X-Y Plane)", color="#f1f5f9", fontsize=12, fontweight="bold", pad=10)
    ax4.set_xlabel("-X (Reverse Length mm)", color="#94a3b8")
    ax4.set_ylabel("Y (Height mm)", color="#94a3b8")
    ax4.tick_params(colors="#94a3b8")
    ax4.grid(True, linestyle="--", alpha=0.2, color="#38bdf8")

    # Subplot 5: Top-Down Plan (X-Z)
    ax5 = fig.add_subplot(2, 3, 5, facecolor="#0f172a")
    ax5.scatter(pts[:, 0], pts[:, 2], c=pts[:, 1], cmap="plasma", s=2.5, alpha=0.85)
    ax5.set_title("(e) Top-Down Plan (X-Z Plane)", color="#f1f5f9", fontsize=12, fontweight="bold", pad=10)
    ax5.set_xlabel("X (Length mm)", color="#94a3b8")
    ax5.set_ylabel("Z (Depth mm)", color="#94a3b8")
    ax5.tick_params(colors="#94a3b8")
    ax5.grid(True, linestyle="--", alpha=0.2, color="#38bdf8")

    # Subplot 6: Bottom-Up View (X--Z)
    ax6 = fig.add_subplot(2, 3, 6, facecolor="#0f172a")
    ax6.scatter(pts[:, 0], -pts[:, 2], c=pts[:, 1], cmap="plasma", s=2.5, alpha=0.85)
    ax6.set_title("(f) Bottom-Up View (X--Z Plane)", color="#f1f5f9", fontsize=12, fontweight="bold", pad=10)
    ax6.set_xlabel("X (Length mm)", color="#94a3b8")
    ax6.set_ylabel("-Z (Inverted Depth mm)", color="#94a3b8")
    ax6.tick_params(colors="#94a3b8")
    ax6.grid(True, linestyle="--", alpha=0.2, color="#38bdf8")

    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.94])
    out_file = Path("docs/images/fig2_spatial_coordinate_analysis.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, facecolor="#0b0f17", dpi=200)
    plt.close()
    print(f"Saved {out_file} ({out_file.stat().st_size} bytes)")

if __name__ == "__main__":
    generate_fig2()
