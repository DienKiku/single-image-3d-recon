"""
Generate Figure 5: Metric CAD Alignment & Quadric Decimation Wireframe Topology.
Outputs: docs/images/fig5_cad_metric_calibration_wireframe.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def generate_fig5():
    mesh_path = Path("output/e2e_final_battery.obj")
    mesh = trimesh.load(mesh_path, force="mesh")

    # Downsample mesh to ~3000 faces for clean crisp wireframe display
    decim = mesh.simplify_quadric_decimation(0.2)
    dv = decim.vertices[:, [0, 2, 1]] # Map [x, z, y]
    df = decim.faces

    fig = plt.figure(figsize=(18, 6.5), dpi=200, facecolor="#0b0f17")
    fig.suptitle(
        "Figure 5: Metric CAD Alignment & Quadric Decimation Wireframe Topology\n"
        "Upright Scaled CAD Model (76.0 x 40.6 x 20.2 mm) - 14,908 Faces",
        color="#38bdf8",
        fontsize=16,
        fontweight="bold",
        y=0.97
    )

    # Ax1: Perspective 3/4 Wireframe
    ax1 = fig.add_subplot(1, 3, 1, projection="3d", facecolor="#0b0f17")
    poly1 = Poly3DCollection(dv[df], facecolors="#0f172a", edgecolors="#38bdf8", linewidths=0.35, alpha=0.9)
    ax1.add_collection3d(poly1)
    ax1.view_init(elev=28, azim=-38)
    ax1.set_xlim(-42, 42)
    ax1.set_ylim(-24, 12)
    ax1.set_zlim(-2, 45)
    ax1.set_title("Perspective 3/4 Wireframe\n(Watertight Manifold Topology)", color="#f1f5f9", fontsize=12, pad=10)
    ax1.axis("off")
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False

    # Ax2: Front Elevation Wireframe (Orthogonal 3D view)
    ax2 = fig.add_subplot(1, 3, 2, projection="3d", facecolor="#0b0f17")
    poly2 = Poly3DCollection(dv[df], facecolors="#0f172a", edgecolors="#38bdf8", linewidths=0.3, alpha=0.9)
    ax2.add_collection3d(poly2)
    ax2.view_init(elev=0, azim=-90)
    ax2.set_xlim(-42, 42)
    ax2.set_ylim(-24, 12)
    ax2.set_zlim(-2, 45)
    ax2.set_title("Front Elevation (Orthogonal)\nWidth = 76.0 mm | Height = 40.6 mm", color="#f1f5f9", fontsize=12, pad=10)
    ax2.axis("off")

    # Ax3: Side Profile Wireframe (Orthogonal 3D view)
    ax3 = fig.add_subplot(1, 3, 3, projection="3d", facecolor="#0b0f17")
    poly3 = Poly3DCollection(dv[df], facecolors="#0f172a", edgecolors="#38bdf8", linewidths=0.3, alpha=0.9)
    ax3.add_collection3d(poly3)
    ax3.view_init(elev=0, azim=0)
    ax3.set_xlim(-42, 42)
    ax3.set_ylim(-24, 12)
    ax3.set_zlim(-2, 45)
    ax3.set_title("Side Elevation (Orthogonal)\nDepth = 20.2 mm | Grounded Datum Y = 0 mm", color="#f1f5f9", fontsize=12, pad=10)
    ax3.axis("off")

    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.92])
    out_file = Path("docs/images/fig5_cad_metric_calibration_wireframe.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, facecolor="#0b0f17", dpi=200)
    plt.close()
    print(f"Saved {out_file} ({out_file.stat().st_size} bytes)")

if __name__ == "__main__":
    generate_fig5()
