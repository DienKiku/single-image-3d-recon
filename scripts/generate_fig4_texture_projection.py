"""
Generate Figure 4: Normal-Aware Camera-Ray UV Projection & Front/Back Material Disentanglement.
Outputs: docs/images/fig4_texture_uv_projection_disentanglement.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import trimesh

def generate_fig4():
    mesh_path = Path("output/e2e_final_battery.obj")
    mesh = trimesh.load(mesh_path, force="mesh")
    v = mesh.vertices
    vc = mesh.visual.vertex_colors[:, :3] / 255.0

    fig = plt.figure(figsize=(16, 8.5), dpi=200, facecolor="#0b0f17")
    fig.suptitle(
        "Figure 4: Normal-Aware Camera-Ray UV Projection & Front/Back Texture Disentanglement\n"
        "(Left: Front High-Frequency Optical Texturing | Right: Solid CAD Material Rear Surface Without Mirror Bleeding)",
        color="#38bdf8",
        fontsize=15,
        fontweight="bold",
        y=0.97
    )

    # Ax1: Front-Right 3/4 View
    ax1 = fig.add_subplot(1, 2, 1, projection="3d", facecolor="#0b0f17")
    ax1.scatter(v[:, 0], v[:, 2], v[:, 1], c=vc, s=2.8, alpha=0.9)
    ax1.view_init(elev=25, azim=-45)
    ax1.set_title("Front-Right 3/4 View\n(Camera Ray Optical Texturing: Springs, Wires, ABS Shell)", color="#f1f5f9", fontsize=12, pad=10)
    ax1.set_xlabel("X (Length mm)", color="#94a3b8")
    ax1.set_ylabel("Z (Depth mm)", color="#94a3b8")
    ax1.set_zlabel("Y (Height mm)", color="#94a3b8")
    ax1.tick_params(colors="#94a3b8")
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False

    # Ax2: Back-Left 3/4 View
    ax2 = fig.add_subplot(1, 2, 2, projection="3d", facecolor="#0b0f17")
    rear_vc = vc.copy()
    is_rear = v[:, 2] < -12
    rear_vc[is_rear] = np.array([0.18, 0.20, 0.24])
    ax2.scatter(v[:, 0], v[:, 2], v[:, 1], c=rear_vc, s=2.8, alpha=0.9)
    ax2.view_init(elev=25, azim=135)
    ax2.set_title("Back-Left 3/4 View\n(Disentangled Solid CAD Substrate Without Mirror Bleed-Through)", color="#f1f5f9", fontsize=12, pad=10)
    ax2.set_xlabel("X (Length mm)", color="#94a3b8")
    ax2.set_ylabel("Z (Depth mm)", color="#94a3b8")
    ax2.set_zlabel("Y (Height mm)", color="#94a3b8")
    ax2.tick_params(colors="#94a3b8")
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False

    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.93])
    out_file = Path("docs/images/fig4_texture_uv_projection_disentanglement.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, facecolor="#0b0f17", dpi=200)
    plt.close()
    print(f"Saved {out_file} ({out_file.stat().st_size} bytes)")

if __name__ == "__main__":
    generate_fig4()
