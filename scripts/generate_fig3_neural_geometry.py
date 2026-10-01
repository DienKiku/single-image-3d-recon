"""
Generate Figure 3: Multi-View Omnidirectional Neural 3D Geometry Reconstruction.
Outputs: docs/images/fig3_neural_geometry_multiview.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def generate_fig3():
    mesh_path = Path("output/e2e_final_battery.obj")
    mesh = trimesh.load(mesh_path, force="mesh")
    if len(mesh.faces) > 8000:
        mesh = mesh.simplify_quadric_decimation(0.6)

    # Map [x, z, y] so Y is vertical elevation
    vertices = mesh.vertices[:, [0, 2, 1]]
    faces = mesh.faces

    # Face normals in mapped coordinate system
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    normals = normals / norm

    # Diffuse lighting
    light_dir = np.array([0.35, -0.45, 0.8])
    light_dir = light_dir / np.linalg.norm(light_dir)

    intensity = np.clip(np.abs(np.dot(normals, light_dir)), 0.25, 1.0)
    base_col = np.array([0.68, 0.74, 0.82])
    face_colors = intensity[:, None] * base_col

    views = [
        ("Chính diện (Front View)", 20, -70),
        ("Góc nghiêng 3/4 (Isometric 3/4)", 30, -35),
        ("Cạnh bên (Side Profile)", 15, 20),
        ("Từ trên nhìn xuống (Top View)", 85, -90),
        ("Mặt sau (Back View)", 20, 110),
        ("Góc nghiêng mặt sau (Rear 3/4)", 30, 145)
    ]

    fig = plt.figure(figsize=(16, 10.5), dpi=200, facecolor="#0b0f17")
    fig.suptitle(
        "Figure 3: Multi-View Omnidirectional Neural 3D Geometry Reconstruction",
        color="#38bdf8",
        fontsize=18,
        fontweight="bold",
        y=0.97
    )

    for idx, (title, elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 3, idx + 1, projection="3d", facecolor="#0b0f17")
        poly = Poly3DCollection(vertices[faces], facecolors=face_colors, edgecolors="#1e293b", linewidths=0.15)
        ax.add_collection3d(poly)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(-45, 45)
        ax.set_ylim(-25, 15)
        ax.set_zlim(-5, 45)
        ax.set_title(title, color="#f1f5f9", fontsize=12, fontweight="bold", pad=8)
        ax.axis("off")
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.94])
    out_file = Path("docs/images/fig3_neural_geometry_multiview.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, facecolor="#0b0f17", dpi=200)
    plt.close()
    print(f"Saved {out_file} ({out_file.stat().st_size} bytes)")

if __name__ == "__main__":
    generate_fig3()
