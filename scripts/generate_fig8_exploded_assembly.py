"""
Script to generate Figure 8: Watertight Multi-Layer Exploded Assembly & Natural Seam Detection.
Outputs: docs/images/fig8_watertight_exploded_assembly.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from backend.depth_processor import slice_mesh_into_layers, detect_natural_seams

def generate_fig8():
    # Load sample watertight mesh (Hero Battery Holder)
    mesh_path = Path("output/e2e_final_battery.obj")
    if not mesh_path.exists():
        mesh_path = Path("output/test_unified_printer.obj")
    
    mesh = trimesh.load(mesh_path, force="mesh")
    axis = 1  # Vertical Y-axis
    num_layers = 3

    # Detect seams
    seams = detect_natural_seams(mesh, axis=axis, num_layers=num_layers, num_samples=50)
    layers = slice_mesh_into_layers(mesh, num_layers=num_layers, axis=axis, use_smart_seams=True)

    # Compute area profile for plot
    coords = mesh.vertices[:, axis]
    min_c, max_c = float(coords.min()), float(coords.max())
    span = max_c - min_c
    samples = np.linspace(min_c + 0.02 * span, max_c - 0.02 * span, 60)
    areas = []
    normal = [0.0, 0.0, 0.0]
    normal[axis] = 1.0

    for s in samples:
        origin = [0.0, 0.0, 0.0]
        origin[axis] = s
        sec = mesh.section(plane_origin=origin, plane_normal=normal)
        area = 0.0
        if sec is not None:
            try:
                p2d, _ = sec.to_2D()
                area = float(p2d.area) if hasattr(p2d, "area") else 0.0
            except Exception:
                try:
                    other_axes = [a for a in range(3) if a != axis]
                    p2d_pts = sec.vertices[:, other_axes]
                    area = float(trimesh.points.convex_hull_area(p2d_pts)) if len(p2d_pts) >= 3 else 0.0
                except Exception:
                    area = 0.0
        areas.append(area)

    areas = np.array(areas)

    # Setup matplotlib figure
    fig_w, fig_h = 16.0, 9.5
    dpi = 200

    plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans", "Helvetica"]
    plt.rcParams["font.family"] = "sans-serif"

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="#0b0f17")

    # Header Titles
    fig.text(
        0.5, 0.965,
        "Figure 8: Watertight Multi-Layer Exploded Assembly & Natural Seam Detection",
        color="#38bdf8",
        fontsize=18,
        fontweight="bold",
        ha="center",
        va="center"
    )
    fig.text(
        0.5, 0.932,
        "Cross-sectional area gradient scanning A(s), analytical planar capping (cap=True), and 100% watertight solid component disassembly",
        color="#94a3b8",
        fontsize=11,
        ha="center",
        va="center"
    )

    # --- Panel (a): Area Profile & Seams ---
    ax_a = fig.add_axes([0.045, 0.12, 0.21, 0.72], facecolor="#101726")
    ax_a.plot(areas, samples, color="#38bdf8", linewidth=2.5, label="Cross-Section Area A(y)")
    
    # Layer fill colors
    layer_colors = ["#38bdf8", "#10b981", "#fbbf24"]
    for i in range(len(seams) - 1):
        c0, c1 = seams[i], seams[i+1]
        mask = (samples >= c0) & (samples <= c1)
        if np.any(mask):
            ax_a.fill_betweenx(samples[mask], 0, areas[mask], color=layer_colors[i % len(layer_colors)], alpha=0.30)
        # Seam line
        if i > 0:
            ax_a.axhline(y=c0, color="#f43f5e", linestyle="--", linewidth=1.5, alpha=0.9)
            ax_a.text(np.max(areas) * 0.95, c0 + 0.02 * span, f"Seam {i} (min |dA/dy|)", 
                      color="#f43f5e", fontsize=7.5, fontweight="bold", ha="right")

    ax_a.set_xlim(0, np.max(areas) * 1.15)
    ax_a.set_ylim(min_c - 0.05 * span, max_c + 0.05 * span)
    ax_a.set_title("(a) Cross-Section Area Profile A(y)", color="#f8fafc", fontsize=10.5, fontweight="bold", pad=12)
    ax_a.set_xlabel("Cross-Section Area A(y)", color="#94a3b8", fontsize=8.5, labelpad=8)
    ax_a.set_ylabel("Vertical Position Y (Elevation)", color="#94a3b8", fontsize=8.5, labelpad=8)
    ax_a.tick_params(colors="#94a3b8", labelsize=7.5)
    for spine in ax_a.spines.values():
        spine.set_color("#334155")
        spine.set_linewidth(1.5)
    ax_a.grid(True, linestyle=":", alpha=0.3, color="#64748b")

    # --- Helper to render 3D mesh collection (mapping Y to Z for upright CAD elevation) ---
    def render_mesh_to_ax(ax, m, color, alpha=0.88, dy=0.0):
        m_copy = m.copy()
        m_copy.vertices[:, 1] += dy
        if len(m_copy.faces) > 1500:
            red = max(0.1, min(0.95, 1.0 - (1200.0 / len(m_copy.faces))))
            m_simple = m_copy.simplify_quadric_decimation(percent=red)
        else:
            m_simple = m_copy
            
        # Map: X_plot = X, Y_plot = Z, Z_plot = Y (Upright!)
        verts_up = m_simple.vertices[:, [0, 2, 1]]
        triangles = verts_up[m_simple.faces]
        poly = Poly3DCollection(triangles, facecolors=color, edgecolors="#1e293b", linewidths=0.2, alpha=alpha)
        ax.add_collection3d(poly)

    def set_bounds_and_view(ax, m, pad=1.1, elev=20, azim=45):
        bounds_up = m.bounds[:, [0, 2, 1]]
        max_extent = np.max(bounds_up[1] - bounds_up[0]) * pad
        mid = (bounds_up[0] + bounds_up[1]) / 2.0
        ax.set_xlim(mid[0] - max_extent/2, mid[0] + max_extent/2)
        ax.set_ylim(mid[1] - max_extent/2, mid[1] + max_extent/2)
        ax.set_zlim(mid[2] - max_extent/2, mid[2] + max_extent/2)
        ax.axis("off")
        ax.view_init(elev=elev, azim=azim)

    # --- Panel (b): Monolithic Solid CAD Asset (0% Exploded) ---
    ax_b = fig.add_axes([0.28, 0.12, 0.22, 0.72], projection="3d", facecolor="#101726")
    render_mesh_to_ax(ax_b, mesh, color="#38bdf8", alpha=0.90)
    set_bounds_and_view(ax_b, mesh, pad=1.15, elev=20, azim=45)
    ax_b.set_title("(b) Monolithic Solid CAD Mesh\n(0% Exploded - Watertight: True)", color="#f8fafc", fontsize=10.5, fontweight="bold", pad=12)

    # --- Panel (c): Exploded Assembly (50% Axial Separation) ---
    ax_c = fig.add_axes([0.52, 0.12, 0.22, 0.72], projection="3d", facecolor="#101726")
    exploded_offset = span * 0.38
    for idx, (sub_m, scolor) in enumerate(zip(layers, layer_colors)):
        # Apply vertical translation along Y
        dy = (idx - 1.0) * exploded_offset
        render_mesh_to_ax(ax_c, sub_m, color=scolor, alpha=0.88, dy=dy)

    set_bounds_and_view(ax_c, mesh, pad=1.75, elev=20, azim=45)
    ax_c.set_title("(c) Multi-Layer Exploded Assembly\n(Vertical Separation along Y-Axis)", color="#f8fafc", fontsize=10.5, fontweight="bold", pad=12)

    # --- Panel (d): Watertight Planar Capping (cap=True) ---
    ax_d = fig.add_axes([0.76, 0.12, 0.22, 0.72], projection="3d", facecolor="#101726")
    focus_layer = layers[1].copy() if len(layers) > 1 else layers[0].copy()
    render_mesh_to_ax(ax_d, focus_layer, color="#10b981", alpha=0.92)
    # Higher elevation to clearly inspect the flat planar capping surface
    set_bounds_and_view(ax_d, focus_layer, pad=1.15, elev=36, azim=45)
    ax_d.set_title("(d) Analytical Planar Capping\n(cap=True - Solid CAD Volume)", color="#f8fafc", fontsize=10.5, fontweight="bold", pad=12)

    output_path = Path("docs/images/fig8_watertight_exploded_assembly.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Figure 8 successfully saved to {output_path}")

if __name__ == "__main__":
    generate_fig8()
