"""
Master script to regenerate all documentation figures (Figures 1-9) with unified hero samples.
Outputs: docs/images/fig1_*.png through fig9_*.png
"""

import sys
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_fig1_preprocessing import generate_fig1
from scripts.generate_fig2_spatial_analysis import generate_fig2
from scripts.generate_fig3_neural_geometry import generate_fig3
from scripts.generate_fig4_texture_projection import generate_fig4
from scripts.generate_fig5_metric_wireframe import generate_fig5
from scripts.generate_fig6_pbr_maps import generate_fig6
from scripts.generate_fig7_dual_hemisphere import generate_fig7
from scripts.generate_fig8_exploded_assembly import generate_fig8
from scripts.generate_fig9_in_the_wild_showcase import generate_fig9

def main():
    print("=" * 70)
    print("REGENERATING ALL PIPELINE DOCUMENTATION FIGURES (FIG 1 - FIG 9)")
    print("Dual Showcase: Hero Battery Holder (Fig 1-8) + Copier Benchmark (Fig 9)")
    print("=" * 70)

    stages = [
        ("Figure 1 (Preprocessing & Depth)", generate_fig1),
        ("Figure 2 (Spatial Projections)", generate_fig2),
        ("Figure 3 (Neural 3D Geometry)", generate_fig3),
        ("Figure 4 (Camera Ray UV & Disentanglement)", generate_fig4),
        ("Figure 5 (Metric CAD Wireframe)", generate_fig5),
        ("Figure 6 (6-Channel PBR Baking)", generate_fig6),
        ("Figure 7 (Dual-Hemisphere UV Atlas)", generate_fig7),
        ("Figure 8 (Watertight Exploded Assembly)", generate_fig8),
        ("Figure 9 (In-The-Wild Industrial Benchmark)", generate_fig9),
    ]

    t0 = time.time()
    for name, fn in stages:
        s_t0 = time.time()
        print(f"\n>>> Running {name}...")
        try:
            fn()
            print(f">>> Completed {name} in {time.time() - s_t0:.2f}s")
        except Exception as e:
            print(f"!!! Error in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"ALL FIGURES COMPLETED IN {time.time() - t0:.2f}s")
    print("=" * 70)

if __name__ == "__main__":
    main()
