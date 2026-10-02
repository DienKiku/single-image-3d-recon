import argparse
import os
import sys
import time
import numpy as np
from PIL import Image

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from depth import estimate_depth
from mesh import create_relief_mesh


def convert(image_path: str, out_path: str, resolution: int = 512,
            depth_scale: float = 0.35, smooth: int = 3, solid: bool = False,
            model_name: str = "MiDaS_small", device: str = "auto") -> str:
    """Dựng phù điêu 2.5D từ ảnh phẳng (chế độ relief)."""
    t0 = time.time()

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > resolution:
        scale = resolution / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)

    img_np = np.array(img)
    depth_map = estimate_depth(img_np, model_name=model_name, device=device)

    mesh = create_relief_mesh(depth_map, img_np, depth_scale=depth_scale,
                              smooth=smooth, solid=solid)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    mesh.export(out_path)

    print(f"[OK Relief] {out_path} | {len(mesh.vertices)} verts | {len(mesh.faces)} faces "
          f"| watertight={mesh.is_watertight} | {time.time() - t0:.1f}s")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Dựng phù điêu 2.5D từ ảnh (MiDaS / DPT)")
    p.add_argument("image")
    p.add_argument("-o", "--output", default="outputs/relief.glb")
    p.add_argument("-r", "--resolution", type=int, default=512)
    p.add_argument("-d", "--depth-scale", type=float, default=0.35)
    p.add_argument("-s", "--smooth", type=int, default=3)
    p.add_argument("--solid", action="store_true", help="tạo khối đặc cho in 3D")
    p.add_argument("-m", "--model", default="MiDaS_small")
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    convert(a.image, a.output, a.resolution, a.depth_scale, a.smooth, a.solid, a.model, a.device)
