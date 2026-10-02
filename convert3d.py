import argparse
import os
import sys
import time

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from tripo_engine import preprocess, reconstruct
import postprocess as pp


def convert3d(image_path, out_path, mc_resolution=256, foreground_ratio=0.85,
              remove_bg=True, target_faces=0, smooth_iter=0, bake_uv=False,
              device="auto", chunk_size=8192):
    t0 = time.time()

    img = preprocess(image_path, foreground_ratio, remove_bg)
    mesh = reconstruct(img, mc_resolution, device, chunk_size)

    mesh = pp.orient(mesh)
    mesh = pp.clean(mesh)
    mesh = pp.smooth(mesh, smooth_iter)
    mesh = pp.decimate(mesh, target_faces)
    mesh = pp.normalize(mesh)

    ext = os.path.splitext(out_path)[1].lower()
    if bake_uv and ext in (".glb", ".gltf", ".obj"):
        mesh = pp.to_uv_textured(mesh)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    pp.export(mesh, out_path)

    print(f"[OK] {out_path} | {len(mesh.vertices)} verts | {len(mesh.faces)} faces "
          f"| watertight={mesh.is_watertight} | {time.time() - t0:.1f}s")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Dựng mô hình 3D đầy đủ từ 1 ảnh (TripoSR)")
    p.add_argument("image")
    p.add_argument("-o", "--output", default="outputs/model3d.glb")
    p.add_argument("-r", "--mc-resolution", type=int, default=256,
                   help="128 nhanh / 256 chuẩn / 320-384 chi tiết cao")
    p.add_argument("-fg", "--foreground-ratio", type=float, default=0.85)
    p.add_argument("--keep-bg", action="store_true", help="không tách nền")
    p.add_argument("-f", "--faces", type=int, default=0, help="giảm còn N mặt")
    p.add_argument("-s", "--smooth", type=int, default=0)
    p.add_argument("--bake-uv", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--chunk", type=int, default=8192)
    a = p.parse_args()

    convert3d(a.image, a.output, a.mc_resolution, a.foreground_ratio,
              not a.keep_bg, a.faces, a.smooth, a.bake_uv, a.device, a.chunk)
