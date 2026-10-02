import os
import sys
import numpy as np
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
TSR_SRC = os.path.join(BASE, "TripoSR")
MODEL_DIR = os.path.join(BASE, "models", "TripoSR")

if TSR_SRC not in sys.path:
    sys.path.insert(0, TSR_SRC)

# Ưu tiên cache offline
os.environ.setdefault("HF_HOME", os.path.join(BASE, "models", "hf"))

_ENGINE = {}
_REMBG = {}


def _get_device(prefer: str = "auto"):
    import torch
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" or (prefer == "auto" and torch.cuda.is_available()):
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


def load_engine(device_pref: str = "auto", chunk_size: int = 8192):
    """Nạp TripoSR một lần duy nhất rồi giữ trong RAM/VRAM."""
    if not chunk_size or chunk_size <= 0:
        chunk_size = 8192

    key = (device_pref, chunk_size)
    if key in _ENGINE:
        return _ENGINE[key]

    import mc_fallback          # vá marching cubes TRƯỚC khi import tsr
    mc_fallback.apply_patch()

    import torch
    from tsr.system import TSR

    device = _get_device(device_pref)

    src = MODEL_DIR if os.path.isdir(MODEL_DIR) else "stabilityai/TripoSR"
    model = TSR.from_pretrained(
        src,
        config_name="config.yaml",
        weight_name="model.ckpt",
    )

    # QUAN TRỌNG: Luôn bật chunk_size (mặc định 8192) trên cả CPU lẫn GPU.
    # Nếu chunk_size = 0, mô hình sẽ nạp toàn bộ 16.7 triệu điểm (256^3) vào 1 tensor,
    # gây lỗi: "DefaultCPUAllocator: not enough memory: you tried to allocate 4294967296 bytes".
    model.renderer.set_chunk_size(chunk_size)
    model.to(device)
    model.eval()

    _ENGINE[key] = (model, device, torch)
    return _ENGINE[key]


def _rembg_session():
    if "s" not in _REMBG:
        import rembg
        u2net_path = os.path.join(BASE, "models", "u2net")
        if os.path.isdir(u2net_path):
            os.environ.setdefault("U2NET_HOME", u2net_path)
        _REMBG["s"] = rembg.new_session("u2net")
    return _REMBG["s"]


def preprocess(image_path: str, foreground_ratio: float = 0.85,
               do_remove_bg: bool = True) -> Image.Image:
    """Tách nền, canh giữa chủ thể, nền xám trung tính (yêu cầu của TripoSR)."""
    from tsr.utils import remove_background, resize_foreground

    img = Image.open(image_path)

    if do_remove_bg:
        img = remove_background(img.convert("RGB"), _rembg_session())
        img = resize_foreground(img, foreground_ratio)
        arr = np.array(img).astype(np.float32) / 255.0
        if arr.shape[-1] == 4:
            arr = arr[:, :, :3] * arr[:, :, 3:4] + 0.5 * (1 - arr[:, :, 3:4])
        img = Image.fromarray((arr * 255.0).astype(np.uint8))
    else:
        img = img.convert("RGB")

    return img


def reconstruct(image: Image.Image, mc_resolution: int = 256,
                device_pref: str = "auto", chunk_size: int = 8192):
    """Trả về trimesh.Trimesh có vertex color."""
    if not chunk_size or chunk_size <= 0:
        chunk_size = 8192

    model, device, torch = load_engine(device_pref, chunk_size)
    model.renderer.set_chunk_size(chunk_size)

    with torch.no_grad():
        scene_codes = model([image], device=device)

    try:
        meshes = model.extract_mesh(scene_codes, True, resolution=mc_resolution)
    except TypeError:  # tương thích bản tsr cũ
        meshes = model.extract_mesh(scene_codes, resolution=mc_resolution)

    return meshes[0]
