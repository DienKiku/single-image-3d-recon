import os
import cv2
import numpy as np
import torch
from PIL import Image

_MODELS = {}

# Đảm bảo torch.hub không hỏi xác thực tương tác làm treo server
if hasattr(torch.hub, "_TRUSTED_REPO_OWNERS"):
    owners = set(torch.hub._TRUSTED_REPO_OWNERS)
    owners.update(["rwightman", "intel-isl", "facebookresearch"])
    torch.hub._TRUSTED_REPO_OWNERS = tuple(owners)


def get_depth_model(model_name="MiDaS_small", device="auto"):
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    key = (model_name, str(dev))
    if key in _MODELS:
        return _MODELS[key]

    model = None
    transform = None

    # 1. Ưu tiên Depth-Anything-V2 nếu người dùng chọn model hoặc khi cần chất lượng cao và đã tải sẵn
    try:
        from transformers import pipeline
        model = pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=0 if dev.type == "cuda" else -1
        )
        transform = "pipeline"
        _MODELS[key] = (model, transform, dev)
        return _MODELS[key]
    except Exception as e_da:
        print(f"[INFO] Depth Anything V2 không nạp được ({e_da}), thử MiDaS...")

    # 2. Thử MiDaS từ torch.hub
    try:
        model = torch.hub.load("intel-isl/MiDaS", model_name, trust_repo=True)
        model.to(dev)
        model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if model_name in ("DPT_Large", "DPT_Hybrid"):
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform
    except Exception as e:
        print(f"[WARN] MiDaS không tải được ({e}). Sẽ dùng bộ ước tính hình học fallback.")
        model = "fallback"

    _MODELS[key] = (model, transform, dev)
    return _MODELS[key]


def estimate_depth(image: np.ndarray, model_name="MiDaS_small", device="auto") -> np.ndarray:
    """Trả về ma trận độ sâu chuẩn hoá [0, 1] dạng float32."""
    h, w = image.shape[:2]
    model, transform, dev = get_depth_model(model_name, device)

    if model == "fallback" or model is None:
        return _heuristic_depth(image)

    try:
        if transform == "pipeline":
            pil_img = Image.fromarray(image)
            res = model(pil_img)
            depth = np.array(res["depth"]).astype(np.float32)
            if depth.shape[:2] != (h, w):
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            input_batch = transform(image).to(dev)
            with torch.no_grad():
                prediction = model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(h, w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            depth = prediction.cpu().numpy()

        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth_norm = (depth - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth)
        return depth_norm.astype(np.float32)
    except Exception as e:
        print(f"[WARN] Lỗi suy luận độ sâu ({e}), dùng bộ ước tính hình học...")
        return _heuristic_depth(image)


def _heuristic_depth(image: np.ndarray) -> np.ndarray:
    """Bộ ước tính hình học dự phòng khi không có mạng nơ-ron."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    texture = np.sqrt(grad_x ** 2 + grad_y ** 2)
    texture = cv2.GaussianBlur(texture, (21, 21), 0)
    if texture.max() > 0:
        texture /= texture.max()

    y, x = np.mgrid[0:h, 0:w]
    dist = np.sqrt((x - w / 2.0) ** 2 + (y - h / 2.0) ** 2)
    max_d = np.max(dist)
    convex = 1.0 - (dist / max_d) ** 1.5 if max_d > 0 else np.ones_like(gray)

    shading = cv2.GaussianBlur(gray, (51, 51), 0)
    if shading.max() > shading.min():
        shading = (shading - shading.min()) / (shading.max() - shading.min())

    combined = 0.55 * convex + 0.30 * texture + 0.15 * shading
    combined = cv2.GaussianBlur(combined, (11, 11), 0)
    c_min, c_max = combined.min(), combined.max()
    if c_max > c_min:
        return ((combined - c_min) / (c_max - c_min)).astype(np.float32)
    return np.zeros((h, w), dtype=np.float32)
