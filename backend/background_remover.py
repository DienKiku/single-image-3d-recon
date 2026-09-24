"""Background removal module using offline U2-Net via rembg / onnxruntime.

Provides clean foreground isolation for arbitrary objects (electronics,
furniture, vehicles, apparel, etc.) without requiring cloud APIs.
"""

from typing import Tuple, Optional
import numpy as np
import cv2
from PIL import Image

_REMBG_SESSION = None

def get_rembg_session(model_name: str = "u2net"):
    """Get or lazily initialize the rembg ONNX session."""
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        try:
            import rembg
            _REMBG_SESSION = rembg.new_session(model_name)
        except Exception as e:
            print(f"Warning: Failed to initialize rembg session: {e}")
            _REMBG_SESSION = None
    return _REMBG_SESSION


def remove_background(
    image: np.ndarray,
    model_name: str = "u2net",
    alpha_threshold: int = 128
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Remove background from an image, isolating the foreground object.
    
    Args:
        image: Input image as RGB or RGBA uint8 numpy array.
        model_name: ONNX model name (default: 'u2net').
        alpha_threshold: Cutoff value for foreground mask (0-255).
        
    Returns:
        Tuple of:
            - rgba_image: Shape (H, W, 4) uint8 array with transparent background.
            - binary_mask: Shape (H, W) uint8 array (255 for foreground, 0 for background).
            - bbox: (x, y, width, height) tight bounding box of the foreground object.
    """
    if image.ndim == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image_rgb = image[:, :, :3]
    else:
        image_rgb = image.copy()
        
    h, w = image_rgb.shape[:2]
    
    # 1. Try deep learning background removal via rembg (U2-Net)
    session = get_rembg_session(model_name)
    if session is not None:
        try:
            import rembg
            pil_img = Image.fromarray(image_rgb)
            out_pil = rembg.remove(pil_img, session=session)
            out_rgba = np.array(out_pil, dtype=np.uint8)
            alpha = out_rgba[:, :, 3]
            binary_mask = (alpha >= alpha_threshold).astype(np.uint8) * 255
            
            # Find bounding box
            ys, xs = np.where(binary_mask > 0)
            if len(xs) > 50:  # Valid foreground detected
                min_x, max_x = int(xs.min()), int(xs.max())
                min_y, max_y = int(ys.min()), int(ys.max())
                bbox = (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
                
                # Clean edges: zero out RGB where alpha is 0
                out_rgba[:, :, :3] = np.where(out_rgba[:, :, 3:4] > 0, out_rgba[:, :, :3], 0)
                return out_rgba, binary_mask, bbox
        except Exception as e:
            print(f"rembg processing error: {e}. Falling back to classical saliency.")

    # 2. Fallback: Classical Saliency / GrabCut segmentation
    from backend.depth_processor import detect_salient_foreground_box
    bx, by, bw, bh = detect_salient_foreground_box(image_rgb)
    
    binary_mask = np.zeros((h, w), dtype=np.uint8)
    binary_mask[by:by+bh, bx:bx+bw] = 255
    
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = image_rgb
    rgba[:, :, 3] = binary_mask
    
    return rgba, binary_mask, (bx, by, bw, bh)


def crop_to_foreground(
    rgba_image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: int = 4
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop an RGBA image tightly around the bounding box with optional padding.
    
    Args:
        rgba_image: Input RGBA image.
        bbox: (x, y, w, h) of foreground.
        padding: Padding in pixels.
        
    Returns:
        Tuple of (cropped_rgba, adjusted_bbox)
    """
    h, w = rgba_image.shape[:2]
    x, y, bw, bh = bbox
    
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + bw + padding)
    y2 = min(h, y + bh + padding)
    
    cropped = rgba_image[y1:y2, x1:x2].copy()
    actual_bbox = (x1, y1, x2 - x1, y2 - y1)
    return cropped, actual_bbox
