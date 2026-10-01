"""Physically Based Rendering (PBR) Texture & Material Baking Engine.

Generates complete, high-fidelity PBR material maps (100% local and CPU-efficient):
  - Base Color / Diffuse Albedo (Illumination-delighted)
  - Tangent-Space Normal Map (Multi-scale Sobel surface curvature & micro-relief)
  - Roughness Map (Physics-aware specular & texture entropy classification)
  - Metallic Map (Conductive vs dielectric material segmentation)
  - Ambient Occlusion (AO) Map (Cavity & crevice contact shadows)
  - Packed ORM Map (glTF 2.0 standard: R=Occlusion, G=Roughness, B=Metallic)
  - High-Resolution Industrial Chassis Rear Panel Synthesizer (Realistic back surface)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger("PBRBaker")


class PBRBaker:
    """Full-featured local PBR material baker and rear chassis texture synthesizer."""

    @staticmethod
    def extract_albedo(image_rgb: np.ndarray, delight_strength: float = 0.4) -> np.ndarray:
        """Extract intrinsic Base Color (Albedo) by separating directional illumination.
        
        Uses guided bilateral filtering on luminance to suppress harsh directional
        shading gradients while preserving sharp optical pigments and text.
        """
        img_float = image_rgb.astype(np.float32) / 255.0
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l_chan = lab[:, :, 0].astype(np.float32) / 255.0

        # Large-kernel bilateral filter captures broad illumination gradient
        illumination = cv2.bilateralFilter(l_chan, d=15, sigmaColor=0.3, sigmaSpace=15)
        mean_l = float(np.mean(illumination))
        if mean_l < 1e-4:
            mean_l = 0.5

        # Ratio delighting with smooth clamping
        delight_ratio = (mean_l / (illumination + 0.15)) ** delight_strength
        delight_ratio = np.clip(delight_ratio, 0.70, 1.35)

        albedo = img_float * delight_ratio[:, :, np.newaxis]
        albedo = np.clip(albedo * 255.0, 0, 255).astype(np.uint8)
        return albedo

    @staticmethod
    def generate_normal_map(
        image_rgb: np.ndarray,
        intensity: float = 2.5,
        depth_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute tangent-space Normal Map (RGB standard) combining multi-scale gradients.
        
        R = Normal X (Left-to-Right), G = Normal Y (Bottom-to-Top, OpenGL standard), B = Normal Z (Outward).
        """
        h, w = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        # Fine details: High-pass Sobel on luminance
        blur_fine = cv2.GaussianBlur(gray, (3, 3), 0)
        gx_fine = cv2.Sobel(blur_fine, cv2.CV_32F, 1, 0, ksize=3)
        gy_fine = cv2.Sobel(blur_fine, cv2.CV_32F, 0, 1, ksize=3)

        # Medium structure: Slightly blurred Sobel
        blur_med = cv2.GaussianBlur(gray, (7, 7), 0)
        gx_med = cv2.Sobel(blur_med, cv2.CV_32F, 1, 0, ksize=5)
        gy_med = cv2.Sobel(blur_med, cv2.CV_32F, 0, 1, ksize=5)

        # Combine fine micro-relief with structural contours
        gx = (gx_fine * 0.7 + gx_med * 0.3) * intensity
        # OpenGL tangent space has Y pointing UP, so invert image row gradient (Y downwards)
        gy = -(gy_fine * 0.7 + gy_med * 0.3) * intensity

        # Incorporate depth map gradients if available
        if depth_map is not None:
            dm = depth_map
            if dm.shape[:2] != (h, w):
                dm = cv2.resize(dm, (w, h), interpolation=cv2.INTER_LINEAR)
            dm_blur = cv2.GaussianBlur(dm.astype(np.float32), (5, 5), 0)
            d_gx = cv2.Sobel(dm_blur, cv2.CV_32F, 1, 0, ksize=3) * 3.0
            d_gy = -cv2.Sobel(dm_blur, cv2.CV_32F, 0, 1, ksize=3) * 3.0
            gx = gx * 0.6 + d_gx * 0.4
            gy = gy * 0.6 + d_gy * 0.4

        gz = np.ones_like(gx, dtype=np.float32)
        norm = np.sqrt(gx * gx + gy * gy + gz * gz)
        norm = np.maximum(norm, 1e-6)

        nx = (gx / norm) * 0.5 + 0.5
        ny = (gy / norm) * 0.5 + 0.5
        nz = (gz / norm) * 0.5 + 0.5

        normal_map = np.dstack([nx, ny, nz])
        return (np.clip(normal_map, 0.0, 1.0) * 255.0).astype(np.uint8)

    @staticmethod
    def generate_roughness_map(
        image_rgb: np.ndarray,
        mask: Optional[np.ndarray] = None,
        base_roughness: float = 0.55,
    ) -> np.ndarray:
        """Compute physically-based Roughness map (0.0 = mirror/glossy, 1.0 = fully matte/rough).
        
        Classifies surface materials:
        - High specular reflection points & glossy screens -> 0.15 - 0.25 (glossy)
        - Smooth painted / injection molded plastic -> 0.40 - 0.55
        - Matte industrial chassis / micro-grain surfaces -> 0.65 - 0.80
        - Crevices, textured patterns, and edges -> 0.85 - 0.95
        """
        h, w = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(np.float32) / 255.0
        val = hsv[:, :, 2].astype(np.float32) / 255.0

        # 1. Specular highlight detector (high brightness + very low saturation)
        # Polished screens, glossy plastic highlights, metallic glints
        specular = np.clip((val - 0.82) / 0.18, 0.0, 1.0) * np.clip(1.0 - (sat / 0.25), 0.0, 1.0)

        # 2. Local texture variance / roughness entropy
        mean_filt = cv2.blur(gray, (5, 5))
        sq_mean_filt = cv2.blur(gray * gray, (5, 5))
        local_var = np.maximum(0.0, sq_mean_filt - mean_filt * mean_filt)
        local_std = np.sqrt(local_var)
        max_std = float(local_std.max())
        std_norm = local_std / max_std if max_std > 1e-5 else np.zeros_like(local_std)

        # 3. High-contrast edge grooves are rougher
        edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150).astype(np.float32) / 255.0
        edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8))

        # Base roughness map
        roughness = np.full((h, w), base_roughness, dtype=np.float32)
        # Glossy highlights reduce roughness significantly
        roughness -= specular * 0.45
        # Texture entropy increases roughness
        roughness += std_norm * 0.25
        # Grooves and seams are rough
        roughness += edges_dilated * 0.20

        # Mask background if provided
        if mask is not None:
            roughness[mask < 128] = 0.90

        roughness = np.clip(roughness, 0.10, 0.95)
        return (roughness * 255.0).astype(np.uint8)

    @staticmethod
    def generate_metallic_map(
        image_rgb: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute Metallic map (0.0 = Dielectric plastic/resin/wood, 1.0 = Pure Metal).
        
        Detects conductive metal contacts, screws, connectors, aluminum frames, and gold pins:
        - Achromatic silver/aluminum: High specular value with near-zero saturation.
        - Gold/Copper/Brass: Characteristic warm metallic hue band with high luster.
        """
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        hue = hsv[:, :, 0].astype(np.float32)  # 0 - 180 in OpenCV
        sat = hsv[:, :, 1].astype(np.float32) / 255.0
        val = hsv[:, :, 2].astype(np.float32) / 255.0

        # 1. Silver / Chrome / Polished Steel / Aluminum:
        # High value (> 0.55), very low saturation (< 0.12)
        silver_metal = np.clip((val - 0.55) / 0.30, 0.0, 1.0) * np.clip(1.0 - (sat / 0.12), 0.0, 1.0)

        # 2. Gold / Brass / Copper contacts (Hue between 10 and 32 in OpenCV = 20-64 deg)
        is_gold_hue = (hue >= 10) & (hue <= 32)
        gold_metal = np.zeros_like(val)
        gold_metal[is_gold_hue] = np.clip((val[is_gold_hue] - 0.65) / 0.30, 0.0, 1.0) * np.clip(sat[is_gold_hue] / 0.50, 0.0, 1.0)

        metallic = np.maximum(silver_metal * 0.95, gold_metal * 0.90)

        # Clean up isolated noise: morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        metallic_uint8 = (metallic * 255.0).astype(np.uint8)
        metallic_clean = cv2.morphologyEx(metallic_uint8, cv2.MORPH_OPEN, kernel)

        if mask is not None:
            metallic_clean[mask < 128] = 0

        return metallic_clean

    @staticmethod
    def generate_ao_map(
        image_rgb: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute Ambient Occlusion (AO) map (cavity, seam, and contact shadows).
        
        Deep recesses, chassis seams, and concave areas are darker (0.2 - 0.6),
        while flat or exposed surfaces receive full ambient light (1.0 / 255).
        """
        h, w = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        # Edge valley detection (Laplacian / high-pass grooves)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        grooves = np.clip(-laplacian * 4.0, 0.0, 1.0)
        grooves_blur = cv2.GaussianBlur(grooves, (9, 9), 0)

        # Depth concavity if depth map available
        if depth_map is not None:
            dm = depth_map
            if dm.shape[:2] != (h, w):
                dm = cv2.resize(dm, (w, h), interpolation=cv2.INTER_LINEAR)
            dm_lap = cv2.Laplacian(dm.astype(np.float32), cv2.CV_32F, ksize=5)
            depth_crevices = np.clip(-dm_lap * 3.0, 0.0, 1.0)
            grooves_blur = np.maximum(grooves_blur, depth_crevices)

        # Invert so unoccluded = 1.0, crevices = 0.40 - 0.70
        ao = 1.0 - (grooves_blur * 0.55)
        ao = np.clip(ao, 0.35, 1.0)
        return (ao * 255.0).astype(np.uint8)

    @classmethod
    def generate_rear_chassis_textures(
        cls,
        dominant_color: Tuple[int, int, int] = (50, 52, 60),
        size: Tuple[int, int] = (1024, 1024),
    ) -> Dict[str, Image.Image]:
        """Synthesize a complete industrial rear chassis texture & PBR suite.
        
        Produces:
          - Realistic ABS/polycarbonate polymer micro-grain
          - Structural perimeter chamfer border & ergonomic bevel grooves
          - Central recessed equipment / regulatory marking bay
          - Corner mounting screw bosses with metallic reflections
          - Full PBR channels: Albedo, Normal, Roughness, Metallic, AO, ORM
        """
        w, h = size
        base_r, base_g, base_b = dominant_color

        # 1. Base Albedo with polymer micro-grain
        chassis_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        chassis_rgb[:, :, 0] = base_r
        chassis_rgb[:, :, 1] = base_g
        chassis_rgb[:, :, 2] = base_b

        # High-frequency structural micro-grain (injection molded texture)
        np.random.seed(42)
        grain = np.random.normal(0, 3.5, (h, w)).astype(np.float32)
        grain = cv2.GaussianBlur(grain, (3, 3), 0)
        chassis_f = np.clip(chassis_rgb.astype(np.float32) + grain[:, :, np.newaxis], 0, 255).astype(np.uint8)

        # Draw CAD mechanical panel features using PIL
        pil_panel = Image.fromarray(chassis_f)
        draw = ImageDraw.Draw(pil_panel)

        # Perimeter Chamfer Groove (20px inset)
        inset = int(w * 0.04)
        groove_dark = (max(0, base_r - 28), max(0, base_g - 28), max(0, base_b - 28))
        groove_light = (min(255, base_r + 20), min(255, base_g + 20), min(255, base_b + 20))
        draw.rounded_rectangle([inset, inset, w - inset, h - inset], radius=16, outline=groove_dark, width=3)
        draw.rounded_rectangle([inset + 2, inset + 2, w - inset - 2, h - inset - 2], radius=14, outline=groove_light, width=1)

        # Central Recessed Equipment / Ventilation Bay (middle 60%)
        bay_x0, bay_y0 = int(w * 0.20), int(h * 0.22)
        bay_x1, bay_y1 = int(w * 0.80), int(h * 0.78)
        recess_fill = (max(0, base_r - 18), max(0, base_g - 18), max(0, base_b - 18))
        draw.rounded_rectangle([bay_x0, bay_y0, bay_x1, bay_y1], radius=12, fill=recess_fill, outline=groove_dark, width=2)

        # Horizontal ventilation cooling slats in the upper bay
        vent_y_start = bay_y0 + int((bay_y1 - bay_y0) * 0.12)
        vent_y_end = bay_y0 + int((bay_y1 - bay_y0) * 0.45)
        vent_x0 = bay_x0 + int((bay_x1 - bay_x0) * 0.10)
        vent_x1 = bay_x1 - int((bay_x1 - bay_x0) * 0.10)
        num_slats = 9
        slat_step = (vent_y_end - vent_y_start) // num_slats
        for s in range(num_slats):
            sy = vent_y_start + s * slat_step
            draw.line([(vent_x0, sy), (vent_x1, sy)], fill=groove_dark, width=3)
            draw.line([(vent_x0, sy + 2), (vent_x1, sy + 2)], fill=groove_light, width=1)

        # Regulatory label / technical plate frame in the lower bay
        label_y0 = bay_y0 + int((bay_y1 - bay_y0) * 0.58)
        label_y1 = bay_y1 - int((bay_y1 - bay_y0) * 0.12)
        label_fill = (max(0, base_r - 8), max(0, base_g - 8), max(0, base_b - 8))
        draw.rounded_rectangle([vent_x0, label_y0, vent_x1, label_y1], radius=6, fill=label_fill, outline=groove_dark, width=2)

        # 4 Corner Mounting Screws (Metallic Bosses)
        screw_r = int(w * 0.022)
        screw_coords = [
            (inset + 28, inset + 28),
            (w - inset - 28, inset + 28),
            (inset + 28, h - inset - 28),
            (w - inset - 28, h - inset - 28),
        ]
        screw_mask = np.zeros((h, w), dtype=np.uint8)
        for scx, scy in screw_coords:
            # Outer screw well
            draw.ellipse([scx - screw_r, scy - screw_r, scx + screw_r, scy + screw_r], fill=groove_dark)
            # Inner metallic screw head
            inner_r = int(screw_r * 0.65)
            draw.ellipse([scx - inner_r, scy - inner_r, scx + inner_r, scy + inner_r], fill=(185, 190, 195))
            # Cross slot (Phillips)
            slot_r = int(inner_r * 0.60)
            draw.line([(scx - slot_r, scy), (scx + slot_r, scy)], fill=(70, 75, 80), width=2)
            draw.line([(scx, scy - slot_r), (scx, scy + slot_r)], fill=(70, 75, 80), width=2)
            cv2.circle(screw_mask, (scx, scy), inner_r, 255, -1)

        panel_np = np.array(pil_panel)

        # Generate PBR maps for this rear chassis texture
        normal_np = cls.generate_normal_map(panel_np, intensity=3.2)
        roughness_np = cls.generate_roughness_map(panel_np, base_roughness=0.72)
        # Metallic screws
        metallic_np = np.zeros((h, w), dtype=np.uint8)
        metallic_np[screw_mask > 128] = 235
        roughness_np[screw_mask > 128] = 45  # Screws are polished and shiny

        ao_np = cls.generate_ao_map(panel_np)

        # Pack ORM
        orm_np = np.dstack([ao_np, roughness_np, metallic_np])

        return {
            "albedo": Image.fromarray(panel_np),
            "normal": Image.fromarray(normal_np),
            "roughness": Image.fromarray(roughness_np),
            "metallic": Image.fromarray(metallic_np),
            "ao": Image.fromarray(ao_np),
            "orm": Image.fromarray(orm_np),
        }

    @classmethod
    def bake_pbr_maps(
        cls,
        image: Union[np.ndarray, Image.Image],
        depth_map: Optional[np.ndarray] = None,
        target_size: Optional[Tuple[int, int]] = (1024, 1024),
    ) -> Dict[str, Image.Image]:
        """Bake full suite of 5 PBR material maps + glTF ORM from input image."""
        if isinstance(image, Image.Image):
            rgb_arr = np.array(image.convert("RGB"))
            alpha_mask = np.array(image.split()[-1]) if image.mode == "RGBA" else None
        elif isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 4:
                rgb_arr = image[:, :, :3]
                alpha_mask = image[:, :, 3]
            elif image.ndim == 3:
                rgb_arr = image
                alpha_mask = None
            else:
                rgb_arr = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                alpha_mask = None
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        if target_size is not None and rgb_arr.shape[:2] != (target_size[1], target_size[0]):
            rgb_arr = cv2.resize(rgb_arr, target_size, interpolation=cv2.INTER_LANCZOS4)
            if alpha_mask is not None:
                alpha_mask = cv2.resize(alpha_mask, target_size, interpolation=cv2.INTER_NEAREST)

        albedo = cls.extract_albedo(rgb_arr)
        normal = cls.generate_normal_map(rgb_arr, intensity=2.6, depth_map=depth_map)
        roughness = cls.generate_roughness_map(rgb_arr, mask=alpha_mask, base_roughness=0.55)
        metallic = cls.generate_metallic_map(rgb_arr, mask=alpha_mask)
        ao = cls.generate_ao_map(rgb_arr, depth_map=depth_map)

        # Packed ORM map (R=AO, G=Roughness, B=Metallic)
        orm = np.dstack([ao, roughness, metallic])

        return {
            "albedo": Image.fromarray(albedo),
            "normal": Image.fromarray(normal),
            "roughness": Image.fromarray(roughness),
            "metallic": Image.fromarray(metallic),
            "ao": Image.fromarray(ao),
            "orm": Image.fromarray(orm),
        }

    @classmethod
    def save_pbr_maps(
        cls,
        pbr_maps: Dict[str, Image.Image],
        output_dir: Union[str, Path],
        base_name: str,
    ) -> Dict[str, Path]:
        """Save baked PBR map images to disk with standardized naming."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        saved_paths = {}
        for key in ["albedo", "normal", "roughness", "metallic", "ao", "orm"]:
            if key in pbr_maps and pbr_maps[key] is not None:
                file_path = out_path / f"{base_name}_{key}.png"
                pbr_maps[key].save(file_path, format="PNG", optimize=True)
                saved_paths[key] = file_path

        return saved_paths
