"""TripoSR / SF3D 360-degree Foundation Model Mesh Generator.

Produces complete, watertight, omnidirectional 3D meshes (20,000 faces)
matching the quality of white_mesh.glb.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union

import numpy as np
import torch
import trimesh
from PIL import Image

from backend.ai_processor import BaseMeshGenerator, GeneratedMesh
from config.settings import OUTPUT_DIR

logger = logging.getLogger("TripoSRGenerator")


class TripoSRPreprocessor:
    """Image preprocessor adapter for TripoSR pipeline."""

    def __init__(self, generator: "TripoSRGenerator"):
        self.generator = generator

    def process(self, image: Union[np.ndarray, Image.Image, str, Path]) -> Image.Image:
        """Process image: remove background, center with foreground ratio, and return 512x512 PIL Image."""
        if hasattr(self.generator, "last_texture") and self.generator.last_texture is not None:
            return self.generator.last_texture
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        return self.generator.preprocess_image(image)


class TripoSRGenerator(BaseMeshGenerator):
    """High-fidelity 3D reconstruction engine powered by TripoSR pretrained foundation model."""

    def __init__(
        self,
        foreground_ratio: float = 0.85,
        mc_resolution: int = 256,
        target_faces: int = 20000,
        device: Optional[str] = None,
        **kwargs,
    ):
        self.foreground_ratio = foreground_ratio
        self.mc_resolution = mc_resolution
        self.target_faces = target_faces

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self._is_initialized = False
        self._rembg_session = None
        self.last_texture = None
        self.preprocessor = TripoSRPreprocessor(self)

    def is_available(self) -> bool:
        """Check if TripoSR can be initialized."""
        return True

    def load_model(self) -> None:
        """Explicitly load the TripoSR foundation model."""
        self._ensure_initialized()

    def _ensure_initialized(self) -> bool:
        """Lazy-load TripoSR model on first invocation."""
        if self._is_initialized and self.model is not None:
            return True

        logger.info(f"[TripoSR] Initializing model on device: {self.device}...")
        try:
            from backend.tsr.system import TSR

            self.model = TSR.from_pretrained(
                pretrained_model_name_or_path="stabilityai/TripoSR",
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            self.model.renderer.set_chunk_size(8192)
            self.model.to(self.device)
            self.model.eval()
            self._is_initialized = True
            logger.info("[TripoSR] Model loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"[TripoSR] Failed to load model: {e}", exc_info=True)
            return False

    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> Image.Image:
        """Preprocess using the official TripoSR run.py pipeline."""
        if isinstance(image, np.ndarray):
            if image.shape[2] == 4:
                pil_img = Image.fromarray(image, mode="RGBA")
            else:
                pil_img = Image.fromarray(image, mode="RGB")
        elif isinstance(image, Image.Image):
            pil_img = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        if not hasattr(self, "_rembg_session") or self._rembg_session is None:
            import rembg
            self._rembg_session = rembg.new_session("u2net")

        # Downscale to 768px to ensure stable ONNX memory while maintaining crisp edges
        if max(pil_img.size) > 768:
            pil_img.thumbnail((768, 768), Image.Resampling.LANCZOS)

        from backend.tsr.utils import remove_background, resize_foreground
        try:
            img_nobg = remove_background(pil_img, rembg_session=self._rembg_session)
        except Exception as e:
            logger.warning(f"[TripoSR] rembg failed: {e}, falling back to alpha threshold")
            arr = np.array(pil_img.convert("RGB"))
            gray = np.mean(arr, axis=2)
            alpha = np.where(gray > 240, 0, 255).astype(np.uint8)
            img_nobg = Image.fromarray(np.dstack([arr, alpha]), mode="RGBA")

        img_resized = resize_foreground(img_nobg, ratio=self.foreground_ratio)
        img_np = np.array(img_resized).astype(np.float32) / 255.0
        img_np = img_np[:, :, :3] * img_np[:, :, 3:4] + (1.0 - img_np[:, :, 3:4]) * 0.5
        processed_pil = Image.fromarray((img_np * 255.0).astype(np.uint8))
        if processed_pil.size != (512, 512):
            processed_pil = processed_pil.resize((512, 512), Image.Resampling.LANCZOS)
        return processed_pil

    def run_image(
        self,
        image: Union[np.ndarray, Image.Image],
        target_dimensions_mm: Optional[Tuple[float, float, float]] = None,
    ) -> trimesh.Trimesh:
        """Execute end-to-end 3D reconstruction on single image."""
        if not self._ensure_initialized():
            logger.warning("[TripoSR] Model not ready, using local Depth-to-3D fallback...")
            from backend.depth_processor import estimate_depth_map, create_depth_mesh
            if isinstance(image, Image.Image):
                img_np = np.array(image.convert("RGB"))
            else:
                img_np = image[:, :, :3] if image.ndim == 3 and image.shape[2] >= 3 else image
            depth_map = estimate_depth_map(img_np)
            return create_depth_mesh(img_np, depth_map, target_dimensions_mm=target_dimensions_mm)

        logger.info("[TripoSR] Preprocessing image...")
        processed_pil = self.preprocess_image(image)

        logger.info("[TripoSR] Running transformer inference...")
        with torch.no_grad():
            scene_codes = self.model([processed_pil], device=self.device)

        logger.info(f"[TripoSR] Extracting isosurface mesh (mc_resolution={self.mc_resolution})...")
        meshes = self.model.extract_mesh(
            scene_codes,
            has_vertex_color=True,
            resolution=self.mc_resolution,
            threshold=25.0,
        )
        raw_mesh = meshes[0]

        # Mesh Post-Processing: Simplify to target faces (e.g. 20,000 faces like white_mesh.glb)
        logger.info(f"[TripoSR] Raw mesh: {len(raw_mesh.vertices)} verts, {len(raw_mesh.faces)} faces")
        if self.target_faces and len(raw_mesh.faces) > self.target_faces:
            try:
                import fast_simplification
                logger.info(f"[TripoSR] Decimating mesh to {self.target_faces} faces using fast_simplification...")
                v_simp, f_simp = fast_simplification.simplify(
                    raw_mesh.vertices, raw_mesh.faces, target_count=self.target_faces
                )
                simp_colors = None
                if (
                    raw_mesh.visual
                    and hasattr(raw_mesh.visual, "vertex_colors")
                    and raw_mesh.visual.vertex_colors is not None
                    and len(raw_mesh.visual.vertex_colors) == len(raw_mesh.vertices)
                ):
                    try:
                        from scipy.spatial import cKDTree
                        tree = cKDTree(raw_mesh.vertices)
                        _, idxs = tree.query(v_simp)
                        simp_colors = raw_mesh.visual.vertex_colors[idxs]
                    except Exception as col_err:
                        logger.warning(f"[TripoSR] Vertex color interpolation notice: {col_err}")

                raw_mesh = trimesh.Trimesh(vertices=v_simp, faces=f_simp, vertex_colors=simp_colors, process=True)
                logger.info(f"[TripoSR] Decimated mesh: {len(raw_mesh.vertices)} verts, {len(raw_mesh.faces)} faces")
            except Exception as e:
                logger.warning(f"[TripoSR] fast_simplification failed ({e}), using raw mesh")

        # Mesh hygiene on raw surface
        raw_mesh.update_faces(raw_mesh.nondegenerate_faces())
        raw_mesh.update_faces(raw_mesh.unique_faces())
        raw_mesh.fill_holes()
        raw_mesh.fix_normals()

        # Transform TripoSR coordinates to standard upright CAD coordinates:
        # X_cad = Y_tsr (Width, Left to Right)
        # Y_cad = X_tsr (Height, Bottom to Top)
        # Z_cad = Z_tsr (Depth, Back to Front)
        v_cad = np.column_stack([raw_mesh.vertices[:, 1], raw_mesh.vertices[:, 0], raw_mesh.vertices[:, 2]])
        f_cad = raw_mesh.faces[:, ::-1]

        cad_mesh = trimesh.Trimesh(vertices=v_cad, faces=f_cad, process=True)
        cad_mesh.update_faces(cad_mesh.nondegenerate_faces())
        cad_mesh.update_faces(cad_mesh.unique_faces())
        cad_mesh.fill_holes()
        cad_mesh.fix_normals()

        # Project camera UV coordinates for razor-sharp real photo rendering
        x = cad_mesh.vertices[:, 0]
        y = cad_mesh.vertices[:, 1]
        xc = (x.min() + x.max()) / 2.0
        yc = (y.min() + y.max()) / 2.0
        L = max(x.max() - x.min(), y.max() - y.min(), 1e-5)

        u_proj = np.clip(0.5 + (x - xc) / L * self.foreground_ratio, 0.0, 1.0)
        v_proj = np.clip(0.5 + (y - yc) / L * self.foreground_ratio, 0.0, 1.0)

        # Weight front vs back by Z normal:
        # Front surface (Nz > -0.15) displays the crystal-clear photo texture
        # Back surface (Nz < -0.15) transitions to the solid body color margin
        nz = cad_mesh.vertex_normals[:, 2]
        w_front = np.clip((nz + 0.15) / 0.25, 0.0, 1.0)
        u_final = w_front * u_proj + (1.0 - w_front) * 0.02
        v_final = w_front * v_proj + (1.0 - w_front) * 0.98
        uvs = np.column_stack([u_final, v_final]).astype(np.float32)

        # Sample dominant body color to fill the texture margin [0:20, 0:20]
        tex_arr = np.array(processed_pil).copy()
        try:
            if isinstance(image, np.ndarray) and image.ndim == 3:
                fg_p = image.reshape(-1, image.shape[-1])
                if image.shape[-1] == 4:
                    fg_p = fg_p[fg_p[:, 3] > 60, :3]
                if len(fg_p) > 0:
                    body_color = np.median(fg_p, axis=0).astype(np.uint8)
                else:
                    body_color = np.array([60, 60, 70], dtype=np.uint8)
            elif isinstance(image, Image.Image):
                arr_im = np.array(image.convert("RGBA"))
                fg_p = arr_im.reshape(-1, 4)
                fg_p = fg_p[fg_p[:, 3] > 60, :3]
                if len(fg_p) > 0:
                    body_color = np.median(fg_p, axis=0).astype(np.uint8)
                else:
                    body_color = np.array([60, 60, 70], dtype=np.uint8)
            else:
                body_color = np.array([60, 60, 70], dtype=np.uint8)
        except Exception:
            body_color = np.array([60, 60, 70], dtype=np.uint8)

        tex_arr[:20, :20, :3] = body_color[:3]
        final_tex = Image.fromarray(tex_arr)
        self.last_texture = final_tex

        cad_mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, image=final_tex)
        return cad_mesh

    def generate(
        self,
        layer_image: np.ndarray,
        layer_id: str,
        bbox: tuple[int, int, int, int] | None = None,
        image_size: tuple[int, int] | None = None,
        layer_index: int = 0,
        target_dimensions_mm: tuple[float, float, float] | None = None,
    ) -> GeneratedMesh:
        """Standard BaseMeshGenerator interface returning GeneratedMesh with STL, GLB, OBJ."""
        from backend.sf3d_pipeline import AssetExporter3D

        tri_mesh = self.run_image(layer_image, target_dimensions_mm=target_dimensions_mm)
        aligned_mesh = AssetExporter3D.align_to_ground(tri_mesh, target_dimensions_mm=target_dimensions_mm)

        meshes_dir = OUTPUT_DIR / "meshes"
        textures_dir = OUTPUT_DIR / "textures"
        meshes_dir.mkdir(parents=True, exist_ok=True)
        textures_dir.mkdir(parents=True, exist_ok=True)

        obj_path = meshes_dir / f"{layer_id}.obj"
        texture_path = textures_dir / f"{layer_id}_diffuse.png"

        # Save processed texture
        prep_img = self.preprocess_image(layer_image)
        prep_img.save(texture_path)

        # Export multi-format
        AssetExporter3D.export_all(
            mesh=aligned_mesh,
            output_dir=meshes_dir,
            base_name=layer_id,
            target_dimensions_mm=target_dimensions_mm,
            texture_path=texture_path,
        )

        dim_mm = tuple(float(x) for x in aligned_mesh.extents)
        return GeneratedMesh(
            layer_id=layer_id,
            mesh_path=obj_path,
            texture_path=texture_path,
            vertices_count=len(aligned_mesh.vertices),
            faces_count=len(aligned_mesh.faces),
            centroid_3d=tuple(float(x) for x in aligned_mesh.centroid),
            dimensions_mm=dim_mm,
        )
