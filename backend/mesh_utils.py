"""Mesh conversion utilities for bridging trimesh objects to Three.js viewer data."""

import numpy as np
import trimesh
from pathlib import Path
from PIL import Image


def compute_dominant_color(image_crop: np.ndarray) -> tuple[int, int, int]:
    """Extract the dominant color from an image crop.
    
    Args:
        image_crop: RGB or RGBA image array.
        
    Returns:
        Dominant color as (R, G, B) tuple.
    """
    # If RGBA, use only non-transparent pixels
    if image_crop.ndim == 3 and image_crop.shape[2] == 4:
        alpha = image_crop[:, :, 3]
        mask = alpha > 128
        if mask.any():
            pixels = image_crop[mask][:, :3]
        else:
            pixels = image_crop[:, :, :3].reshape(-1, 3)
    elif image_crop.ndim == 3:
        pixels = image_crop.reshape(-1, 3)
    else:
        return (128, 128, 128)
    
    if len(pixels) == 0:
        return (128, 128, 128)
    
    # Use mean color as dominant (fast approximation)
    mean_color = np.mean(pixels, axis=0).astype(int)
    return (int(mean_color[0]), int(mean_color[1]), int(mean_color[2]))


def trimesh_to_viewer_data(
    mesh: trimesh.Trimesh,
    color: tuple[int, int, int],
    base_position: tuple[float, float, float],
    explosion_direction: tuple[float, float, float],
    layer_id: str = "",
    dimensions_mm: tuple[float, float, float] | None = None,
    texture_path: Path | str | None = None,
) -> dict:
    """Convert a trimesh object to a dict suitable for embedding in Three.js HTML.
    
    Args:
        mesh: The trimesh object.
        color: RGB color tuple (0-255).
        base_position: (x, y, z) resting position.
        explosion_direction: (dx, dy, dz) direction vector for exploded view.
        layer_id: Identifier string for the layer.
        dimensions_mm: Optional (W, H, D) dimensions in mm.
        texture_path: Optional path to diffuse texture image file.
        
    Returns:
        Dict with keys: vertices, faces, color, vertexColors, uvs, textureDataUri,
        basePosition, explosionDir, layerId, dimensionsMm.
    """
    vertices = mesh.vertices.flatten().tolist()
    faces = mesh.faces.flatten().tolist()
    
    # Extract vertex colors if available
    vertex_colors_list = None
    if hasattr(mesh, "visual") and hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        vc = np.asarray(mesh.visual.vertex_colors)
        if len(vc) == len(mesh.vertices) and vc.shape[1] >= 3:
            # Normalize to [0.0, 1.0] for Three.js Float32BufferAttribute
            vertex_colors_list = (vc[:, :3] / 255.0).astype(np.float32).flatten().tolist()

    # Extract UV texture coordinates if available
    uv_list = None
    if hasattr(mesh, "visual") and hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
        uvs = np.asarray(mesh.visual.uv)
        if len(uvs) == len(mesh.vertices) and uvs.ndim == 2 and uvs.shape[1] == 2:
            uv_list = uvs.astype(np.float32).flatten().tolist()

    # Base64 encode texture data URI for embedded Three.js rendering
    texture_data_uri = None
    if texture_path is not None:
        p = Path(texture_path)
        if p.exists():
            import base64
            try:
                with open(p, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                    texture_data_uri = f"data:image/png;base64,{b64}"
            except Exception:
                pass

    hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
    
    return {
        "vertices": vertices,
        "faces": faces,
        "color": hex_color,
        "vertexColors": vertex_colors_list,
        "uvs": uv_list,
        "textureDataUri": texture_data_uri,
        "basePosition": list(base_position),
        "explosionDir": list(explosion_direction),
        "layerId": layer_id,
        "dimensionsMm": list(dimensions_mm) if dimensions_mm else None,
    }


def load_obj_as_trimesh(obj_path: Path) -> trimesh.Trimesh:
    """Load an OBJ file as a trimesh object.
    
    Args:
        obj_path: Path to the .obj file.
        
    Returns:
        Loaded Trimesh object.
        
    Raises:
        FileNotFoundError: If the OBJ file does not exist.
    """
    if not obj_path.exists():
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")
    loaded = trimesh.load(str(obj_path), force='mesh')
    if isinstance(loaded, trimesh.Scene):
        # Combine all meshes in the scene
        meshes = []
        for geometry in loaded.geometry.values():
            if isinstance(geometry, trimesh.Trimesh):
                meshes.append(geometry)
        if meshes:
            loaded = trimesh.util.concatenate(meshes)
        else:
            raise ValueError(f"No meshes found in {obj_path}")

    # If loaded mesh doesn't have vertex colors, check if OBJ has per-vertex RGB (v x y z r g b)
    has_vc = hasattr(loaded, "visual") and hasattr(loaded.visual, "vertex_colors") and loaded.visual.vertex_colors is not None
    if not has_vc:
        try:
            with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
                colors = []
                for line in f:
                    if line.startswith("v "):
                        parts = line.split()
                        if len(parts) >= 7:
                            colors.append([float(parts[4]), float(parts[5]), float(parts[6])])
                if colors and len(colors) == len(loaded.vertices):
                    rgb_bytes = (np.clip(np.array(colors, dtype=np.float32), 0.0, 1.0) * 255).astype(np.uint8)
                    rgba_bytes = np.column_stack([rgb_bytes, np.full(len(rgb_bytes), 255, dtype=np.uint8)])
                    loaded.visual.vertex_colors = rgba_bytes
        except Exception:
            pass

    # If loaded mesh doesn't have UVs, check if OBJ has vt definitions
    has_uv = hasattr(loaded, "visual") and hasattr(loaded.visual, "uv") and loaded.visual.uv is not None
    if not has_uv or len(loaded.visual.uv) != len(loaded.vertices):
        try:
            with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
                uvs = []
                for line in f:
                    if line.startswith("vt "):
                        parts = line.split()
                        if len(parts) >= 3:
                            uvs.append([float(parts[1]), float(parts[2])])
                if uvs and len(uvs) == len(loaded.vertices):
                    loaded.visual.uv = np.array(uvs, dtype=np.float32)
        except Exception:
            pass

    return loaded


def create_textured_box(
    width: float,
    height: float,
    depth: float,
    color: tuple[int, int, int],
    output_dir: Path,
    layer_id: str,
) -> tuple[Path, Path, Path]:
    """Create a box mesh with a solid-color texture and write to disk.
    
    Args:
        width: Box width.
        height: Box height.  
        depth: Box depth.
        color: RGB color for the texture.
        output_dir: Directory to write files to.
        layer_id: Layer identifier for filenames.
        
    Returns:
        Tuple of (obj_path, mtl_path, texture_path).
    """
    meshes_dir = output_dir / "meshes"
    textures_dir = output_dir / "textures"
    meshes_dir.mkdir(parents=True, exist_ok=True)
    textures_dir.mkdir(parents=True, exist_ok=True)
    
    # Create box mesh
    mesh = trimesh.creation.box(extents=[width, height, depth])
    
    # Create solid color texture
    texture_size = 64
    texture = Image.new('RGB', (texture_size, texture_size), color)
    texture_path = textures_dir / f"{layer_id}_diffuse.png"
    texture.save(str(texture_path))
    
    # Write MTL file
    mtl_path = meshes_dir / f"{layer_id}.mtl"
    # Use relative path from meshes/ to textures/
    relative_tex_path = f"../textures/{layer_id}_diffuse.png"
    mtl_content = f"""# Material for {layer_id}
newmtl {layer_id}_material
Ka 0.2 0.2 0.2
Kd {color[0]/255:.3f} {color[1]/255:.3f} {color[2]/255:.3f}
Ks 0.1 0.1 0.1
Ns 10.0
d 1.0
illum 2
map_Kd {relative_tex_path}
"""
    mtl_path.write_text(mtl_content, encoding='utf-8')
    
    # Write OBJ file with mtllib reference
    obj_path = meshes_dir / f"{layer_id}.obj"
    obj_content = mesh.export(file_type='obj')
    if isinstance(obj_content, bytes):
        obj_content = obj_content.decode('utf-8')
    
    # Prepend material reference
    obj_with_mtl = f"mtllib {layer_id}.mtl\nusemtl {layer_id}_material\n" + obj_content
    obj_path.write_text(obj_with_mtl, encoding='utf-8')
    
    return obj_path, mtl_path, texture_path
