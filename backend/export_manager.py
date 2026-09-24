import json
import shutil
import datetime
from pathlib import Path
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)

def export_project(
    generated_meshes: list,
    segmented_layers: list,
    metadata: dict,
    output_dir: Path,
) -> Path:
    """Export the project into a structured ZIP archive.
    
    Creates a ZIP containing:
    - project_metadata.json — full project metadata
    - meshes/ — .obj and .mtl files for each layer
    - textures/ — .png texture maps for each layer
    
    Args:
        generated_meshes: List of GeneratedMesh objects with valid file paths.
        segmented_layers: List of SegmentedLayer objects.
        metadata: Dictionary with calibration and summary data.
        output_dir: Base output directory.
        
    Returns:
        Path to the generated .zip file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    export_folder = output_dir / f"export_{timestamp}"
    export_folder.mkdir(parents=True, exist_ok=True)
    
    meshes_dir = export_folder / "meshes"
    meshes_dir.mkdir()
    textures_dir = export_folder / "textures"
    textures_dir.mkdir()
    
    layers_meta = []
    
    for i, mesh in enumerate(generated_meshes):
        layer = segmented_layers[i] if i < len(segmented_layers) else None
        
        # Copy obj
        mesh_file_path = None
        if mesh.mesh_path and mesh.mesh_path.exists():
            shutil.copy(mesh.mesh_path, meshes_dir / mesh.mesh_path.name)
            mesh_file_path = f"meshes/{mesh.mesh_path.name}"
        else:
            print(f"Warning: Mesh file not found {mesh.mesh_path}")
            
        # Copy mtl
        mtl_file_path = None
        if mesh.mesh_path:
            mtl_path = mesh.mesh_path.with_suffix('.mtl')
            if mtl_path.exists():
                shutil.copy(mtl_path, meshes_dir / mtl_path.name)
                mtl_file_path = f"meshes/{mtl_path.name}"

        # Copy stl
        stl_file_path = None
        if mesh.mesh_path:
            stl_path = mesh.mesh_path.with_suffix('.stl')
            if stl_path.exists():
                shutil.copy(stl_path, meshes_dir / stl_path.name)
                stl_file_path = f"meshes/{stl_path.name}"

        # Copy glb
        glb_file_path = None
        if mesh.mesh_path:
            glb_path = mesh.mesh_path.with_suffix('.glb')
            if glb_path.exists():
                shutil.copy(glb_path, meshes_dir / glb_path.name)
                glb_file_path = f"meshes/{glb_path.name}"
        
        # Copy texture
        texture_file_path = None
        if mesh.texture_path and mesh.texture_path.exists():
            shutil.copy(mesh.texture_path, textures_dir / mesh.texture_path.name)
            texture_file_path = f"textures/{mesh.texture_path.name}"
        else:
            print(f"Warning: Texture file not found {mesh.texture_path}")
            
        layer_meta = {
            "layer_id": mesh.layer_id,
            "mesh_file": mesh_file_path,
            "material_file": mtl_file_path,
            "stl_file": stl_file_path,
            "glb_file": glb_file_path,
            "texture_file": texture_file_path,
            "centroid_3d": list(mesh.centroid_3d),
        }
        
        if hasattr(mesh, "dimensions_mm") and mesh.dimensions_mm and len(mesh.dimensions_mm) >= 3:
            layer_meta["dimensions_mm"] = {
                "width": float(mesh.dimensions_mm[0]),
                "height": float(mesh.dimensions_mm[1]),
                "depth": float(mesh.dimensions_mm[2]),
            }
        elif layer and metadata.get("scale_calibration", {}).get("pixel_to_mm", 0) > 0:
            px2mm = metadata["scale_calibration"]["pixel_to_mm"]
            _, _, bw, bh = layer.bbox
            depth_est = bw * px2mm * 0.70
            layer_meta["dimensions_mm"] = {
                "width": bw * px2mm,
                "height": bh * px2mm,
                "depth": depth_est,
            }

        if layer:
            layer_meta["confidence"] = getattr(layer, 'confidence', 1.0)
                
        layers_meta.append(layer_meta)
        
    project_metadata = {
        "project_name": "2D-to-3D Studio Export",
        "export_timestamp": timestamp,
        "total_parts": len(generated_meshes),
        "scale_calibration": metadata.get("scale_calibration", {}),
        "overall_dimensions_mm": metadata.get("overall_dimensions_mm", {}),
        "layers": layers_meta
    }
    
    meta_path = export_folder / "project_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(project_metadata, f, indent=4, cls=NumpyEncoder)
        
    zip_path_str = shutil.make_archive(str(export_folder), 'zip', root_dir=export_folder)
    shutil.rmtree(export_folder)
    
    return Path(zip_path_str)
