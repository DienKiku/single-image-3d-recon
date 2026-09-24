"""Unit tests for Depth-to-3D surface reconstruction and processing."""

from pathlib import Path
import numpy as np
import pytest
import trimesh

from backend.depth_processor import (
    estimate_depth_map,
    create_depth_mesh,
    slice_mesh_into_layers,
    export_textured_obj,
    detect_salient_foreground_box,
)


def test_estimate_depth_map_shape_and_range() -> None:
    """Verify that depth estimation returns a valid [0, 1] normalized map matching image size."""
    image = np.full((120, 160, 3), 128, dtype=np.uint8)
    # Add a bright square in center to create contrast
    image[40:80, 50:110] = 240

    depth = estimate_depth_map(image)

    assert depth.shape == (120, 160)
    assert depth.dtype == np.float32
    assert depth.min() >= 0.0
    assert depth.max() <= 1.0


def test_create_depth_mesh_properties() -> None:
    """Verify that create_depth_mesh constructs a valid Trimesh with vertices, faces, and colors."""
    image = np.full((80, 80, 3), 200, dtype=np.uint8)
    image[20:60, 20:60] = [255, 0, 0]  # Red square
    depth = np.linspace(0.0, 1.0, 80 * 80).reshape(80, 80).astype(np.float32)

    mesh = create_depth_mesh(image, depth, max_grid_size=30, depth_scale=0.2)

    assert isinstance(mesh, trimesh.Trimesh)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0
    assert hasattr(mesh.visual, "vertex_colors")
    assert mesh.visual.vertex_colors is not None
    assert len(mesh.visual.vertex_colors) == len(mesh.vertices)


def test_create_depth_mesh_target_dimensions() -> None:
    """Verify that specifying target dimensions scales the mesh extents accurately."""
    image = np.full((60, 60, 3), 100, dtype=np.uint8)
    depth = np.ones((60, 60), dtype=np.float32) * 0.5
    target_dims = (300.0, 300.0, 50.0)

    mesh = create_depth_mesh(
        image,
        depth,
        max_grid_size=20,
        target_dimensions_mm=target_dims,
    )

    extents = mesh.extents
    assert extents[0] == pytest.approx(300.0, rel=1e-2)
    assert extents[1] == pytest.approx(300.0, rel=1e-2)


def test_slice_mesh_into_layers() -> None:
    """Verify that slice_mesh_into_layers splits the mesh along Y into multiple layers."""
    image = np.full((60, 60, 3), 150, dtype=np.uint8)
    depth = np.ones((60, 60), dtype=np.float32) * 0.5
    mesh = create_depth_mesh(image, depth, max_grid_size=25)

    layers = slice_mesh_into_layers(mesh, num_layers=3, axis=1)

    assert len(layers) == 3
    for lyr in layers:
        assert isinstance(lyr, trimesh.Trimesh)
        assert len(lyr.vertices) > 0
        assert len(lyr.faces) > 0


def test_export_textured_obj(tmp_path: Path) -> None:
    """Verify that export_textured_obj writes valid OBJ, MTL, and PNG texture files."""
    image = np.full((40, 40, 3), 180, dtype=np.uint8)
    depth = np.ones((40, 40), dtype=np.float32) * 0.5
    mesh = create_depth_mesh(image, depth, max_grid_size=15)

    obj_path = tmp_path / "meshes" / "test_part.obj"
    tex_path = tmp_path / "textures" / "test_part_diffuse.png"

    out_obj, out_mtl, out_tex = export_textured_obj(
        mesh=mesh,
        texture_image=image,
        output_obj_path=obj_path,
        output_texture_path=tex_path,
        layer_id="test_part",
    )

    assert out_obj.exists()
    assert out_mtl.exists()
    assert out_tex.exists()
    assert out_obj.with_suffix(".stl").exists()
    assert out_obj.stat().st_size > 0
    assert "mtllib test_part.mtl" in out_obj.read_text(encoding="utf-8")


def test_detect_salient_foreground_box() -> None:
    """Verify that detect_salient_foreground_box returns a valid bounding box."""
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    # Bright center square
    image[50:150, 50:150] = 255

    depth = np.zeros((200, 200), dtype=np.float32)
    depth[50:150, 50:150] = 0.9

    bbox = detect_salient_foreground_box(image, depth_map=depth)

    assert len(bbox) == 4
    x, y, w, h = bbox
    assert x >= 0 and y >= 0
    assert w > 50 and h > 50
    assert x + w <= 200 and y + h <= 200


def test_create_depth_mesh_with_silhouette_mask() -> None:
    """Verify that create_depth_mesh with mask produces a watertight solid matching the silhouette."""
    image = np.full((80, 80, 3), 180, dtype=np.uint8)
    depth = np.ones((80, 80), dtype=np.float32) * 0.5
    
    # Circular mask in center
    mask = np.zeros((80, 80), dtype=np.uint8)
    import cv2
    cv2.circle(mask, (40, 40), 25, 255, -1)
    
    mesh = create_depth_mesh(image, depth, mask=mask, max_grid_size=30, solid=True)
    
    assert isinstance(mesh, trimesh.Trimesh)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0
    assert mesh.is_watertight
    assert mesh.volume > 0

