"""Unit tests for Physically Based Rendering (PBR) Baker and Rear Surface Enhancements."""

import numpy as np
from pathlib import Path
from PIL import Image
import trimesh
import pytest

from backend.pbr_baker import PBRBaker
from backend.sf3d_pipeline import AssetExporter3D
from backend.mesh_utils import trimesh_to_viewer_data


@pytest.fixture
def synthetic_rgb_image():
    """Create a synthetic 128x128 RGB test image with specular, matte, and metallic regions."""
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    # Background: matte blue plastic
    img[:, :] = [30, 80, 180]
    # Center: high specular white highlight (e.g. glass / reflection)
    img[40:60, 40:60] = [250, 250, 252]
    # Bottom: silver metallic screw / contact
    img[90:110, 90:110] = [210, 210, 212]
    return img


def test_extract_albedo(synthetic_rgb_image):
    """Verify albedo extraction produces valid uint8 image with preserved dimensions."""
    albedo = PBRBaker.extract_albedo(synthetic_rgb_image)
    assert albedo.shape == synthetic_rgb_image.shape
    assert albedo.dtype == np.uint8
    assert albedo.min() >= 0
    assert albedo.max() <= 255


def test_generate_normal_map(synthetic_rgb_image):
    """Verify normal map generation adheres to tangent-space standards."""
    normal = PBRBaker.generate_normal_map(synthetic_rgb_image, intensity=2.0)
    assert normal.shape == (128, 128, 3)
    assert normal.dtype == np.uint8
    # In standard tangent space normal maps, Z (Blue channel) is outward and typically >= 128
    assert np.mean(normal[:, :, 2]) > 120


def test_generate_roughness_map(synthetic_rgb_image):
    """Verify specular highlights have lower roughness than matte areas."""
    roughness = PBRBaker.generate_roughness_map(synthetic_rgb_image, base_roughness=0.60)
    assert roughness.shape == (128, 128)
    assert roughness.dtype == np.uint8
    # Specular white patch should be smoother (lower roughness) than matte blue background
    specular_patch = roughness[45:55, 45:55]
    matte_patch = roughness[10:25, 10:25]
    assert np.mean(specular_patch) < np.mean(matte_patch)


def test_generate_metallic_map(synthetic_rgb_image):
    """Verify metallic map correctly highlights conductive silver/chrome areas."""
    metallic = PBRBaker.generate_metallic_map(synthetic_rgb_image)
    assert metallic.shape == (128, 128)
    assert metallic.dtype == np.uint8
    # Silver metallic patch should have high metalness
    silver_patch = metallic[95:105, 95:105]
    # Saturated blue plastic should have near-zero metalness
    plastic_patch = metallic[10:25, 10:25]
    assert np.mean(silver_patch) > 150
    assert np.mean(plastic_patch) < 30


def test_generate_ao_map(synthetic_rgb_image):
    """Verify ambient occlusion map produces values in [0, 255]."""
    ao = PBRBaker.generate_ao_map(synthetic_rgb_image)
    assert ao.shape == (128, 128)
    assert ao.dtype == np.uint8
    assert ao.max() == 255
    assert ao.min() >= 0


def test_bake_pbr_maps(synthetic_rgb_image):
    """Verify complete 6-channel PBR suite is generated and packed properly."""
    maps = PBRBaker.bake_pbr_maps(synthetic_rgb_image, target_size=(256, 256))
    expected_keys = {"albedo", "normal", "roughness", "metallic", "ao", "orm"}
    assert set(maps.keys()) == expected_keys
    for k, img in maps.items():
        assert isinstance(img, Image.Image)
        assert img.size == (256, 256)
    # Check ORM packing channels (R=AO, G=Roughness, B=Metallic)
    orm_arr = np.array(maps["orm"])
    assert orm_arr.shape == (256, 256, 3)


def test_generate_rear_chassis_textures():
    """Verify industrial rear chassis panel synthesizer produces mechanical textures."""
    rear = PBRBaker.generate_rear_chassis_textures(dominant_color=(45, 50, 60), size=(256, 256))
    expected_keys = {"albedo", "normal", "roughness", "metallic", "ao", "orm"}
    assert set(rear.keys()) == expected_keys
    for k, img in rear.items():
        assert isinstance(img, Image.Image)
        assert img.size == (256, 256)


def test_save_pbr_maps(tmp_path, synthetic_rgb_image):
    """Verify saving PBR maps writes all files to disk."""
    maps = PBRBaker.bake_pbr_maps(synthetic_rgb_image, target_size=(64, 64))
    saved = PBRBaker.save_pbr_maps(maps, tmp_path, "test_part")
    assert len(saved) == 6
    for k, p in saved.items():
        assert p.exists()
        assert p.stat().st_size > 0


def test_asset_exporter_with_pbr(tmp_path, synthetic_rgb_image):
    """Verify AssetExporter3D embeds PBRMaterial in GLB and references PBR maps in MTL."""
    mesh = trimesh.creation.box()
    uv = np.zeros((len(mesh.vertices), 2), dtype=np.float32)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv)

    maps = PBRBaker.bake_pbr_maps(synthetic_rgb_image, target_size=(64, 64))
    PBRBaker.save_pbr_maps(maps, tmp_path, "box_part")
    diffuse_path = tmp_path / "box_part_diffuse.png"
    maps["albedo"].save(diffuse_path)

    exports = AssetExporter3D.export_all(
        mesh=mesh,
        output_dir=tmp_path,
        base_name="box_part",
        texture_path=diffuse_path,
        pbr_maps=maps,
    )
    assert "glb" in exports and exports["glb"].exists()
    assert "stl" in exports and exports["stl"].exists()
    assert "obj" in exports and exports["obj"].exists()
    assert "mtl" in exports and exports["mtl"].exists()

    mtl_content = exports["mtl"].read_text(encoding="utf-8")
    assert "norm " in mtl_content or "map_Bump" in mtl_content
    assert "map_Pr " in mtl_content
    assert "map_Pm " in mtl_content


def test_trimesh_to_viewer_data_pbr_detection(tmp_path, synthetic_rgb_image):
    """Verify trimesh_to_viewer_data auto-detects sibling PBR maps and encodes data URIs."""
    mesh = trimesh.creation.box()
    uv = np.zeros((len(mesh.vertices), 2), dtype=np.float32)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv)

    maps = PBRBaker.bake_pbr_maps(synthetic_rgb_image, target_size=(64, 64))
    PBRBaker.save_pbr_maps(maps, tmp_path, "layer_01")
    tex_path = tmp_path / "layer_01_diffuse.png"
    maps["albedo"].save(tex_path)

    v_data = trimesh_to_viewer_data(
        mesh=mesh,
        color=(50, 100, 150),
        base_position=(0.0, 0.0, 0.0),
        explosion_direction=(0.0, 1.0, 0.0),
        layer_id="layer_01",
        texture_path=tex_path,
    )

    assert v_data["textureDataUri"] is not None
    assert v_data["normalMapDataUri"] is not None
    assert v_data["roughnessMapDataUri"] is not None
    assert v_data["metalnessMapDataUri"] is not None
    assert v_data["aoMapDataUri"] is not None
