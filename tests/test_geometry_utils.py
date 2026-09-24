"""Unit tests for geometry utility functions and scale calibration."""

import numpy as np
import pytest
import trimesh

from backend.geometry_utils import (
    ScaleCalibration,
    annotate_dimensions,
    calibrate_scale,
    compute_exploded_positions,
    compute_pixel_to_mm_ratio,
    detect_aruco_markers,
    scale_mesh_to_metric,
)


def test_compute_pixel_to_mm_ratio_known_square() -> None:
    """Verify pixel-to-mm ratio calculation for a known 100x100 square."""
    marker_corners = np.array(
        [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
        dtype=np.float32,
    )
    known_size_mm = 50.0
    ratio = compute_pixel_to_mm_ratio(marker_corners, known_size_mm)

    assert ratio == pytest.approx(0.5, rel=1e-5)


def test_compute_pixel_to_mm_ratio_rectangle() -> None:
    """Verify pixel-to-mm ratio computation on rectangular corner inputs."""
    marker_corners = np.array(
        [[0.0, 0.0], [200.0, 0.0], [200.0, 100.0], [0.0, 100.0]],
        dtype=np.float32,
    )
    known_size_mm = 75.0
    ratio = compute_pixel_to_mm_ratio(marker_corners, known_size_mm)

    assert ratio == pytest.approx(0.5, rel=1e-5)


def test_compute_pixel_to_mm_ratio_zero_size_raises() -> None:
    """Ensure zero-sized corner geometry raises a ValueError."""
    marker_corners = np.array(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match="near zero"):
        compute_pixel_to_mm_ratio(marker_corners, 50.0)


def test_compute_exploded_positions_zero_factor() -> None:
    """Verify that an expansion factor of zero returns zero offset vectors."""
    centroids = [
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
    ]
    global_centroid = np.array([2.5, 3.5, 4.5])
    expansion_factor = 0.0

    offsets = compute_exploded_positions(centroids, global_centroid, expansion_factor)

    assert len(offsets) == 2
    for offset in offsets:
        assert np.allclose(offset, np.zeros(3))


def test_compute_exploded_positions_unit_factor() -> None:
    """Verify offset computations with an expansion factor of 1.0."""
    centroids = [
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        np.array([0.0, 2.0, 0.0]),
    ]
    global_centroid = np.mean(centroids, axis=0)
    expansion_factor = 1.0

    offsets = compute_exploded_positions(centroids, global_centroid, expansion_factor)

    assert len(offsets) == 3
    expected_first_offset = np.array([-2.0 / 3.0, -2.0 / 3.0, 0.0])
    assert np.allclose(offsets[0], expected_first_offset, atol=1e-5)

    for centroid, offset in zip(centroids, offsets):
        expected_offset = (centroid - global_centroid) * expansion_factor
        assert np.allclose(offset, expected_offset, atol=1e-5)


def test_compute_exploded_positions_single_centroid() -> None:
    """Verify that a single centroid returns a zero vector."""
    centroids = [np.array([5.0, 5.0, 5.0])]
    global_centroid = np.array([5.0, 5.0, 5.0])
    expansion_factor = 1.0

    offsets = compute_exploded_positions(centroids, global_centroid, expansion_factor)

    assert len(offsets) == 1
    assert np.allclose(offsets[0], np.zeros(3))


def test_scale_mesh_to_metric() -> None:
    """Verify mesh scaling matches target metric dimensions from pixel bbox."""
    mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    pixel_to_mm = 2.0
    original_pixel_bbox = (0.0, 0.0, 50.0, 50.0)

    scaled_mesh = scale_mesh_to_metric(mesh, pixel_to_mm, original_pixel_bbox)

    assert scaled_mesh.extents[0] == pytest.approx(100.0, rel=1e-4)


def test_detect_aruco_markers_no_markers() -> None:
    """Ensure detect_aruco_markers returns an empty list for a blank image."""
    blank_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    markers = detect_aruco_markers(blank_image)

    assert markers == []


def test_annotate_dimensions_no_boxes() -> None:
    """Verify annotate_dimensions returns an image of matching shape when empty."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated = annotate_dimensions(image, [], 0.5)

    assert annotated.shape == image.shape


def test_annotate_dimensions_with_boxes() -> None:
    """Verify annotate_dimensions draws bounding boxes onto the target image."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    bboxes = [(10, 10, 100, 50)]
    pixel_to_mm = 0.5

    annotated = annotate_dimensions(image, bboxes, pixel_to_mm)

    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)


def test_calibrate_scale_custom_manual() -> None:
    """Verify manual scale calibration creates expected ScaleCalibration object."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    calibration = calibrate_scale(
        image=image,
        reference_type="Custom (Manual Input)",
        manual_pixel_distance=200.0,
        manual_real_mm=100.0,
    )

    assert isinstance(calibration, ScaleCalibration)
    assert calibration.pixel_to_mm == pytest.approx(0.5, rel=1e-5)
    assert calibration.marker_detected is False
    assert calibration.marker_id is None
    assert calibration.marker_corners is None


def test_calibrate_scale_no_marker_found() -> None:
    """Verify calibration falls back gracefully when no ArUco marker exists."""
    blank_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    calibration = calibrate_scale(
        image=blank_image,
        reference_type="ArUco Marker (5x5cm)",
    )

    assert isinstance(calibration, ScaleCalibration)
    assert calibration.pixel_to_mm == pytest.approx(0.0)
    assert calibration.marker_detected is False
    assert calibration.marker_id is None
    assert calibration.marker_corners is None
