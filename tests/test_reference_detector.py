"""Unit tests for reference_detector module."""
import numpy as np
import cv2
import pytest

from backend.reference_detector import (
    order_quadrilateral_points,
    measure_pixel_distance,
    detect_rectangular_reference,
    draw_ruler_annotation,
    draw_detected_rectangle
)

def test_order_quadrilateral_points():
    unordered = np.array([[100, 100], [0, 0], [100, 0], [0, 100]], dtype=np.float32)
    ordered = order_quadrilateral_points(unordered)
    assert np.allclose(ordered[0], [0, 0])
    assert np.allclose(ordered[1], [100, 0])
    assert np.allclose(ordered[2], [100, 100])
    assert np.allclose(ordered[3], [0, 100])

def test_measure_pixel_distance():
    dist = measure_pixel_distance((0.0, 0.0), (30.0, 40.0))
    assert dist == pytest.approx(50.0)

def test_detect_rectangular_reference_synthetic():
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    # Draw a clear white rectangle on black background: 317w x 200h (aspect ratio ~ 1.585)
    cv2.rectangle(img, (100, 100), (417, 300), (255, 255, 255), -1)
    results = detect_rectangular_reference(img, expected_ratio=1.586, ratio_tolerance=0.1)
    assert len(results) > 0
    best = results[0]
    assert best['aspect_ratio'] == pytest.approx(317.0 / 200.0, rel=0.05)
    assert best['confidence'] > 0.5

def test_draw_ruler_annotation():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    annotated = draw_ruler_annotation(img, (10, 10), (100, 10), distance_mm=45.0)
    assert annotated.shape == img.shape
    assert not np.array_equal(annotated, img)

def test_draw_detected_rectangle():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    mock_rect = {
        'corners': np.array([[10, 10], [80, 10], [80, 50], [10, 50]], dtype=np.float32),
        'width_px': 70.0,
        'height_px': 40.0,
        'aspect_ratio': 1.75,
        'center': np.array([45.0, 30.0]),
        'confidence': 0.9
    }
    annotated = draw_detected_rectangle(img, mock_rect, label='Test Card', pixel_to_mm=0.5)
    assert annotated.shape == img.shape
    assert not np.array_equal(annotated, img)
