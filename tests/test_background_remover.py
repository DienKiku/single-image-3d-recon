"""Unit tests for offline background removal module."""

import numpy as np
import pytest
from backend.background_remover import remove_background, crop_to_foreground


def test_remove_background_synthetic() -> None:
    """Verify that remove_background isolates a distinct foreground object."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Bright red square in center
    img[25:75, 25:75] = [255, 0, 0]
    
    rgba, mask, bbox = remove_background(img)
    
    assert rgba.shape == (100, 100, 4)
    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
    assert len(bbox) == 4
    
    # Foreground must have positive area
    x, y, w, h = bbox
    assert w > 0 and h > 0
    assert np.sum(mask > 0) > 0


def test_crop_to_foreground() -> None:
    """Verify crop_to_foreground tightly slices the object."""
    rgba = np.zeros((100, 100, 4), dtype=np.uint8)
    rgba[30:70, 30:70] = [200, 100, 50, 255]
    bbox = (30, 30, 40, 40)
    
    cropped, new_bbox = crop_to_foreground(rgba, bbox, padding=2)
    
    assert cropped.shape[0] == 44  # 40 + 2*padding
    assert cropped.shape[1] == 44
    assert cropped.shape[2] == 4
    assert new_bbox[0] == 28
    assert new_bbox[1] == 28
