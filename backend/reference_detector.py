"""
Reference detection module for 2D to 3D computer vision tasks.
"""

import math
from typing import List, Dict, Optional, Tuple, Any

import cv2
import numpy as np


def order_quadrilateral_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 points into [top-left, top-right, bottom-right, bottom-left].

    Args:
        pts (np.ndarray): Array of 4 points, shape (4, 2).

    Returns:
        np.ndarray: Array of 4 points ordered as top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def measure_pixel_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Computes the Euclidean distance between 2 points.

    Args:
        p1 (Tuple[float, float]): The first point (x, y).
        p2 (Tuple[float, float]): The second point (x, y).

    Returns:
        float: Euclidean distance between p1 and p2.
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def detect_rectangular_reference(
    image: np.ndarray,
    expected_ratio: float = 1.586,
    ratio_tolerance: float = 0.18,
    min_area_fraction: float = 0.005,
) -> List[Dict[str, Any]]:
    """
    Detects rectangular reference objects in an image.

    Converts to gray, applies GaussianBlur, Canny edge detection, and morphological close.
    Finds external contours, filters by minimum area, and uses cv2.approxPolyDP to detect
    4-sided convex polygons. Sorts corners and checks aspect ratio.

    Args:
        image (np.ndarray): The input image (BGR or grayscale).
        expected_ratio (float): Expected aspect ratio (longer side / shorter side).
            Default is 1.586 (ISO ID card).
        ratio_tolerance (float): Tolerance for aspect ratio matching.
        min_area_fraction (float): Minimum area fraction of the total image size.

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing detection results, sorted by confidence.
            Each dictionary has:
                "corners" (np.ndarray): Shape (4, 2), ordered corners.
                "width_px" (float): Width in pixels.
                "height_px" (float): Height in pixels.
                "aspect_ratio" (float): Aspect ratio.
                "center" (np.ndarray): Shape (2,), center point (x, y).
                "confidence" (float): Confidence score based on ratio match and area.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 150)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_area = image.shape[0] * image.shape[1]
    min_area = total_area * min_area_fraction
    
    results = []
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
            
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2)
            ordered_pts = order_quadrilateral_points(pts)
            
            (tl, tr, br, bl) = ordered_pts
            
            width_a = measure_pixel_distance(tuple(br), tuple(bl))
            width_b = measure_pixel_distance(tuple(tr), tuple(tl))
            width = max(width_a, width_b)
            
            height_a = measure_pixel_distance(tuple(tr), tuple(br))
            height_b = measure_pixel_distance(tuple(tl), tuple(bl))
            height = max(height_a, height_b)
            
            if width == 0 or height == 0:
                continue
                
            longer_side = max(width, height)
            shorter_side = min(width, height)
            aspect_ratio = longer_side / shorter_side
            
            if abs(aspect_ratio - expected_ratio) <= ratio_tolerance:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                else:
                    cx = np.mean(ordered_pts[:, 0])
                    cy = np.mean(ordered_pts[:, 1])
                    
                center = np.array([cx, cy])
                
                # Confidence based on closeness to expected ratio (higher is better, max 1.0)
                # and size relative to image
                ratio_error = abs(aspect_ratio - expected_ratio) / expected_ratio
                ratio_conf = max(0.0, 1.0 - ratio_error)
                size_conf = min(1.0, area / total_area * 10) # arbitrary scaling for size
                confidence = (ratio_conf * 0.7) + (size_conf * 0.3)
                
                results.append({
                    "corners": ordered_pts,
                    "width_px": float(width),
                    "height_px": float(height),
                    "aspect_ratio": float(aspect_ratio),
                    "center": center,
                    "confidence": float(confidence)
                })
                
    # Sort results by confidence descending
    results.sort(key=lambda x: x["confidence"], reverse=True)
    
    return results


def draw_ruler_annotation(
    image: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    distance_mm: Optional[float] = None,
    color: Tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    """
    Draws a ruler-style line between p1 and p2 with distance label.

    Args:
        image (np.ndarray): The image to draw on.
        p1 (Tuple[int, int]): First endpoint (x, y).
        p2 (Tuple[int, int]): Second endpoint (x, y).
        distance_mm (Optional[float]): Distance in mm. If None, uses pixels.
        color (Tuple[int, int, int]): Color of the annotation (B, G, R).

    Returns:
        np.ndarray: The image with the annotation drawn.
    """
    out_image = image.copy()
    
    # Draw line with tick marks / arrows
    cv2.line(out_image, p1, p2, color, 2)
    cv2.circle(out_image, p1, 4, color, -1)
    cv2.circle(out_image, p2, 4, color, -1)
    
    # Midpoint for label
    mid_x = (p1[0] + p2[0]) // 2
    mid_y = (p1[1] + p2[1]) // 2
    
    if distance_mm is not None:
        label = f"{distance_mm:.1f} mm"
    else:
        dist_px = measure_pixel_distance(p1, p2)
        label = f"{dist_px:.1f} px"
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # Text background
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.rectangle(out_image, (mid_x - text_w // 2 - 2, mid_y - text_h // 2 - 2), 
                  (mid_x + text_w // 2 + 2, mid_y + text_h // 2 + baseline + 2), 
                  (0, 0, 0), -1)
                  
    cv2.putText(out_image, label, (mid_x - text_w // 2, mid_y + text_h // 2), 
                font, font_scale, color, thickness)
                
    return out_image


def draw_detected_rectangle(
    image: np.ndarray,
    rect_info: Dict[str, Any],
    label: str = "Reference Card",
    pixel_to_mm: Optional[float] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    Draws a detected rectangle boundary and text annotation.

    Args:
        image (np.ndarray): The image to draw on.
        rect_info (Dict[str, Any]): The rectangle information dictionary.
        label (str): Label text for the rectangle.
        pixel_to_mm (Optional[float]): Conversion factor from pixels to mm.
        color (Tuple[int, int, int]): Color of the boundary and text (B, G, R).

    Returns:
        np.ndarray: The image with the rectangle drawn.
    """
    out_image = image.copy()
    
    corners = rect_info["corners"].astype(int)
    
    # Draw boundary
    cv2.polylines(out_image, [corners], isClosed=True, color=color, thickness=2)
    
    # Corner markers
    for pt in corners:
        cv2.circle(out_image, tuple(pt), 4, (0, 0, 255), -1)
        
    # Text annotation
    tl = corners[0]
    
    text = label
    if pixel_to_mm is not None:
        w_mm = rect_info["width_px"] * pixel_to_mm
        h_mm = rect_info["height_px"] * pixel_to_mm
        text += f" ({w_mm:.1f}x{h_mm:.1f} mm)"
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # Text background
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(out_image, (tl[0], tl[1] - text_h - 10), 
                  (tl[0] + text_w, tl[1]), 
                  (0, 0, 0), -1)
                  
    cv2.putText(out_image, text, (tl[0], tl[1] - 5), font, font_scale, color, thickness)
    
    return out_image
