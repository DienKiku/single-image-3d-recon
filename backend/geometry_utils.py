import dataclasses
import numpy as np
import cv2
import trimesh
from config.settings import REFERENCE_OBJECTS

@dataclasses.dataclass
class ScaleCalibration:
    """Dataclass holding calibration details for image to real-world scale."""
    pixel_to_mm: float
    marker_detected: bool
    marker_id: int | None
    marker_corners: np.ndarray | None
    annotated_image: np.ndarray | None


def detect_aruco_markers(
    image: np.ndarray,
    aruco_dict_type: int = cv2.aruco.DICT_6X6_250,
) -> list[dict]:
    """
    Detect ArUco markers in an image.

    Args:
        image: The input image.
        aruco_dict_type: The predefined ArUco dictionary type.

    Returns:
        A list of dictionaries, each containing 'id', 'corners', and 'center' of a detected marker.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    
    corners, ids, rejected = detector.detectMarkers(image)
    
    markers = []
    if ids is not None and len(ids) > 0:
        for i in range(len(ids)):
            marker_id = int(np.ravel(ids[i])[0])
            marker_corners = np.reshape(corners[i], (4, 2)).astype(np.float32)
            center = np.mean(marker_corners, axis=0)
            markers.append({
                "id": marker_id,
                "corners": marker_corners,
                "center": center
            })
    return markers


def compute_pixel_to_mm_ratio(
    marker_corners: np.ndarray,
    known_size_mm: float,
) -> float:
    """
    Compute the pixel to mm ratio based on the physical size of a detected marker.

    Args:
        marker_corners: The 4 corners of the marker in pixels (shape: 4x2).
        known_size_mm: The physical size of the marker side in mm.

    Returns:
        The ratio of physical size (mm) per pixel.

    Raises:
        ValueError: If the mean edge length in pixels is near zero.
    """
    edge_lengths = []
    for i in range(4):
        p1 = marker_corners[i]
        p2 = marker_corners[(i + 1) % 4]
        edge_lengths.append(np.linalg.norm(p1 - p2))
    
    mean_edge_pixels = float(np.mean(edge_lengths))
    
    if mean_edge_pixels < 1e-5:
        raise ValueError("Mean edge pixels is near zero, invalid marker corners.")
        
    return known_size_mm / mean_edge_pixels


def calibrate_scale(
    image: np.ndarray,
    reference_type: str = "ArUco Marker (5x5cm)",
    known_size_mm: float | None = None,
    manual_pixel_distance: float | None = None,
    manual_real_mm: float | None = None,
    target_bbox: tuple[int, int, int, int] | None = None,
) -> ScaleCalibration:
    """
    Calibrate the scale of the image to physical dimensions.

    Args:
        image: The input image.
        reference_type: The type of reference object used for scale.
        known_size_mm: Overrides the size defined by reference_type.
        manual_pixel_distance: Manual pixel distance for custom calibration.
        manual_real_mm: Manual real-world mm size for custom calibration.
        target_bbox: Optional bounding box (x, y, w, h) of the target object.

    Returns:
        ScaleCalibration object with the computed ratio and annotation details.
    """
    if reference_type in ("None / Auto (Arbitrary Object)", "None", "Auto", None):
        obj_h = float(target_bbox[3]) if target_bbox and len(target_bbox) >= 4 else float(image.shape[0])
        # Default ~350mm height for arbitrary objects to ensure proportional 3D visualization
        ratio = 350.0 / obj_h if obj_h > 0 else 1.0
        return ScaleCalibration(
            pixel_to_mm=ratio,
            marker_detected=False,
            marker_id=None,
            marker_corners=None,
            annotated_image=image.copy()
        )

    if reference_type == "Custom (Manual Input)" or (manual_pixel_distance and manual_real_mm):
        if not manual_pixel_distance or not manual_real_mm:
            raise ValueError("Manual pixel distance and real mm must be provided for custom calibration.")
        if manual_pixel_distance <= 0 or manual_real_mm <= 0:
            raise ValueError("Manual pixel distance and real mm must be positive values.")
            
        ratio = manual_real_mm / manual_pixel_distance
        return ScaleCalibration(
            pixel_to_mm=ratio,
            marker_detected=False,
            marker_id=None,
            marker_corners=None,
            annotated_image=image.copy()
        )

    annotated_image = image.copy()

    # Check for Semantic Object Dimension presets (e.g. Printer 1150mm H)
    ref_obj = REFERENCE_OBJECTS.get(reference_type)
    if isinstance(ref_obj, dict) and "height" in ref_obj and ("Printer" in reference_type or "Office" in reference_type or "Laptop" in reference_type):
        real_h = float(ref_obj["height"])
        obj_h = float(target_bbox[3]) if target_bbox and len(target_bbox) >= 4 else float(image.shape[0])
        ratio = real_h / obj_h if obj_h > 0 else 1.0
        return ScaleCalibration(
            pixel_to_mm=ratio,
            marker_detected=True,
            marker_id=None,
            marker_corners=None,
            annotated_image=annotated_image
        )

    # Check for Rectangular Reference Objects (ID Card, A4 paper)
    from backend.reference_detector import detect_rectangular_reference, draw_detected_rectangle
    if isinstance(ref_obj, dict) and "width" in ref_obj and "height" in ref_obj:
        real_w = float(ref_obj["width"])
        real_h = float(ref_obj["height"])
        expected_ratio = max(real_w, real_h) / min(real_w, real_h)
        rectangles = detect_rectangular_reference(image, expected_ratio=expected_ratio)
        if rectangles:
            best_rect = rectangles[0]
            long_edge_px = max(best_rect["width_px"], best_rect["height_px"])
            ratio = max(real_w, real_h) / long_edge_px if long_edge_px > 0 else 0.0
            annotated_image = draw_detected_rectangle(
                annotated_image,
                best_rect,
                label=reference_type,
                pixel_to_mm=ratio
            )
            return ScaleCalibration(
                pixel_to_mm=ratio,
                marker_detected=True,
                marker_id=None,
                marker_corners=best_rect["corners"],
                annotated_image=annotated_image
            )
        
    markers = detect_aruco_markers(image)
    
    if markers:
        marker = markers[0]
        ref_size = known_size_mm if known_size_mm is not None else REFERENCE_OBJECTS.get(reference_type)
        if isinstance(ref_size, dict):
            ref_size = ref_size.get("width", 50.0)
            
        try:
            ratio = compute_pixel_to_mm_ratio(marker["corners"], float(ref_size))
        except ValueError:
            ratio = 0.0
            
        corners_array = [np.array(marker["corners"], dtype=np.float32).reshape(1, 4, 2)]
        ids_array = np.array([marker["id"]], dtype=np.int32)
        cv2.aruco.drawDetectedMarkers(annotated_image, corners_array, ids_array)
        cv2.putText(
            annotated_image, 
            f"Scale: {ratio:.4f} mm/px", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.0, 
            (0, 255, 0), 
            2
        )
        
        return ScaleCalibration(
            pixel_to_mm=ratio,
            marker_detected=True,
            marker_id=marker["id"],
            marker_corners=marker["corners"],
            annotated_image=annotated_image
        )
        
    return ScaleCalibration(
        pixel_to_mm=0.0,
        marker_detected=False,
        marker_id=None,
        marker_corners=None,
        annotated_image=annotated_image
    )


def scale_mesh_to_metric(
    mesh: trimesh.Trimesh,
    pixel_to_mm: float,
    original_pixel_bbox: tuple[float, float, float, float],
) -> trimesh.Trimesh:
    """
    Scale a mesh to real-world metric dimensions based on image calibration.

    Args:
        mesh: The input trimesh object.
        pixel_to_mm: The scale ratio in mm/pixel.
        original_pixel_bbox: The bounding box in pixels (x, y, width_px, height_px).

    Returns:
        The scaled mesh.
    """
    _, _, width_px, _ = original_pixel_bbox
    target_width_mm = width_px * pixel_to_mm
    
    scaled_mesh = mesh.copy()
    extents = scaled_mesh.extents
    
    # Use max extent on X or Y to determine the current 'width' to scale. 
    # Usually X is width for meshes, let's use X extent. 
    current_width = extents[0] 
    if current_width < 1e-5:
        current_width = 1e-5
        
    scale_factor = target_width_mm / current_width
    scaled_mesh.apply_scale(scale_factor)
    
    return scaled_mesh


def compute_exploded_positions(
    centroids: list[np.ndarray],
    global_centroid: np.ndarray,
    expansion_factor: float,
    axis: int | None = None,
    total_span: float | None = None,
) -> list[np.ndarray]:
    """Compute offset positions for an exploded view of layers.

    Args:
        centroids: List of layer centroid positions.
        global_centroid: The global center of the object.
        expansion_factor: The factor to expand by (0.0 to 1.0+).
        axis: Optional axis index (0 for X, 1 for Y, 2 for Z) to restrict explosion strictly along that axis.
        total_span: Optional total dimension along the axis to ensure proportional physical displacement.

    Returns:
        A list of offset vectors for each centroid.
    """
    if len(centroids) <= 1:
        return [np.zeros(3)]
        
    num_layers = len(centroids)
    
    if axis is not None:
        step = (total_span * 0.40) if (total_span is not None and total_span > 1.0) else 100.0
        axis_coords = [float(c[axis]) for c in centroids]
        sorted_indices = np.argsort(axis_coords)
        ranks = np.zeros(num_layers, dtype=float)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = (rank - (num_layers - 1) / 2.0)
            
        offsets = []
        for i in range(num_layers):
            vec = np.zeros(3, dtype=float)
            vec[axis] = ranks[i] * step * expansion_factor
            offsets.append(vec)
        return offsets

    offsets = []
    for centroid in centroids:
        offset = (centroid - global_centroid) * expansion_factor
        offsets.append(offset)
        
    return offsets


def annotate_dimensions(
    image: np.ndarray,
    bounding_boxes: list[tuple[int, int, int, int]],
    pixel_to_mm: float,
    labels: list[str] | None = None,
) -> np.ndarray:
    """
    Draw bounding boxes and size annotations on an image.

    Args:
        image: The input image.
        bounding_boxes: List of bounding boxes as (x, y, w, h).
        pixel_to_mm: Scale ratio.
        labels: Optional list of labels for each box.

    Returns:
        The annotated image.
    """
    annotated = image.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for i, bbox in enumerate(bounding_boxes):
        x, y, w, h = bbox
        color = colors[i % len(colors)]
        
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        
        width_mm = w * pixel_to_mm
        height_mm = h * pixel_to_mm
        text = f"{width_mm:.1f}x{height_mm:.1f}mm"
        
        if labels and i < len(labels):
            text = f"{labels[i]} - " + text
            
        cv2.putText(annotated, text, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
    return annotated
