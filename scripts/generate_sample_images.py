"""Generate sample test images containing ArUco markers for scale calibration.

This script creates two sample images:
1. sample_aruco_objects.png (1280x960): Contains an ArUco marker (ID 42)
   and three colored labeled rectangles simulating object parts.
2. sample_aruco_simple.png (640x480): Contains a centered ArUco marker (ID 7).

Both generated images are verified by running ArUco detection to ensure they
are detectable by computer vision pipelines.
"""

from pathlib import Path
import cv2
import numpy as np


def create_aruco_marker(
    marker_id: int,
    marker_size_pixels: int,
    dict_type: int = cv2.aruco.DICT_6X6_250,
) -> np.ndarray:
    """Generate an ArUco marker as a 3-channel BGR image.

    Args:
        marker_id: The integer ID of the marker to generate.
        marker_size_pixels: The width and height of the marker in pixels.
        dict_type: The OpenCV ArUco predefined dictionary identifier.

    Returns:
        A 3-channel BGR numpy array containing the marker image.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
    marker_mono = cv2.aruco.generateImageMarker(
        aruco_dict,
        marker_id,
        marker_size_pixels,
    )
    return cv2.cvtColor(marker_mono, cv2.COLOR_GRAY2BGR)


def generate_objects_sample(output_path: Path) -> Path:
    """Generate a 1280x960 test image with marker ID 42 and colored object parts.

    Args:
        output_path: Destination file path for the generated image.

    Returns:
        The path where the image was saved.
    """
    width = 1280
    height = 960
    image = np.full((height, width, 3), 255, dtype=np.uint8)

    # Generate and place marker ID=42 at (20, 20) with 200x200 px
    marker_id = 42
    marker_size = 200
    padding = 20
    marker_bgr = create_aruco_marker(marker_id, marker_size)
    image[padding:padding + marker_size, padding:padding + marker_size] = marker_bgr

    # Define parts: (top_left, bottom_right, fill_bgr, label)
    parts = [
        (
            (400, 200),
            (650, 500),
            (50, 50, 220),  # Soft red
            "Part A",
        ),
        (
            (500, 350),
            (800, 650),
            (50, 180, 50),  # Soft green
            "Part B",
        ),
        (
            (700, 150),
            (950, 550),
            (210, 120, 40),  # Soft blue
            "Part C",
        ),
    ]

    # Draw colored rectangles with thin black borders and text labels
    for pt1, pt2, fill_color, label in parts:
        # Fill rectangle
        cv2.rectangle(image, pt1, pt2, fill_color, thickness=-1)
        # Draw black border
        cv2.rectangle(image, pt1, pt2, (0, 0, 0), thickness=2)
        # Draw label text above rectangle
        label_pos = (pt1[0], pt1[1] - 12)
        cv2.putText(
            image,
            label,
            label_pos,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.75,
            color=(0, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return output_path


def generate_simple_sample(output_path: Path) -> Path:
    """Generate a 640x480 test image with a centered marker ID 7.

    Args:
        output_path: Destination file path for the generated image.

    Returns:
        The path where the image was saved.
    """
    width = 640
    height = 480
    image = np.full((height, width, 3), 255, dtype=np.uint8)

    # Generate and place marker ID=7 at center with 150x150 px
    marker_id = 7
    marker_size = 150
    x_offset = (width - marker_size) // 2
    y_offset = (height - marker_size) // 2

    marker_bgr = create_aruco_marker(marker_id, marker_size)
    image[y_offset:y_offset + marker_size, x_offset:x_offset + marker_size] = marker_bgr

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return output_path


def verify_marker_detection(
    image_path: Path,
    expected_id: int,
    dict_type: int = cv2.aruco.DICT_6X6_250,
) -> list[int]:
    """Load an image from disk and detect ArUco markers to verify detectability.

    Args:
        image_path: Path to the image file.
        expected_id: The marker ID expected to be found.
        dict_type: The OpenCV ArUco predefined dictionary identifier.

    Returns:
        A list of integer marker IDs detected in the image.

    Raises:
        FileNotFoundError: If the image file cannot be read.
        AssertionError: If the expected marker ID is not detected.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image at {image_path}")

    dictionary = cv2.aruco.getPredefinedDictionary(dict_type)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    corners, ids, _ = detector.detectMarkers(image)

    detected_ids: list[int] = []
    if ids is not None and len(ids) > 0:
        detected_ids = [int(x) for x in np.ravel(ids)]

    if expected_id not in detected_ids:
        raise AssertionError(
            f"Verification failed for {image_path.name}: "
            f"Expected ID {expected_id}, but detected {detected_ids}"
        )

    return detected_ids


def main() -> None:
    """Run sample image generation and verification workflow."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    samples_dir = project_root / "assets" / "samples"

    print("=" * 60)
    print("Generating ArUco sample images...")
    print(f"Output directory: {samples_dir}")
    print("=" * 60)

    # 1. Generate Image 1
    image1_path = samples_dir / "sample_aruco_objects.png"
    print(f"[1/2] Creating sample_aruco_objects.png (1280x960, ID=42)...")
    generate_objects_sample(image1_path)
    print(f"      Saved to: {image1_path}")

    # 2. Generate Image 2
    image2_path = samples_dir / "sample_aruco_simple.png"
    print(f"[2/2] Creating sample_aruco_simple.png (640x480, ID=7)...")
    generate_simple_sample(image2_path)
    print(f"      Saved to: {image2_path}")

    # 3. Verify markers
    print("\nVerifying marker detectability...")

    ids1 = verify_marker_detection(image1_path, expected_id=42)
    print(f"  - {image1_path.name}: Detected IDs = {ids1} (Verification PASSED)")

    ids2 = verify_marker_detection(image2_path, expected_id=7)
    print(f"  - {image2_path.name}: Detected IDs = {ids2} (Verification PASSED)")

    print("\nAll sample images generated and verified successfully!")


if __name__ == "__main__":
    main()
