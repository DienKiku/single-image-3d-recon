from pathlib import Path
import cv2

SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png"]

REFERENCE_OBJECTS = {
    "None / Auto (Arbitrary Object)": None,
    "Office Printer / Photocopier (1150mm H)": {"width": 600.0, "height": 1150.0, "depth": 650.0},
    "ArUco Marker (5x5cm)": 50.0,
    "ArUco Marker (10x10cm)": 100.0,
    "ISO/IEC 7810 ID Card": {"width": 85.6, "height": 53.98},
    "A4 Paper": {"width": 210.0, "height": 297.0},
    "Interactive Ruler (2 Points)": None,
    "Custom (Manual Input)": None
}

SEMANTIC_OBJECT_DIMENSIONS = {
    "Office Printer / Photocopier": (600.0, 1150.0, 650.0),
    "Laptop / Notebook": (320.0, 18.0, 220.0),
    "Desktop Tower": (200.0, 450.0, 450.0),
    "Coffee Mug / Cup": (85.0, 95.0, 85.0),
    "Smartphone": (75.0, 150.0, 8.0),
}

ARUCO_DICT_TYPE = cv2.aruco.DICT_6X6_250
DEFAULT_MARKER_SIZE_MM = 50.0
OUTPUT_DIR = Path("output")
ASSETS_DIR = Path("assets")
SAMPLES_DIR = ASSETS_DIR / "samples"

# AI Backend: 100% Offline Local (Stable Fast 3D)
DEFAULT_MESH_GENERATOR = "sf3d"

