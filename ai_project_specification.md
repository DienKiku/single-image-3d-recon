# SYSTEM SPECIFICATION: 2D TO MULTI-LAYER INTERACTIVE 3D STUDIO (WITH METRIC SCALE & TEXTURE)

## 1. PROJECT OVERVIEW & CONTEXT
* **Goal:** Build a desktop/web-based application that transforms a single 2D image of a multi-part object into a **fully textured, metric-accurate, and multi-layer 3D model**.
* **Key Distinction:** This is not a generic generative 3D tool. It must accurately preserve **original textures (colors/materials)**, compute **real-world physical dimensions (mm/cm)**, and allow **layered disassembly (Exploded View)** for future hardware integration (e.g., 3D printing, robotic sorting).
* **Target Audience:** Open-source developers, hardware makers, and reverse-engineering enthusiasts on GitHub.

---

## 2. SYSTEM REQUIREMENTS (TECH STACK & HARDWARE)
* **Programming Language:** Python 3.10+
* **Core AI Backends (Open-Source Modules):**
    * *Computer Vision & Scaling:* `OpenCV` (for AprilTag/ArUco Marker detection to anchor physical scale).
    * *2D Layer Segmentation:* Meta's `SAM 2 (Segment Anything Model 2)` via `torch` or Hugging Face.
    * *2D-to-3D Generation:* `Unique3D` or `TRELLIS` (for high-fidelity PBR texture and mesh generation).
* **3D Geometry Processing:** `Trimesh` and `Open3D` (for file manipulation, scaling, and transformation matrix computation).
* **Frontend & 3D Rendering:** `Streamlit` combined with `streamlit-threejs` or HTML/JS custom component utilizing `Three.js` (React Three Fiber via iframe is also acceptable).
* **Hardware Requirements (Minimum):** NVIDIA GPU with 8GB+ VRAM (for local AI inference) OR fallback to cloud inference API endpoints (Replicate/Hugging Face). *Please structure the backend code modularly so APIs can be swapped easily.*

---

## 3. UI/UX & INTERFACE REQUIREMENTS
The interface must be clean, scannable, and developer-friendly. It requires a split-pane layout:

* **Left Control Panel (Sidebar):**
    * File Uploader: Supports `.jpg`, `.jpeg`, `.png`.
    * Reference Object Configuration: Dropdown to select reference marker (e.g., "ArUco Marker (5x5cm)", "Standard ID Card").
    * Action Buttons: `Analyze & Segment`, `Generate 3D Layers`, `Export Project`.
    * Exploded View Controller: A horizontal **Slider (0% to 100%)** to control the separation distance between generated 3D layers.
* **Right Interactive Viewport (Main Canvas):**
    * **Tab 1: 2D Analysis View:** Displays the uploaded image overlayed with colored masks of segmented parts from SAM 2 and bounding boxes showing calculated real-world dimensions.
    * **Tab 2: 3D Studio Canvas:** A WebGL canvas powered by `Three.js` displaying the composite 3D object. Must support standard mouse controls: left-click to orbit/rotate, right-click to pan, scroll to zoom.
    * **Metadata Overlay:** A dynamic HUD display showing: Total Parts Detected, Overall Dimensions (W x H x D in mm), Material Type estimation.

---

## 4. CORE REVERSE-ENGINEERING PIPELINE (STEP-BY-STEP EXECUTION)

### Step 1: Physical Scale Calibration (Computer Vision)
* Scan the raw input image for an ArUco Marker or predefined reference object using `cv2.aruco`.
* Calculate the pixel-to-millimeter ratio based on the known physical size of the marker.
* If no marker is detected, fallback to an interactive prompt allowing the user to input a single known dimension (e.g., "This object is 10cm wide").

### Step 2: Layer & Object Segmentation (AI Engine)
* Pass the input image to the `SAM 2` model.
* Generate distinct, clean 2D alpha masks for each visual layer/component of the object.
* Assign an incremental ID to each isolated layer (`layer_01.png`, `layer_02.png`, etc.).

### Step 3: High-Fidelity 3D Generation & Texturing (AI Engine)
* For each segmented 2D layer, apply an inpainting/padding preprocessing step to guess hidden edges.
* Feed the 2D component images into `Unique3D` or `TRELLIS` pipeline.
* **Enforce Output Assets:** For each layer, generate:
    1. A 3D Mesh file (`.obj` or `.stl`).
    2. A High-Resolution PBR Texture Map (`.png`).
* Apply the pixel-to-mm ratio calculated in Step 1 to scale the generated meshes to exact metric proportions.

### Step 4: Spatial Integration & Exploded View Math (3D Engine)
* Align all generated 3D meshes inside the `Three.js` scene based on their original 2D relative centers.
* Define an expansion vector relative to the object's global centroid for each layer.
* **Exploded View Formula:** When the UI slider value changes, update the 3D position of each layer using the expansion vector and distance factor to smoothly separate the components.

---

## 5. TESTING & VERIFICATION PLAN

### Unit Testing (Mocked Backend)
* Test the `cv2.aruco` detection script with a pre-shot sample image to verify it prints the correct pixel-to-mm scale.
* Verify that passing a multi-colored dummy image to the segmentation module returns a segmented dictionary of at least 2 distinct masks.

### Integration UI Testing
* Ensure that adjusting the UI Exploded View slider actively changes the transformation matrix of the meshes in the 3D canvas without lagging or breaking textures.
* Verify that exporting the project outputs a unified zip folder containing: `project_metadata.json`, individual layer `.obj` meshes, and `.png` texture maps.

---

## 6. FINAL EXPECTED ARTIFACTS & DELIVERABLES
Upon execution, the code architecture should deliver:
1. **Fully Functional Source Code:** Adhering to clean code principles with separate files for backend processing (`ai_processor.py`), geometric math (`geometry_utils.py`), and frontend layout (`app.py`).
2. **Comprehensive README.md:** Containing installation instructions, asset paths, model weight download links, and an architecture diagram.
3. **Export Package:** A valid, standard 3D file structure ready to be fed into slicer software (like Cura or PrusaSlicer) for physical fabrication.

---

## AI INSTRUCTION / INITIATION PROMPT:
> "Analyze the technical specifications provided above. Act as a senior AI Engineer and Full-stack Developer. Begin by generating the foundational folder structure and the complete, working Python implementation for `geometry_utils.py` (scale calculation) and the main Streamlit application UI skeleton. Write robust, modular code in active voice with zero pseudocode."
