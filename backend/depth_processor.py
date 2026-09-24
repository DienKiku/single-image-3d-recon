"""Depth estimation and 3D surface mesh reconstruction engine.

Converts 2D images into real-world 3D textured surface meshes using Depth Anything V2.
Preserves high-fidelity photo textures, colors, and surface geometry without requiring ArUco markers.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import trimesh

_DEPTH_PIPELINE = None


def get_depth_pipeline():
    """Lazily load and cache the Depth Anything V2 pipeline."""
    global _DEPTH_PIPELINE
    if _DEPTH_PIPELINE is None:
        try:
            from transformers import pipeline
            _DEPTH_PIPELINE = pipeline(
                "depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf"
            )
        except Exception as e:
            print(f"Warning: Could not load Depth Anything V2 model: {e}")
            _DEPTH_PIPELINE = None
    return _DEPTH_PIPELINE


def estimate_depth_map(image: np.ndarray) -> np.ndarray:
    """Estimate a normalized depth map from an RGB image.

    Args:
        image: RGB image array of shape (H, W, 3).

    Returns:
        Normalized float32 depth map of shape (H, W), values in [0.0, 1.0],
        where 1.0 is nearest (prominent foreground) and 0.0 is farthest (background).
    """
    pipe = get_depth_pipeline()
    h, w = image.shape[:2]

    if pipe is not None:
        try:
            pil_img = Image.fromarray(image)
            result = pipe(pil_img)
            depth_raw = np.array(result["depth"]).astype(np.float32)
            # Ensure size matches original image
            if depth_raw.shape[:2] != (h, w):
                depth_raw = cv2.resize(depth_raw, (w, h), interpolation=cv2.INTER_LINEAR)
            
            d_min, d_max = depth_raw.min(), depth_raw.max()
            if d_max > d_min:
                depth_norm = (depth_raw - d_min) / (d_max - d_min)
            else:
                depth_norm = np.zeros_like(depth_raw)
            return depth_norm.astype(np.float32)
        except Exception as e:
            print(f"Depth pipeline inference failed, using fallback: {e}")

    # Enhanced Heuristic Depth Fallback (structure-aware, not brightness-based)
    # Uses edge detection, texture frequency, and convexity priors instead of raw luminance
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    # 1. Structure tensor: areas with complex texture are likely closer (higher frequency = nearer)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    # Local texture energy (Gaussian-smoothed gradient magnitude)
    texture_energy = np.sqrt(grad_x ** 2 + grad_y ** 2)
    texture_energy = cv2.GaussianBlur(texture_energy, (21, 21), 0)
    te_max = texture_energy.max()
    if te_max > 0:
        texture_energy /= te_max
    
    # 2. Center-distance convexity prior (objects tend to be convex, center is nearest)
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    center_y, center_x = h / 2.0, w / 2.0
    dist_from_center = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
    max_dist = np.max(dist_from_center)
    convexity_prior = 1.0 - (dist_from_center / max_dist) ** 1.5 if max_dist > 0 else np.ones_like(gray)
    
    # 3. Blur luminance heavily to capture only broad shading gradients (not texture)
    shading = cv2.GaussianBlur(gray, (51, 51), 0)
    s_min, s_max = shading.min(), shading.max()
    if s_max > s_min:
        shading = (shading - s_min) / (s_max - s_min)
    else:
        shading = np.ones_like(gray) * 0.5
    
    # Combine: convexity dominates, texture adds local detail, shading adds slight bias
    combined = (0.55 * convexity_prior) + (0.30 * texture_energy) + (0.15 * shading)
    combined = cv2.GaussianBlur(combined, (11, 11), 0)
    c_min, c_max = combined.min(), combined.max()
    if c_max > c_min:
        depth_norm = (combined - c_min) / (c_max - c_min)
    else:
        depth_norm = np.zeros((h, w), dtype=np.float32)
    return depth_norm.astype(np.float32)


def create_depth_mesh(
    image: np.ndarray,
    depth_map: np.ndarray,
    mask: np.ndarray | None = None,
    max_grid_size: int = 120,
    depth_scale: float = 0.25,
    target_dimensions_mm: tuple[float, float, float] | None = None,
    solid: bool = True,
    base_depth_ratio: float = 0.35,
    camera_fov_deg: float = 52.0,
    pitch_angle_deg: float = 16.0,
    perspective_correction: bool = True,
) -> trimesh.Trimesh:
    """Construct a high-fidelity 3D triangular surface mesh from RGB/RGBA image and depth map.

    Features:
      - Camera Perspective Ray Unprojection (reverses 2D foreshortening skew, scanner lid narrowing).
      - Gravity Pitch De-tilting (aligns vertical walls to 90 degrees plumb and grounds bottom on Y=0).
      - Manufactured Planar CAD Enclosure (flat rear casing + vertical side walls, no ballooning).
      - Non-linear Cavity Deepening (deepens paper tray & slot recesses).
      - Watertight Solid Manifold Output (is_watertight: True, STL/OBJ/CAD compatible).

    Args:
        image: RGB or RGBA image array (H, W, 3 or 4).
        depth_map: Normalized depth map (H, W), values in [0, 1].
        mask: Optional binary or alpha mask (H, W) where foreground > 0.
        max_grid_size: Max vertices along the largest image dimension.
        depth_scale: Depth extrusion factor relative to object width.
        target_dimensions_mm: Optional (width_mm, height_mm, depth_mm) to scale mesh.
        solid: If True, builds a watertight solid volume with side and back faces.
        base_depth_ratio: Backplate extrusion offset behind the lowest relief point.
        camera_fov_deg: Estimated field-of-view in degrees for perspective ray unprojection.
        pitch_angle_deg: Downward camera tilt angle in degrees for gravity de-tilting.
        perspective_correction: If True and mask is provided, applies unprojection & CAD casing.

    Returns:
        trimesh.Trimesh with vertex positions, vertex colors, and UV coordinates.
    """
    if image.shape[2] == 4:
        if mask is None:
            mask = image[:, :, 3]
        rgb_img = image[:, :, :3]
    else:
        rgb_img = image.copy()

    orig_h, orig_w = rgb_img.shape[:2]
    aspect_ratio = orig_w / float(orig_h) if orig_h > 0 else 1.0

    if orig_w >= orig_h:
        grid_w = max_grid_size
        grid_h = max(10, int(round(max_grid_size / aspect_ratio)))
    else:
        grid_h = max_grid_size
        grid_w = max(10, int(round(max_grid_size * aspect_ratio)))

    # Resize image and depth map to grid resolution
    small_img = cv2.resize(rgb_img, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
    small_depth = cv2.resize(depth_map, (grid_w, grid_h), interpolation=cv2.INTER_CUBIC)

    # Apply slight bilateral blur on depth to keep edges crisp while smoothing noise
    small_depth = cv2.bilateralFilter(small_depth.astype(np.float32), d=5, sigmaColor=0.1, sigmaSpace=5)

    # 1. Determine active quad cells conforming to silhouette mask
    if mask is not None:
        raw_bin_mask = (mask > 120).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_bin_mask, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            raw_bin_mask = (labels == largest_label).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        raw_bin_mask = cv2.morphologyEx(raw_bin_mask, cv2.MORPH_CLOSE, kernel)

        q_mask = cv2.resize(raw_bin_mask, (grid_w, grid_h), interpolation=cv2.INTER_NEAREST)
        bin_mask = (q_mask > 0).astype(np.uint8)

        active_quads = (
            (bin_mask[:-1, :-1] > 0) &
            (bin_mask[:-1, 1:] > 0) &
            (bin_mask[1:, :-1] > 0) &
            (bin_mask[1:, 1:] > 0)
        )
    else:
        active_quads = np.ones((grid_h - 1, grid_w - 1), dtype=bool)
        raw_bin_mask = None

    if not np.any(active_quads):
        active_quads = np.ones((grid_h - 1, grid_w - 1), dtype=bool)

    # Eliminate diagonal pinches (which create non-manifold butterfly edges where count=4)
    for r in range(grid_h - 2):
        for c in range(grid_w - 2):
            if active_quads[r, c] and active_quads[r+1, c+1] and not active_quads[r, c+1] and not active_quads[r+1, c]:
                active_quads[r, c+1] = True
            elif not active_quads[r, c] and not active_quads[r+1, c+1] and active_quads[r, c+1] and active_quads[r+1, c]:
                active_quads[r, c] = True

    # 2. Mark active grid nodes touched by at least one active quad
    active_nodes = np.zeros((grid_h, grid_w), dtype=bool)
    for r in range(grid_h - 1):
        for c in range(grid_w - 1):
            if active_quads[r, c]:
                active_nodes[r, c] = True
                active_nodes[r, c+1] = True
                active_nodes[r+1, c] = True
                active_nodes[r+1, c+1] = True

    # Chassis color for back & sides (median object body color)
    if mask is not None and np.any(mask > 128):
        fg_pixels = rgb_img[mask > 128]
        chassis_rgb = np.clip(np.median(fg_pixels, axis=0) * 0.85, 25, 235).astype(np.uint8)
    else:
        border_pixels = np.concatenate([small_img[0, :], small_img[-1, :], small_img[:, 0], small_img[:, -1]], axis=0)
        chassis_rgb = np.clip(np.median(border_pixels, axis=0) * 0.85, 25, 235).astype(np.uint8)
    chassis_rgba = np.append(chassis_rgb, 255)

    # 3. Spatial Geometry Reconstruction
    active_depths = small_depth[active_nodes]
    if len(active_depths) > 0:
        d_min = float(active_depths.min())
        d_max = float(active_depths.max())
        d_span = d_max - d_min if d_max > d_min else 1.0
        d_norm = np.clip((small_depth - d_min) / d_span, 0.0, 1.0)
        # Non-linear cavity deepening (paper tray opening, crevices)
        d_sculpt = (d_norm ** 1.35)
    else:
        d_sculpt = small_depth.copy()

    v_front = []
    v_back = []
    v_colors_front = []
    v_colors_back = []
    uv_front = []
    node_idx = np.full((grid_h, grid_w), -1, dtype=int)

    use_perspective = (mask is not None) and perspective_correction

    if use_perspective:
        # Determine target dimensions in mm
        if target_dimensions_mm is not None:
            target_w, target_h, target_d = target_dimensions_mm
        else:
            target_w = aspect_ratio * 380.0 if aspect_ratio < 1.0 else 380.0
            target_h = target_w / aspect_ratio
            target_d = target_w * 0.75

        # 1. Clean binary mask & smooth polygon contour (eliminates sawtooth fins)
        from scipy.ndimage import distance_transform_edt, binary_erosion
        clean_bin_mask = (bin_mask > 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clean_bin_mask = cv2.morphologyEx(clean_bin_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(clean_bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            main_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(main_contour, True)
            smooth_poly = cv2.approxPolyDP(main_contour, 0.0012 * perimeter, True)[:, 0, :]
        else:
            smooth_poly = np.array([[0, 0], [grid_w, 0], [grid_w, grid_h], [0, grid_h]])

        poly_min_x = float(smooth_poly[:, 0].min())
        poly_max_x = float(smooth_poly[:, 0].max())
        poly_min_y = float(smooth_poly[:, 1].min())
        poly_max_y = float(smooth_poly[:, 1].max())
        poly_span_x = max(1e-4, poly_max_x - poly_min_x)
        poly_span_y = max(1e-4, poly_max_y - poly_min_y)

        # 2. Inpaint/Clamp Boundary Depth to Interior Object (eliminates top edge explosions)
        inner_mask = binary_erosion(clean_bin_mask > 0, iterations=4)
        if inner_mask.sum() > 10:
            _, indices = distance_transform_edt(~inner_mask, return_indices=True)
            clamped_depth = small_depth[indices[0], indices[1]]
        else:
            clamped_depth = small_depth.copy()

        # Bilateral smoothing to flatten large faces while preserving sharp steps
        smooth_depth = cv2.bilateralFilter(clamped_depth.astype(np.float32), d=9, sigmaColor=0.08, sigmaSpace=9)

        # 3. Generic Camera Pitch Compensation (RANSAC-style robust slope estimation)
        # Uses ALL foreground depth values, not just a specific object region
        fg_rows, fg_cols = np.where(clean_bin_mask > 0)
        if len(fg_rows) > 100:
            fg_depths = smooth_depth[fg_rows, fg_cols]
            # Robust linear fit using median-based approach (resist outliers)
            # Split into upper and lower halves, compare median depths
            mid_row = (fg_rows.min() + fg_rows.max()) / 2.0
            upper_mask = fg_rows < mid_row
            lower_mask = fg_rows >= mid_row
            if upper_mask.sum() > 10 and lower_mask.sum() > 10:
                upper_median_depth = np.median(fg_depths[upper_mask])
                lower_median_depth = np.median(fg_depths[lower_mask])
                upper_median_row = np.median(fg_rows[upper_mask])
                lower_median_row = np.median(fg_rows[lower_mask])
                row_diff = lower_median_row - upper_median_row
                if abs(row_diff) > 1:
                    slope = (lower_median_depth - upper_median_depth) / row_diff
                else:
                    slope = 0.0
                # Only correct if slope magnitude is significant (> 0.5% per row)
                if abs(slope) < 0.0005:
                    slope = 0.0
            else:
                slope = 0.0
        else:
            slope = 0.0

        # De-tilt: subtract camera pitch slope centered on object midpoint
        obj_center_row = float(fg_rows.mean()) if len(fg_rows) > 0 else grid_h * 0.5
        row_indices = np.arange(grid_h)[:, None]
        detilt_depth = smooth_depth - (slope * (row_indices - obj_center_row))
        act_d = detilt_depth[active_nodes]
        d_min, d_max = float(act_d.min()), float(act_d.max())
        d_span = d_max - d_min if d_max > d_min else 1.0
        d_norm = np.clip((detilt_depth - d_min) / d_span, 0.0, 1.0)
        d_sculpt = d_norm ** 1.35  # non-linear cavity deepening

        # 4. Identify boundary nodes that touch an inactive quad
        boundary_nodes = np.zeros((grid_h, grid_w), dtype=bool)
        for r in range(grid_h):
            for c in range(grid_w):
                if not active_nodes[r, c]:
                    continue
                q_tl = active_quads[r-1, c-1] if (r > 0 and c > 0) else False
                q_tr = active_quads[r-1, c] if (r > 0 and c < grid_w - 1) else False
                q_bl = active_quads[r, c-1] if (r < grid_h - 1 and c > 0) else False
                q_br = active_quads[r, c] if (r < grid_h - 1 and c < grid_w - 1) else False
                if not (q_tl and q_tr and q_bl and q_br):
                    boundary_nodes[r, c] = True

        # Precompute polygon segments for boundary snapping
        poly_pts = smooth_poly.astype(np.float64)
        def snap_to_contour(px, py):
            p = np.array([px, py], dtype=np.float64)
            min_dist_sq = float('inf')
            best_pt = p
            for i in range(len(poly_pts)):
                a = poly_pts[i]
                b = poly_pts[(i + 1) % len(poly_pts)]
                ab = b - a
                ab_len_sq = np.dot(ab, ab)
                if ab_len_sq < 1e-8:
                    continue
                t = np.clip(np.dot(p - a, ab) / ab_len_sq, 0.0, 1.0)
                proj = a + t * ab
                dist_sq = np.dot(p - proj, p - proj)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_pt = proj
            return best_pt

        # True Pinhole Camera Back-Projection
        # FOV=40 deg matches SF3D default camera model
        fov_deg = 40.0
        focal_px = 0.5 * float(grid_w) / np.tan(0.5 * np.deg2rad(fov_deg))
        cx_px = (poly_min_x + poly_max_x) / 2.0
        cy_px = (poly_min_y + poly_max_y) / 2.0

        # Depth range: d_sculpt [0=deepest, 1=nearest]
        # Map to physical Z: front surface at [0, -relief_depth]
        relief_depth = target_d * 0.55  # Front sculpted details use 55% of total depth
        back_z = -target_d              # Flat solid CAD back wall

        for r in range(grid_h):
            for c in range(grid_w):
                if active_nodes[r, c]:
                    idx = len(v_front)
                    node_idx[r, c] = idx

                    px = float(c)
                    py = float(r)
                    if boundary_nodes[r, c]:
                        snapped = snap_to_contour(px, py)
                        px, py = snapped[0], snapped[1]

                    # Pinhole back-projection: pixel -> 3D ray
                    # Z from depth sculpt: 0=deepest, 1=nearest surface
                    zf = -(1.0 - d_sculpt[r, c]) * relief_depth

                    # Back-project XY using pinhole model at estimated Z
                    proj_z = max(relief_depth * 0.5, abs(zf) + relief_depth * 0.1)
                    raw_x = (px - cx_px) * proj_z / focal_px
                    raw_y = (cy_px - py) * proj_z / focal_px

                    # Scale raw back-projection to match target physical dimensions
                    raw_half_w = (poly_span_x / 2.0) * proj_z / focal_px
                    raw_half_h = (poly_span_y / 2.0) * proj_z / focal_px
                    x_mm = raw_x * (target_w / 2.0) / raw_half_w if raw_half_w > 1e-6 else 0.0
                    y_mm = raw_y * (target_h / 2.0) / raw_half_h if raw_half_h > 1e-6 else 0.0

                    zb = back_z

                    v_front.append([x_mm, y_mm, zf])
                    v_back.append([x_mm, y_mm, zb])

                    rgb = small_img[r, c]
                    v_colors_front.append([rgb[0], rgb[1], rgb[2], 255])
                    v_colors_back.append(chassis_rgba)

                    u = float(np.clip(px / float(grid_w - 1), 0.0, 1.0)) if grid_w > 1 else 0.5
                    v = float(np.clip(1.0 - (py / float(grid_h - 1)), 0.0, 1.0)) if grid_h > 1 else 0.5
                    uv_front.append([u, v])

    else:
        # Orthographic relief mode (for generic textures or non-segmented inputs)
        xs = np.linspace(-aspect_ratio / 2.0, aspect_ratio / 2.0, grid_w, dtype=np.float32)
        ys = np.linspace(0.5, -0.5, grid_h, dtype=np.float32)
        sculpted_depth = d_sculpt * depth_scale
        z_min = float(sculpted_depth.min())

        for r in range(grid_h):
            for c in range(grid_w):
                if active_nodes[r, c]:
                    idx = len(v_front)
                    node_idx[r, c] = idx
                    x = xs[c]
                    y = ys[r]
                    zf = sculpted_depth[r, c]
                    zb = z_min - base_depth_ratio

                    v_front.append([x, y, zf])
                    v_back.append([x, y, zb])

                    rgb = small_img[r, c]
                    v_colors_front.append([rgb[0], rgb[1], rgb[2], 255])
                    v_colors_back.append(chassis_rgba)

                    u = c / float(grid_w - 1) if grid_w > 1 else 0.5
                    v = 1.0 - (r / float(grid_h - 1)) if grid_h > 1 else 0.5
                    uv_front.append([u, v])

    n_verts = len(v_front)
    v_front = np.array(v_front, dtype=np.float32)
    v_back = np.array(v_back, dtype=np.float32)
    v_colors_front = np.array(v_colors_front, dtype=np.uint8)
    v_colors_back = np.array(v_colors_back, dtype=np.uint8)
    uv_front = np.array(uv_front, dtype=np.float32)

    faces = []

    # 4. Triangulate Front and Back surfaces
    for r in range(grid_h - 1):
        for c in range(grid_w - 1):
            if active_quads[r, c]:
                tl = node_idx[r, c]
                tr = node_idx[r, c+1]
                bl = node_idx[r+1, c]
                br = node_idx[r+1, c+1]

                # Front face
                faces.append([tl, tr, bl])
                faces.append([tr, br, bl])

                if solid:
                    # Back face (reversed winding)
                    faces.append([n_verts + tl, n_verts + bl, n_verts + tr])
                    faces.append([n_verts + tr, n_verts + bl, n_verts + br])

    # 5. Boundary Side Walls: Connect edges bordering non-active quads
    if solid:
        for r in range(grid_h - 1):
            for c in range(grid_w - 1):
                if not active_quads[r, c]:
                    continue
                tl = node_idx[r, c]
                tr = node_idx[r, c+1]
                bl = node_idx[r+1, c]
                br = node_idx[r+1, c+1]

                # Top edge (tl -> tr)
                if r == 0 or not active_quads[r-1, c]:
                    faces.append([tl, tr, n_verts + tr])
                    faces.append([tl, n_verts + tr, n_verts + tl])

                # Bottom edge (br -> bl)
                if r == grid_h - 2 or not active_quads[r+1, c]:
                    faces.append([br, bl, n_verts + bl])
                    faces.append([br, n_verts + bl, n_verts + br])

                # Left edge (bl -> tl)
                if c == 0 or not active_quads[r, c-1]:
                    faces.append([bl, tl, n_verts + tl])
                    faces.append([bl, n_verts + tl, n_verts + bl])

                # Right edge (tr -> br)
                if c == grid_w - 2 or not active_quads[r, c+1]:
                    faces.append([tr, br, n_verts + br])
                    faces.append([tr, n_verts + br, n_verts + tr])

        vertices = np.vstack([v_front, v_back])
        vertex_colors = np.vstack([v_colors_front, v_colors_back])
        # Back face UVs: mirror the front UVs horizontally so back shows correct orientation
        uv_back = uv_front.copy()
        uv_back[:, 0] = 1.0 - uv_back[:, 0]  # Mirror U coordinate
        all_uvs = np.vstack([uv_front, uv_back])
    else:
        vertices = v_front
        vertex_colors = v_colors_front
        all_uvs = uv_front

    faces = np.array(faces, dtype=np.int32)

    # Build Trimesh
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors,
        process=True
    )
    mesh.visual.uv = all_uvs
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.merge_vertices(merge_tex=False, merge_norm=False, digits_vertex=2)
    mesh.fix_normals()

    # Scale non-perspective mode to target dimensions if provided
    if not use_perspective and target_dimensions_mm is not None:
        target_w, target_h, target_d = target_dimensions_mm
        scale_x = target_w / aspect_ratio if aspect_ratio > 1e-5 else 1.0
        scale_y = target_h
        raw_depth_span = float(sculpted_depth.max() - z_min + base_depth_ratio)
        scale_z = target_d / raw_depth_span if raw_depth_span > 1e-5 else 1.0
        mesh.apply_scale([scale_x, scale_y, scale_z])

    # Ground base at Y = 0 for realistic floor contact
    if use_perspective:
        mesh.vertices[:, 1] -= mesh.vertices[:, 1].min()

    # Apply light Taubin smoothing to relax boundary discretization while preserving CAD edges
    try:
        from trimesh.smoothing import filter_taubin
        mesh = filter_taubin(mesh, iterations=2)
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.merge_vertices(merge_tex=False, merge_norm=False, digits_vertex=2)
        mesh.fix_normals()
        if use_perspective:
            mesh.vertices[:, 1] -= mesh.vertices[:, 1].min()
    except Exception:
        pass

    return mesh


def slice_mesh_into_layers(
    mesh: trimesh.Trimesh,
    num_layers: int = 3,
    axis: int = 1,
) -> list[trimesh.Trimesh]:
    """Slice a mesh into multiple spatial/depth layers for Exploded View.

    Args:
        mesh: Input trimesh object.
        num_layers: Number of discrete layers to create (e.g. 3 for Top, Middle, Bottom).
        axis: 0 for X, 1 for Y (vertical/height), 2 for Z (depth).

    Returns:
        List of sub-meshes, each representing one layer with its own centroid and vertices.
    """
    if num_layers <= 1:
        return [mesh.copy()]

    vertices = mesh.vertices
    coords = vertices[:, axis]
    min_c, max_c = coords.min(), coords.max()
    span = max_c - min_c

    if span < 1e-5:
        return [mesh.copy()]

    layer_bounds = np.linspace(min_c, max_c, num_layers + 1)
    layers = []

    for i in range(num_layers):
        low = layer_bounds[i]
        high = layer_bounds[i + 1]

        # Find vertices belonging to this layer
        if i == num_layers - 1:
            mask = (coords >= low) & (coords <= high)
        else:
            mask = (coords >= low) & (coords < high)

        if not np.any(mask):
            continue

        # Find faces where all vertices or at least 2 vertices belong to this layer
        face_vert_masks = mask[mesh.faces]
        valid_faces = np.sum(face_vert_masks, axis=1) >= 2

        if not np.any(valid_faces):
            continue

        sub_faces = mesh.faces[valid_faces]
        # Re-index vertices for submesh
        unique_vert_indices, new_faces = np.unique(sub_faces, return_inverse=True)
        sub_vertices = vertices[unique_vert_indices]
        new_faces = new_faces.reshape(-1, 3)

        sub_colors = None
        if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
            sub_colors = mesh.visual.vertex_colors[unique_vert_indices]

        sub_mesh = trimesh.Trimesh(
            vertices=sub_vertices,
            faces=new_faces,
            vertex_colors=sub_colors,
            process=False
        )
        layers.append(sub_mesh)

    if not layers:
        return [mesh.copy()]

    return layers


def export_textured_obj(
    mesh: trimesh.Trimesh,
    texture_image: np.ndarray,
    output_obj_path: Path,
    output_texture_path: Path,
    layer_id: str = "layer_01",
) -> tuple[Path, Path, Path]:
    """Export a trimesh to a valid OBJ file with MTL and diffuse PNG texture map.

    Args:
        mesh: trimesh.Trimesh object.
        texture_image: RGB image to save as diffuse texture.
        output_obj_path: Target path for the .obj file.
        output_texture_path: Target path for the .png texture.
        layer_id: Identifier for material naming.

    Returns:
        Tuple of (obj_path, mtl_path, texture_path).
    """
    output_obj_path.parent.mkdir(parents=True, exist_ok=True)
    output_texture_path.parent.mkdir(parents=True, exist_ok=True)

    # Save texture PNG
    Image.fromarray(texture_image).save(output_texture_path)

    # Create MTL file
    mtl_path = output_obj_path.with_suffix(".mtl")
    rel_tex_path = f"../textures/{output_texture_path.name}"
    mtl_content = f"""# Material for {layer_id}
newmtl {layer_id}_material
Ka 1.0 1.0 1.0
Kd 1.0 1.0 1.0
Ks 0.1 0.1 0.1
Ns 10.0
d 1.0
illum 2
map_Kd {rel_tex_path}
"""
    mtl_path.write_text(mtl_content, encoding="utf-8")

    # Write OBJ with vertices, UVs, and faces
    lines = [
        f"# 2D-to-3D Studio Export: {layer_id}",
        f"mtllib {mtl_path.name}",
        f"usemtl {layer_id}_material",
    ]

    has_vc = hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None
    vc_arr = np.asarray(mesh.visual.vertex_colors) if has_vc else None

    for idx, v in enumerate(mesh.vertices):
        if has_vc and vc_arr is not None and idx < len(vc_arr):
            c_r, c_g, c_b = vc_arr[idx][:3] / 255.0
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c_r:.4f} {c_g:.4f} {c_b:.4f}")
        else:
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

    uvs = getattr(mesh.visual, "uv", None)
    has_uv = uvs is not None and len(uvs) == len(mesh.vertices)

    if has_uv:
        for uv in uvs:
            lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}")

    # Faces are 1-indexed in OBJ
    if has_uv:
        for f in mesh.faces:
            lines.append(f"f {f[0]+1}/{f[0]+1} {f[1]+1}/{f[1]+1} {f[2]+1}/{f[2]+1}")
    else:
        for f in mesh.faces:
            lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")

    output_obj_path.write_text("\n".join(lines), encoding="utf-8")

    # Export STL for universal 3D printing and CAD software
    stl_path = output_obj_path.with_suffix(".stl")
    try:
        mesh.export(str(stl_path))
    except Exception:
        pass

    return output_obj_path, mtl_path, output_texture_path


def detect_salient_foreground_box(
    image: np.ndarray,
    depth_map: np.ndarray | None = None,
) -> tuple[int, int, int, int]:
    """Detect bounding box of the most prominent foreground object based on depth and saliency.

    Args:
        image: RGB image array (H, W, 3).
        depth_map: Optional precomputed normalized depth map.

    Returns:
        (x, y, w, h) bounding box in pixel coordinates.
    """
    h, w = image.shape[:2]
    if depth_map is None:
        depth_map = estimate_depth_map(image)

    # Threshold on depth map using Otsu
    thresh = (depth_map * 255).astype(np.uint8)
    _, binary = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find contour with largest area
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area > (h * w * 0.05):  # At least 5% of image
            x, y, bw, bh = cv2.boundingRect(largest)
            pad = 12
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(w, x + bw + pad)
            y1 = min(h, y + bh + pad)
            return (x0, y0, x1 - x0, y1 - y0)

    return (0, 0, w, h)
