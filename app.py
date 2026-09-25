import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path

from backend.geometry_utils import compute_exploded_positions
from backend.export_manager import export_project
from config.settings import OUTPUT_DIR
from frontend.components import (
    render_sidebar_controls,
    render_metadata_hud,
    render_empty_state,
    SidebarState
)
from frontend.threejs_viewer import (
    generate_placeholder_scene_html,
    generate_viewer_with_meshes_html
)

def render_html_content(html: str, height: int = 650):
    """Render HTML content using st.iframe (Streamlit 1.6+) or components.html fallback."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height, width="stretch")
    else:
        import streamlit.components.v1 as components
        components.html(html, height=height, scrolling=False)

st.set_page_config(layout="wide", page_title="Single-Image to 3D Reconstruction", page_icon="🧊")

# Custom CSS
st.markdown(
    """
    <style>
    div[data-testid="metric-container"] {
        background-color: #1e1e2e;
        border: 1px solid #333;
        padding: 10px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid #444;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
    }
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# Session state initialization
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "isolated_image" not in st.session_state:
    st.session_state.isolated_image = None
if "object_bbox" not in st.session_state:
    st.session_state.object_bbox = None
if "segmented_layers" not in st.session_state:
    st.session_state.segmented_layers = None
if "generated_meshes" not in st.session_state:
    st.session_state.generated_meshes = None
if "mesh_viewer_data" not in st.session_state:
    st.session_state.mesh_viewer_data = None
if "explosion_factor" not in st.session_state:
    st.session_state.explosion_factor = 0.0
if "physical_dims" not in st.session_state:
    st.session_state.physical_dims = None
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "generation_complete" not in st.session_state:
    st.session_state.generation_complete = False


def main():
    # 1. Sidebar Controls (Clean, no Reference Calibration)
    sidebar_state: SidebarState = render_sidebar_controls()
    
    # 2. Handle file upload
    if sidebar_state.uploaded_file is not None:
        if (
            st.session_state.uploaded_image is None or 
            getattr(st.session_state, "last_uploaded_name", None) != sidebar_state.uploaded_file.name
        ):
            try:
                image = Image.open(sidebar_state.uploaded_file).convert("RGB")
                st.session_state.uploaded_image = np.array(image)
                st.session_state.last_uploaded_name = sidebar_state.uploaded_file.name
                
                # Reset states for new image
                st.session_state.analysis_complete = False
                st.session_state.generation_complete = False
                st.session_state.isolated_image = None
                st.session_state.segmented_layers = None
                st.session_state.generated_meshes = None
                st.session_state.mesh_viewer_data = None
                st.session_state.physical_dims = None
            except Exception as e:
                st.error(f"Lỗi khi đọc file ảnh: {e}")
                
    st.session_state.explosion_factor = sidebar_state.explosion_factor
    
    # 3. Handle Analyze & Isolate
    if sidebar_state.analyze_clicked and st.session_state.uploaded_image is not None:
        with st.spinner("Đang phân tích hình ảnh và tách nền vật thể (U2-Net)..."):
            img = st.session_state.uploaded_image
            img_h, img_w = img.shape[:2]
            
            from backend.background_remover import remove_background
            clean_rgba, mask, bbox = remove_background(img)
            st.session_state.isolated_image = clean_rgba
            st.session_state.object_mask = mask
            st.session_state.object_bbox = bbox
            
            # 2. Compute standardized physical dimensions based on selected calibration mode
            x, y, bw, bh = bbox
            ref_type = getattr(sidebar_state, "reference_type", "Tự động")
            manual_w = getattr(sidebar_state, "manual_dimension_mm", None)

            if ("Manual" in ref_type or "thủ công" in ref_type) and manual_w and manual_w > 0:
                target_w = float(manual_w)
                px2mm = target_w / float(bw) if bw > 0 else 1.0
                target_h = bh * px2mm
                target_d = target_w * 0.70
            elif "Máy in" in ref_type:
                target_w, target_h, target_d = 600.0, 1150.0, 650.0
                px2mm = target_w / float(bw) if bw > 0 else 1.0
            elif "Cốc" in ref_type or "nhỏ" in ref_type:
                target_w, target_h, target_d = 85.0, 95.0, 85.0
                px2mm = target_w / float(bw) if bw > 0 else 1.0
            elif "Thẻ ID" in ref_type:
                try:
                    from backend.reference_detector import detect_rectangular_reference
                    cards = detect_rectangular_reference(img)
                    if cards:
                        card_w_px = cards[0].get("width_px", 0)
                        px2mm = 85.6 / float(card_w_px) if card_w_px > 0 else (380.0 / float(bw) if bw > 0 else 1.0)
                    else:
                        px2mm = 380.0 / float(bw) if bw > 0 else 1.0
                except Exception:
                    px2mm = 380.0 / float(bw) if bw > 0 else 1.0
                target_w = bw * px2mm
                target_h = bh * px2mm
                target_d = target_w * 0.70
            elif "ArUco" in ref_type:
                try:
                    from backend.reference_detector import detect_rectangular_reference
                    markers = detect_rectangular_reference(img, expected_ratio=1.0, ratio_tolerance=0.15)
                    if markers:
                        m_w_px = markers[0].get("width_px", 0)
                        px2mm = 50.0 / float(m_w_px) if m_w_px > 0 else (380.0 / float(bw) if bw > 0 else 1.0)
                    else:
                        px2mm = 380.0 / float(bw) if bw > 0 else 1.0
                except Exception:
                    px2mm = 380.0 / float(bw) if bw > 0 else 1.0
                target_w = bw * px2mm
                target_h = bh * px2mm
                target_d = target_w * 0.70
            else:
                target_w = 380.0
                px2mm = 380.0 / float(bw) if bw > 0 else 1.0
                target_h = bh * px2mm
                target_d = target_w * 0.70

            st.session_state.pixel_to_mm = px2mm
            st.session_state.physical_dims = (target_w, target_h, target_d)
            st.session_state.calibration_method = ref_type
            
            # Extract and preserve full isolated foreground object for 3D generation
            pad = 8
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(img_w, x + bw + pad)
            y1 = min(img_h, y + bh + pad)
            cropped_full = clean_rgba[y0:y1, x0:x1].copy()
            st.session_state.full_foreground_image = cropped_full

            from backend.ai_processor import SegmentedLayer
            if sidebar_state.reconstruction_mode == "layers":
                from backend.ai_processor import get_segmenter
                segmenter = get_segmenter(sidebar_state.ai_segmenter)
                layers = segmenter.segment(img, target_roi=bbox)
            else:
                # Unified Single-Mesh Mode: entire isolated object
                layers = [SegmentedLayer(
                    layer_id="full_model",
                    mask=mask,
                    cropped_image=cropped_full,
                    bbox=(x0, y0, x1 - x0, y1 - y0),
                    confidence=0.99
                )]

            st.session_state.segmented_layers = layers
            st.session_state.analysis_complete = True
            st.success(f"Đã bóc tách vật thể thành công ({len(layers)} thành phần)!")
            
    # 4. Handle Generate 3D Model
    if sidebar_state.generate_3d_clicked and st.session_state.segmented_layers:
        with st.spinner("Đang dựng mô hình 3D chất lượng cao với Stable Fast 3D (SF3D)..."):
            from backend.ai_processor import get_mesh_generator, GeneratedMesh
            from backend.depth_processor import slice_mesh_into_layers
            from backend.sf3d_pipeline import AssetExporter3D

            generator = get_mesh_generator(
                backend="sf3d",
                foreground_ratio=sidebar_state.foreground_ratio,
                texture_resolution=sidebar_state.texture_resolution,
                remesh_option=sidebar_state.remesh_option,
                target_vertex_count=sidebar_state.target_vertex_count,
            )
            img_h, img_w = st.session_state.uploaded_image.shape[:2]
            target_dim = getattr(st.session_state, "physical_dims", (380.0, 500.0, 320.0))

            meshes_dir = OUTPUT_DIR / "meshes"
            textures_dir = OUTPUT_DIR / "textures"
            meshes_dir.mkdir(parents=True, exist_ok=True)
            textures_dir.mkdir(parents=True, exist_ok=True)

            # Generate base 3D mesh from complete foreground object
            source_img = getattr(st.session_state, "full_foreground_image", None)
            if source_img is None:
                source_img = st.session_state.segmented_layers[0].cropped_image

            full_mesh_obj = generator.run_image(
                source_img,
                target_dimensions_mm=target_dim,
            )
            aligned_full = AssetExporter3D.align_to_ground(
                full_mesh_obj, target_dimensions_mm=target_dim
            )

            # Prepare diffuse texture from the full source image
            if hasattr(generator, "preprocessor"):
                processed_tex = generator.preprocessor.process(source_img)
            elif hasattr(generator, "preprocess_image"):
                processed_tex = generator.preprocess_image(source_img)
            else:
                processed_tex = Image.fromarray(source_img) if isinstance(source_img, np.ndarray) else source_img

            # Determine slicing axis: 1 for Y (vertical), 2 for Z (depth)
            chosen_axis_name = getattr(sidebar_state, "explosion_axis", "y")
            if chosen_axis_name == "z":
                slice_axis = 2
            elif chosen_axis_name == "auto":
                # Flat/thin objects (depth < 0.35 * max(width, height)) explode along Z
                w_ext, h_ext, d_ext = aligned_full.extents
                slice_axis = 2 if d_ext < 0.35 * max(w_ext, h_ext) else 1
            else:
                slice_axis = 1

            if sidebar_state.reconstruction_mode == "layers":
                # Multi-layer mode: Slice full 3D model into aligned watertight layers
                sliced_trimeshes = slice_mesh_into_layers(
                    aligned_full,
                    num_layers=3,
                    axis=slice_axis,
                    use_smart_seams=True,
                    foreground_ratio=sidebar_state.foreground_ratio,
                )
                generated_meshes = []
                for idx, sub_m in enumerate(sliced_trimeshes):
                    lid = f"layer_{idx+1:02d}"
                    obj_path = meshes_dir / f"{lid}.obj"
                    tex_path = textures_dir / f"{lid}_diffuse.png"
                    processed_tex.save(tex_path)

                    AssetExporter3D.export_all(
                        mesh=sub_m,
                        output_dir=meshes_dir,
                        base_name=lid,
                        target_dimensions_mm=None,
                        texture_path=tex_path,
                    )
                    dim_m = tuple(float(x) for x in sub_m.extents)
                    generated_meshes.append(GeneratedMesh(
                        layer_id=lid,
                        mesh_path=obj_path,
                        texture_path=tex_path,
                        vertices_count=len(sub_m.vertices),
                        faces_count=len(sub_m.faces),
                        centroid_3d=tuple(float(x) for x in sub_m.centroid),
                        dimensions_mm=dim_m,
                    ))
            else:
                # Unified 3D Model
                lid = "full_model"
                obj_path = meshes_dir / f"{lid}.obj"
                tex_path = textures_dir / f"{lid}_diffuse.png"
                processed_tex.save(tex_path)

                AssetExporter3D.export_all(
                    mesh=aligned_full,
                    output_dir=meshes_dir,
                    base_name=lid,
                    target_dimensions_mm=target_dim,
                    texture_path=tex_path,
                )
                dim_m = tuple(float(x) for x in aligned_full.extents)
                generated_meshes = [GeneratedMesh(
                    layer_id=lid,
                    mesh_path=obj_path,
                    texture_path=tex_path,
                    vertices_count=len(aligned_full.vertices),
                    faces_count=len(aligned_full.faces),
                    centroid_3d=tuple(float(x) for x in aligned_full.centroid),
                    dimensions_mm=dim_m,
                )]

            st.session_state.generated_meshes = generated_meshes

            # Build mesh viewer data for Three.js
            from backend.mesh_utils import trimesh_to_viewer_data, compute_dominant_color, load_obj_as_trimesh
            
            centroids = [np.array(m.centroid_3d) for m in generated_meshes]
            global_centroid = np.mean(centroids, axis=0) if centroids else np.zeros(3)
            
            # Axial explosion for clean mechanical assembly separation
            total_span_axis = float(aligned_full.extents[slice_axis])
            explosion_dirs = compute_exploded_positions(
                centroids,
                global_centroid,
                expansion_factor=1.0,
                axis=slice_axis,
                total_span=total_span_axis,
            ) if len(generated_meshes) > 1 else [np.zeros(3)]

            mesh_viewer_data = []
            for idx, mesh_result in enumerate(generated_meshes):
                tri_mesh = load_obj_as_trimesh(mesh_result.mesh_path)
                color = compute_dominant_color(source_img)
                
                viewer_data = trimesh_to_viewer_data(
                    mesh=tri_mesh,
                    color=color,
                    base_position=tuple(centroids[idx]),
                    explosion_direction=tuple(explosion_dirs[idx]),
                    layer_id=mesh_result.layer_id,
                    dimensions_mm=mesh_result.dimensions_mm if hasattr(mesh_result, "dimensions_mm") else target_dim,
                    texture_path=mesh_result.texture_path,
                )
                mesh_viewer_data.append(viewer_data)

            st.session_state.mesh_viewer_data = mesh_viewer_data
            st.session_state.generation_complete = True
            st.success(f"Tái tạo mô hình 3D thành công ({len(generated_meshes)} khối mesh)!")

    # 5. Handle Export Project
    if sidebar_state.export_clicked and st.session_state.generated_meshes:
        with st.spinner("Đang đóng gói file 3D (OBJ + STL + GLB + Textures)..."):
            p_dims = getattr(st.session_state, "physical_dims", (380.0, 500.0, 320.0))
            metadata = {
                "overall_dimensions_mm": {
                    "width": p_dims[0],
                    "height": p_dims[1],
                    "depth": p_dims[2],
                },
                "scale_calibration": {
                    "pixel_to_mm": getattr(st.session_state, "pixel_to_mm", 1.0),
                    "method": getattr(st.session_state, "calibration_method", "Auto"),
                },
                "total_parts": len(st.session_state.generated_meshes),
            }
            zip_path = export_project(
                generated_meshes=st.session_state.generated_meshes or [],
                segmented_layers=st.session_state.segmented_layers or [],
                metadata=metadata,
                output_dir=OUTPUT_DIR,
            )
            if zip_path.exists():
                with open(zip_path, "rb") as f:
                    zip_bytes = f.read()
                st.sidebar.download_button(
                    label="⬇️ Tải file ZIP (OBJ + STL + GLB + MTL)",
                    data=zip_bytes,
                    file_name=zip_path.name,
                    mime="application/zip",
                    width="stretch"
                )
            else:
                st.warning("Không tìm thấy file ZIP đã xuất.")

    # 6. Main Area Header & Tabs
    st.markdown(
        """
        <div style="margin-top: 5px; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 2.1rem; color: #f0f4fc; font-weight: 700; letter-spacing: -0.5px;">
                🧊 Single-Image to 3D Reconstruction
            </h1>
            <p style="margin: 4px 0 0 0; color: #8899aa; font-size: 0.95rem;">
                Tái tạo mô hình 3D nguyên khối 360° & bóc tách chi tiết từ một bức ảnh đơn lẻ (Foundation Model TripoSR)
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs(["📏 2D Analysis View", "🛠️ 3D Studio"])
    
    with tab1:
        if st.session_state.uploaded_image is not None and st.session_state.analysis_complete:
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.image(st.session_state.uploaded_image, width="stretch", caption="📸 Ảnh gốc (Original Image)")
            with col_v2:
                if getattr(st.session_state, "isolated_image", None) is not None:
                    st.image(st.session_state.isolated_image, width="stretch", caption="✨ Vật thể đã bóc tách (Isolated Object)")
                else:
                    st.image(st.session_state.uploaded_image, width="stretch", caption="Object View")
            
            p_dims = getattr(st.session_state, "physical_dims", None)
            num_parts = len(st.session_state.segmented_layers) if st.session_state.segmented_layers else 1
            render_metadata_hud(parts_count=num_parts, dimensions=p_dims, material_estimate="Plastic / Composite")
            
        elif st.session_state.uploaded_image is not None:
            st.image(st.session_state.uploaded_image, width="stretch", caption="Ảnh gốc")
            st.info("Bấm nút '🔍 1. Phân tích & Tách nền' ở thanh bên để bắt đầu xử lý.")
            
        else:
            render_empty_state("analysis")
            
    with tab2:
        if st.session_state.generation_complete and st.session_state.mesh_viewer_data:
            from frontend.threejs_viewer import generate_viewer_with_meshes_html
            html_content = generate_viewer_with_meshes_html(
                mesh_data=st.session_state.mesh_viewer_data,
                explosion_factor=st.session_state.explosion_factor,
            )
            render_html_content(html_content, height=650)
            
            # Real metadata HUD
            num_parts = len(st.session_state.generated_meshes)
            p_dims = getattr(st.session_state, "physical_dims", None)
            if getattr(st.session_state, "generated_meshes", None):
                m0 = st.session_state.generated_meshes[0]
                if hasattr(m0, "dimensions_mm") and m0.dimensions_mm[0] > 0:
                    p_dims = m0.dimensions_mm
            render_metadata_hud(parts_count=num_parts, dimensions=p_dims, material_estimate="Plastic / Metal")
        else:
            placeholder_html = generate_placeholder_scene_html()
            render_html_content(placeholder_html, height=600)
            render_empty_state("3d")


if __name__ == "__main__":
    main()
