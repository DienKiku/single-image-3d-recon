"""Reusable Streamlit UI component functions."""

import streamlit as st
import numpy as np
from typing import Any
from dataclasses import dataclass

@dataclass
class SidebarState:
    """State returned from sidebar controls."""
    uploaded_file: Any
    analyze_clicked: bool
    generate_3d_clicked: bool
    export_clicked: bool
    explosion_factor: float = 0.0
    reconstruction_mode: str = "Unified 3D Model"
    ai_segmenter: str = "Intelligent Layered / Saliency"
    ai_3d_engine: str = "sf3d"
    reference_type: str = "Tự động (Desktop Standard ~380mm W)"
    manual_dimension_mm: float | None = None
    ruler_p1: tuple[int, int] | None = None
    ruler_p2: tuple[int, int] | None = None
    target_roi: tuple[int, int, int, int] | None = None
    foreground_ratio: float = 0.85
    texture_resolution: int = 1024
    remesh_option: str = "none"
    target_vertex_count: int = 15000


def render_sidebar_controls() -> SidebarState:
    """Render modern streamlined sidebar controls exclusively powered by Stable Fast 3D.
    
    Returns:
        SidebarState: The current values of all sidebar inputs and buttons.
    """
    import os

    with st.sidebar:
        st.title("🧊 Single-Image to 3D")
        st.markdown("**Reconstruction Studio** — Tái tạo mô hình 3D nguyên khối 360° & bóc tách chi tiết từ một bức ảnh.")
        st.divider()
        
        st.subheader("📁 1. Tải ảnh lên (Input)")
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
        
        target_roi = None
        if uploaded_file is not None:
            roi_mode = st.radio(
                "Vùng nhận diện",
                ["Tự động bóc tách vật thể (Auto-Isolate)", "Toàn bộ khung hình (Full Scene)"],
                index=0,
            )
            if roi_mode == "Toàn bộ khung hình (Full Scene)":
                target_roi = "full"

        st.divider()
        st.subheader("📐 2. Chế độ dựng hình (3D Mode)")
        rec_mode_ui = st.radio(
            "Cấu trúc mô hình 3D",
            ["✨ Toàn bộ mô hình (Unified 3D Model)", "💥 Phân tầng bóc tách (Multi-Layer Exploded)"],
            index=0,
            label_visibility="collapsed"
        )
        rec_mode = "unified" if "Unified" in rec_mode_ui else "layers"

        st.divider()
        st.subheader("🤖 3. Engine Dựng Hình 3D")
        st.info("🚀 **Stable Fast 3D (SF3D)**\n\n*Stability AI Reference Architecture*", icon="✨")
        gen_backend = "sf3d"
        
        with st.expander("⚙️ Cấu hình Stable Fast 3D", expanded=True):
            foreground_ratio = st.slider(
                "Tỷ lệ vật thể (Foreground Ratio)",
                min_value=0.50,
                max_value=0.98,
                value=0.85,
                step=0.01,
                help="Tỷ lệ khung hình vật thể sau khi chuẩn hóa vào camera SF3D."
            )
            texture_res = st.selectbox(
                "Độ phân giải Texture",
                [512, 1024, 2048, 4096],
                index=1,
                help="Kích thước bản đồ texture PBR."
            )
            remesh_opt = st.selectbox(
                "Thuật toán Remesh",
                ["none", "triangle", "quad"],
                index=0,
                help="none: giữ nguyên; triangle: Botsch & Kobbelt; quad: Field-Aligned Quads."
            )
            target_vertex_count = st.number_input(
                "Số đỉnh mục tiêu (Target Vertices)",
                min_value=1000,
                max_value=100000,
                value=15000,
                step=1000,
                help="Giới hạn số lượng đỉnh khi bật chế độ remesh."
            )

        st.caption("🔒 100% Cục bộ offline, chuẩn kiến trúc Stability AI, tối ưu VRAM <= 6GB.")

        st.divider()
        st.subheader("📏 4. Hiệu chuẩn kích thước (Scale)")
        with st.expander("📐 Kích thước vật lý thực tế", expanded=False):
            reference_type = st.selectbox(
                "Phương pháp định cỡ mm",
                [
                    "Tự động (Desktop Standard ~380mm W)",
                    "Nhập chiều rộng thủ công (Manual Width mm)",
                    "Thẻ ID / Card (85.6 x 53.98 mm)",
                    "Mẫu ArUco Marker (50mm)",
                    "Preset: Máy in / Thiết bị lớn (600 x 1150 x 650 mm)",
                    "Preset: Đồ gia dụng nhỏ / Cốc (85 x 95 x 85 mm)",
                ],
                index=0,
                help="Hiệu chuẩn kích thước thực tế tính bằng milimet (mm) cho CAD / in 3D."
            )
            manual_dimension_mm = None
            if "Manual" in reference_type or "thủ công" in reference_type:
                manual_dimension_mm = st.number_input(
                    "Chiều rộng vật thể (Width mm)",
                    min_value=10.0,
                    max_value=3000.0,
                    value=380.0,
                    step=10.0,
                )

        st.divider()
        st.subheader("⚡ 5. Thao tác thực hiện")
        has_file = uploaded_file is not None
        
        analyze_clicked = st.button("🔍 1. Phân tích & Tách nền", disabled=not has_file, width="stretch")
        generate_3d_clicked = st.button("🧊 2. Tái tạo mô hình 3D", disabled=not has_file, width="stretch")
        export_clicked = st.button("📦 3. Xuất file 3D (OBJ + STL + ZIP)", disabled=not has_file, width="stretch")
        
        explosion_factor = 0.0
        if rec_mode == "layers":
            st.divider()
            st.subheader("💥 Độ bung phân tầng (Exploded View)")
            explosion_pct = st.slider("Khoảng cách tách tầng (%)", min_value=0, max_value=100, value=25, step=1)
            explosion_factor = explosion_pct / 100.0
        
        return SidebarState(
            uploaded_file=uploaded_file,
            analyze_clicked=analyze_clicked,
            generate_3d_clicked=generate_3d_clicked,
            export_clicked=export_clicked,
            explosion_factor=explosion_factor,
            reconstruction_mode=rec_mode,
            ai_segmenter="Intelligent Layered / Saliency",
            ai_3d_engine=gen_backend,
            reference_type=reference_type,
            manual_dimension_mm=manual_dimension_mm,
            target_roi=target_roi,
            foreground_ratio=foreground_ratio,
            texture_resolution=texture_res,
            remesh_option=remesh_opt,
            target_vertex_count=int(target_vertex_count),
        )


def render_metadata_hud(
    parts_count: int = 0,
    dimensions: tuple[float, float, float] | None = None,
    material_estimate: str = "Unknown",
) -> None:
    """Render the metadata HUD overlay using st.metric cards.
    
    Args:
        parts_count: Total number of identified parts.
        dimensions: (W, H, D) tuple in mm, or None.
        material_estimate: String estimating the material.
    """
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Parts", parts_count)
        
    with col2:
        if dimensions:
            dim_str = f"{dimensions[0]:.1f} × {dimensions[1]:.1f} × {dimensions[2]:.1f} mm"
        else:
            dim_str = "N/A"
        st.metric("Dimensions", dim_str)
        
    with col3:
        st.metric("Material", material_estimate)


def render_analysis_overlay(
    image: np.ndarray,
    masks: list | None = None,
    calibration = None,
) -> np.ndarray:
    """Draw colored segmentation masks and dimension annotations on image.
    
    Args:
        image: Original RGB image array.
        masks: List of mask objects (dicts with 'mask' key or SegmentedLayer dataclasses).
        calibration: Calibration object with pixel_to_mm factor.
        
    Returns:
        np.ndarray: Annotated RGB image array.
    """
    import cv2
    
    overlay = image.copy()
    
    if masks:
        # Predefined colors (RGB format)
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), 
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (255, 128, 0), (128, 0, 255)
        ]
        
        alpha = 0.4
        
        for i, mask_item in enumerate(masks):
            color = colors[i % len(colors)]
            # Support both dict-based and dataclass-based mask objects
            if isinstance(mask_item, dict):
                mask_arr = mask_item.get('mask', None)
            else:
                mask_arr = getattr(mask_item, 'mask', None)
            if mask_arr is not None:
                # Resize mask to image dimensions if needed
                h, w = overlay.shape[:2]
                if mask_arr.shape[:2] != (h, w):
                    mask_arr = cv2.resize(mask_arr, (w, h), interpolation=cv2.INTER_NEAREST)
                # Apply color to mask area
                for c in range(3):
                    overlay[:, :, c] = np.where(
                        mask_arr > 0, 
                        overlay[:, :, c] * (1 - alpha) + color[c] * alpha, 
                        overlay[:, :, c]
                    )
                    
    # Optional: draw dimension labels if calibration is present
    if calibration and getattr(calibration, 'pixel_to_mm', 0) > 0:
        h, w = overlay.shape[:2]
        cv2.putText(overlay, f"Scale: 1px = {calibration.pixel_to_mm:.4f}mm", 
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
    return overlay


def render_empty_state(tab_name: str = "analysis") -> None:
    """Render a styled empty state message for tabs.
    
    Args:
        tab_name: 'analysis' or '3d'
    """
    if tab_name == "analysis":
        st.info("📊 Upload an image and click 'Analyze & Segment' to view the 2D breakdown.")
    elif tab_name == "3d":
        st.info("🧊 Upload an image and click 'Generate 3D Layers' to explore the exploded 3D model.")
    else:
        st.info("No content to display.")
