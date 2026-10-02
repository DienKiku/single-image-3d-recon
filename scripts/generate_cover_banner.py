"""
Production Cover Banner Generator for single-image-3d-recon.
Directly aligns with the user's post & hero case study:
"Biến một tấm hình chụp 2D đơn giản thành mô hình bản vẽ kỹ thuật, khối thạch cao (Gypsum Solid) hoặc dạng lưới (Wireframe/Mesh)"

Generates:
- docs/images/cover.png
- docs/images/cover_showcase.png
- docs/images/cover_branded.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

FONT_REG = "C:/Windows/Fonts/segoeui.ttf"
FONT_BOLD = "C:/Windows/Fonts/segoeuib.ttf"
FONT_SEMI = "C:/Windows/Fonts/seguisb.ttf"
if not Path(FONT_SEMI).exists():
    FONT_SEMI = FONT_BOLD

def get_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

def render_3d_panel_horizontal(mesh, mode="gypsum", elev=36, azim=-58, size=(520, 380)):
    """
    Renders high-res horizontal perspective matching the photograph layout.
    """
    fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100, facecolor="#0e172a")
    ax = fig.add_subplot(1, 1, 1, projection="3d", facecolor="#0e172a")
    
    v = mesh.vertices.copy()
    v[:, 0] -= v[:, 0].mean()
    v[:, 1] -= v[:, 1].mean()
    f = mesh.faces
    
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    normals = normals / norm
    
    light = np.array([0.35, -0.60, 0.85])
    light = light / np.linalg.norm(light)
    diff = np.clip(np.dot(normals, light), 0.0, 1.0)
    
    if mode == "gypsum":
        intensity = 0.38 + 0.62 * diff
        base_clay = np.array([0.94, 0.92, 0.88])
        colors = np.clip(intensity[:, None] * base_clay, 0.0, 1.0)
        poly = Poly3DCollection(v[f], facecolors=colors, edgecolors="#cbd5e1", linewidths=0.15, alpha=1.0)
        ax.add_collection3d(poly)
        
    elif mode == "wireframe":
        decim = mesh.simplify_quadric_decimation(0.25)
        dv = decim.vertices.copy()
        dv[:, 0] -= dv[:, 0].mean()
        dv[:, 1] -= dv[:, 1].mean()
        df = decim.faces
        poly = Poly3DCollection(dv[df], facecolors="#091428", edgecolors="#38bdf8", linewidths=0.70, alpha=0.96)
        ax.add_collection3d(poly)
        
    elif mode == "technical":
        vc = mesh.visual.vertex_colors[:, :3] / 255.0
        face_vc = (vc[f[:, 0]] + vc[f[:, 1]] + vc[f[:, 2]]) / 3.0
        intensity = 0.45 + 0.55 * diff
        colors = np.clip(face_vc * intensity[:, None], 0.0, 1.0)
        poly = Poly3DCollection(v[f], facecolors=colors, edgecolors="none", alpha=1.0)
        ax.add_collection3d(poly)
        
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(-36, 36)
    ax.set_ylim(-20, 20)
    ax.set_zlim(-15, 12)
    ax.axis("off")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    plt.subplots_adjust(left=-0.05, right=1.05, bottom=-0.05, top=1.05)
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, facecolor="#0e172a")
    plt.close()
    buf.seek(0)
    return Image.open(buf).convert("RGBA")

def draw_vector_arrow(draw, start_x, start_y, end_x, end_y, color=(56, 189, 248, 220), width=2):
    draw.line([(start_x, start_y), (end_x, end_y)], fill=color, width=width)
    head_size = 5
    draw.polygon([
        (end_x, end_y),
        (end_x - head_size, end_y - head_size),
        (end_x - head_size + 2, end_y),
        (end_x - head_size, end_y + head_size)
    ], fill=color)

def generate_production_cover():
    W, H = 1280, 640
    
    # 1. Base Canvas & Gradient
    canvas = Image.new("RGBA", (W, H), (8, 12, 20, 255))
    draw = ImageDraw.Draw(canvas)
    
    for y in range(H):
        t = y / H
        r = int(7 * (1 - t) + 14 * t)
        g = int(10 * (1 - t) + 20 * t)
        b = int(18 * (1 - t) + 38 * t)
        draw.line([(0, y), (W, y)], fill=(r, g, b, 255))
        
    # Top Subtle Glow
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.ellipse([W//2 - 450, -180, W//2 + 450, 220], fill=(56, 189, 248, 22))
    glow = glow.filter(ImageFilter.GaussianBlur(60))
    canvas = Image.alpha_composite(canvas, glow)
    draw = ImageDraw.Draw(canvas)
    
    # 2. Header Section
    f_badge = get_font(FONT_BOLD, 11)
    f_title = get_font(FONT_BOLD, 30)
    f_sub = get_font(FONT_REG, 14)
    
    # Top Pill Badge
    badge_text = "AI 3D RECONSTRUCTION ENGINE • 100% LOCAL & OFFLINE"
    bbox = draw.textbbox((0, 0), badge_text, font=f_badge)
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    bx = (W - bw) // 2
    by = 22
    pad_x, pad_y = 16, 5
    draw.rounded_rectangle(
        [bx - pad_x, by - pad_y, bx + bw + pad_x, by + bh + pad_y],
        radius=14,
        fill=(24, 34, 53, 220),
        outline=(56, 189, 248, 180),
        width=1
    )
    dot_x = bx - pad_x + 9
    dot_y = by + bh // 2
    draw.ellipse([dot_x - 3, dot_y - 3, dot_x + 3, dot_y + 3], fill=(56, 189, 248, 255))
    draw.text((bx + 3, by), badge_text, font=f_badge, fill=(56, 189, 248, 255))
    
    # Main Title
    title_text = "Single-Image to 3D CAD Reconstruction"
    t_bbox = draw.textbbox((0, 0), title_text, font=f_title)
    tw = t_bbox[2] - t_bbox[0]
    draw.text(((W - tw) // 2, 48), title_text, font=f_title, fill=(255, 255, 255, 255))
    
    # Subtitle
    sub_text = "Biến ảnh chụp 2D đơn giản thành Khối Thạch Cao (Gypsum) • Dạng Lưới (Wireframe) • Bản Vẽ Kỹ Thuật (Textured CAD)"
    s_bbox = draw.textbbox((0, 0), sub_text, font=f_sub)
    sw = s_bbox[2] - s_bbox[0]
    draw.text(((W - sw) // 2, 90), sub_text, font=f_sub, fill=(148, 163, 184, 255))
    
    # 3. Four Cards Geometry
    card_w = 282
    card_h = 412
    card_top = 126
    start_x = 36
    gap = 24
    
    card_defs = [
        {
            "num": "1",
            "title_vi": "ẢNH CHỤP 2D GỐC",
            "title_en": "Input 2D Photograph",
            "accent": (56, 189, 248),
            "tag": "Raw Single-View",
            "footer_1": "Ảnh chụp thực tế đơn giản",
            "footer_2": "Kích thước: 76 × 41 mm",
            "type": "photo"
        },
        {
            "num": "2",
            "title_vi": "KHỐI THẠCH CAO",
            "title_en": "Gypsum Solid / Clay",
            "accent": (251, 191, 36),
            "tag": "Solid Clay CAD",
            "footer_1": "Mặt khối thạch cao mịn",
            "footer_2": "100% Watertight Solid",
            "type": "gypsum"
        },
        {
            "num": "3",
            "title_vi": "DẠNG LƯỚI TOPO",
            "title_en": "Wireframe / Mesh",
            "accent": (45, 212, 191),
            "tag": "Decimated Mesh",
            "footer_1": "Lưới đa giác kỹ thuật",
            "footer_2": "14,908 Faces Topology",
            "type": "wireframe"
        },
        {
            "num": "4",
            "title_vi": "BẢN VẼ KỸ THUẬT",
            "title_en": "Textured PBR CAD",
            "accent": (52, 211, 153),
            "tag": "Realistic PBR Shading",
            "footer_1": "Vật liệu kim loại & PBR",
            "footer_2": "Xuất chuẩn CAD (GLB/STL)",
            "type": "technical"
        }
    ]
    
    mesh_path = Path("output/e2e_final_battery.obj")
    mesh = trimesh.load(mesh_path, force="mesh")
    
    photo_path = Path("assets/samples/hero_battery_holder.png")
    raw_photo = Image.open(photo_path).convert("RGBA")
    
    f_card_num = get_font(FONT_BOLD, 12)
    f_card_vi = get_font(FONT_BOLD, 13)
    f_card_en = get_font(FONT_REG, 11)
    f_card_foot_b = get_font(FONT_BOLD, 10)
    f_card_foot_r = get_font(FONT_REG, 10)
    f_tag = get_font(FONT_BOLD, 10)
    
    renders = {
        "gypsum": render_3d_panel_horizontal(mesh, mode="gypsum", size=(460, 330)),
        "wireframe": render_3d_panel_horizontal(mesh, mode="wireframe", size=(460, 330)),
        "technical": render_3d_panel_horizontal(mesh, mode="technical", size=(460, 330))
    }
    
    for i, c in enumerate(card_defs):
        cx = start_x + i * (card_w + gap)
        cy = card_top
        
        # Outer Card Container
        draw.rounded_rectangle(
            [cx, cy, cx + card_w, cy + card_h],
            radius=12,
            fill=(14, 21, 36, 240),
            outline=(c["accent"][0], c["accent"][1], c["accent"][2], 130),
            width=1
        )
        
        # Inner Header Bar
        header_h = 56
        draw.rounded_rectangle(
            [cx + 1, cy + 1, cx + card_w - 1, cy + header_h],
            radius=11,
            fill=(19, 28, 48, 220)
        )
        
        # Number Badge
        p_w, p_h = 24, 24
        draw.rounded_rectangle(
            [cx + 12, cy + 16, cx + 12 + p_w, cy + 16 + p_h],
            radius=6,
            fill=(c["accent"][0], c["accent"][1], c["accent"][2], 35),
            outline=c["accent"],
            width=1
        )
        draw.text((cx + 19, cy + 19), c["num"], font=f_card_num, fill=c["accent"])
        
        # Card Header Titles
        draw.text((cx + 44, cy + 13), c["title_vi"], font=f_card_vi, fill=(241, 245, 249, 255))
        draw.text((cx + 44, cy + 33), c["title_en"], font=f_card_en, fill=(148, 163, 184, 255))
        
        # Inner Image Display Area
        img_x = cx + 8
        img_y = cy + header_h + 8
        img_w = card_w - 16
        img_h = 278
        
        draw.rounded_rectangle(
            [img_x, img_y, img_x + img_w, img_y + img_h],
            radius=8,
            fill=(12, 19, 34, 255),
            outline=(30, 41, 59, 140),
            width=1
        )
        
        if c["type"] == "photo":
            p_img = raw_photo.copy()
            pw, ph = p_img.size
            crop_box = (15, 8, pw - 15, ph - 8)
            p_cropped = p_img.crop(crop_box)
            p_cropped.thumbnail((img_w - 14, img_h - 14), Image.Resampling.LANCZOS)
            
            px_offset = img_x + (img_w - p_cropped.width) // 2
            py_offset = img_y + (img_h - p_cropped.height) // 2
            
            draw.rounded_rectangle(
                [px_offset - 2, py_offset - 2, px_offset + p_cropped.width + 2, py_offset + p_cropped.height + 2],
                radius=6,
                fill=(250, 250, 252, 245)
            )
            canvas.alpha_composite(p_cropped, (px_offset, py_offset))
        else:
            r_img = renders[c["type"]].copy()
            r_img.thumbnail((img_w - 6, img_h - 6), Image.Resampling.LANCZOS)
            rx_offset = img_x + (img_w - r_img.width) // 2
            ry_offset = img_y + (img_h - r_img.height) // 2
            canvas.alpha_composite(r_img, (rx_offset, ry_offset))
            
        draw = ImageDraw.Draw(canvas)
        
        # Floating Tag
        t_tag = c["tag"]
        tb = draw.textbbox((0, 0), t_tag, font=f_tag)
        tw = tb[2] - tb[0]
        draw.rounded_rectangle(
            [img_x + 8, img_y + 8, img_x + 18 + tw, img_y + 25],
            radius=4,
            fill=(15, 23, 42, 230),
            outline=(c["accent"][0], c["accent"][1], c["accent"][2], 180),
            width=1
        )
        draw.text((img_x + 13, img_y + 10), t_tag, font=f_tag, fill=c["accent"])
        
        # Footer Descriptions
        foot_y = img_y + img_h + 10
        dot_r = 3
        draw.ellipse([cx + 14 - dot_r, foot_y + 6 - dot_r, cx + 14 + dot_r, foot_y + 6 + dot_r], fill=c["accent"])
        draw.text((cx + 22, foot_y), c["footer_1"], font=f_card_foot_b, fill=(226, 232, 240, 255))
        
        draw.ellipse([cx + 14 - dot_r, foot_y + 22 - dot_r, cx + 14 + dot_r, foot_y + 22 + dot_r], fill=c["accent"])
        draw.text((cx + 22, foot_y + 16), c["footer_2"], font=f_card_foot_r, fill=(148, 163, 184, 255))
        
        # Vector arrow
        if i < len(card_defs) - 1:
            ax1 = cx + card_w + 3
            ax2 = cx + card_w + gap - 3
            ay = cy + card_h // 2
            draw_vector_arrow(draw, ax1, ay, ax2, ay, color=(56, 189, 248, 190), width=2)
            
    # 4. Bottom Footer Info Bar
    f_meta = get_font(FONT_REG, 11)
    f_repo = get_font(FONT_BOLD, 12)
    
    foot_bar_y = 570
    draw.line([(36, foot_bar_y), (W - 36, foot_bar_y)], fill=(30, 41, 59, 180), width=1)
    
    repo_text = "DienKiku/single-image-3d-recon"
    draw.text((38, foot_bar_y + 18), repo_text, font=f_repo, fill=(255, 255, 255, 255))
    meta_text = "• Python 3.10+ & PyTorch 2.4+ • 100% Offline • Dual-Hemisphere UV • Three.js WebGL"
    draw.text((38 + draw.textbbox((0, 0), repo_text, font=f_repo)[2] + 8, foot_bar_y + 19), meta_text, font=f_meta, fill=(148, 163, 184, 255))
    
    tags = [
        ("STL / GLB / OBJ", (56, 189, 248)),
        ("PBR Baked", (52, 211, 153)),
        ("Watertight Solid", (251, 191, 36))
    ]
    cur_rx = W - 38
    for t_text, t_color in reversed(tags):
        tb = draw.textbbox((0, 0), t_text, font=f_meta)
        tw = tb[2] - tb[0]
        cur_rx -= (tw + 18)
        draw.rounded_rectangle(
            [cur_rx, foot_bar_y + 14, cur_rx + tw + 14, foot_bar_y + 34],
            radius=6,
            fill=(24, 34, 53, 180),
            outline=(t_color[0], t_color[1], t_color[2], 130),
            width=1
        )
        draw.text((cur_rx + 7, foot_bar_y + 17), t_text, font=f_meta, fill=t_color)
        cur_rx -= 10
        
    # Save to all target locations
    targets = [
        Path("docs/images/cover.png"),
        Path("docs/images/cover_showcase.png"),
        Path("docs/images/cover_branded.png"),
        Path("output/cover_new.png")
    ]
    
    for t in targets:
        t.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(t, quality=95)
        print(f"Saved banner to {t} ({t.stat().st_size} bytes)")

if __name__ == "__main__":
    generate_production_cover()
