import cv2
import numpy as np
import trimesh


def create_relief_mesh(depth_map: np.ndarray, color_image: np.ndarray,
                       depth_scale: float = 0.35, smooth: int = 3,
                       solid: bool = False, base_height: float = 0.05) -> trimesh.Trimesh:
    """Tạo mô hình lưới 3D dạng phù điêu (2.5D relief) từ depth map và ảnh RGB."""
    h, w = depth_map.shape[:2]

    # Làm mượt depth map trước nếu cần
    if smooth > 0:
        k = smooth * 2 + 1
        depth_map = cv2.GaussianBlur(depth_map, (k, k), 0)

    # Đưa màu về đúng kích thước
    if color_image.shape[:2] != (h, w):
        color_image = cv2.resize(color_image, (w, h), interpolation=cv2.INTER_LINEAR)

    # Toạ độ lưới (chuẩn hóa tỷ lệ theo chiều rộng/cao)
    aspect = w / float(h)
    x = np.linspace(-aspect / 2.0, aspect / 2.0, w, dtype=np.float32)
    y = np.linspace(0.5, -0.5, h, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    zv = depth_map * depth_scale

    # Đỉnh bề mặt trên
    top_verts = np.column_stack([xv.ravel(), yv.ravel(), zv.ravel()])
    top_colors = color_image.reshape(-1, 3)
    # Thêm kênh alpha
    top_colors = np.column_stack([top_colors, np.full((len(top_colors), 1), 255, dtype=np.uint8)])

    # Tạo các mặt (tam giác) bề mặt trên
    r_idx, c_idx = np.mgrid[0:h - 1, 0:w - 1]
    v0 = (r_idx * w + c_idx).ravel()
    v1 = (r_idx * w + (c_idx + 1)).ravel()
    v2 = ((r_idx + 1) * w + (c_idx + 1)).ravel()
    v3 = ((r_idx + 1) * w + c_idx).ravel()

    top_faces = np.empty((len(v0) * 2, 3), dtype=np.int64)
    top_faces[0::2] = np.column_stack([v0, v1, v2])
    top_faces[1::2] = np.column_stack([v0, v2, v3])

    if not solid:
        mesh = trimesh.Trimesh(vertices=top_verts, faces=top_faces, vertex_colors=top_colors, process=False)
        mesh.fix_normals()
        return mesh

    # Chế độ Solid: Thêm đáy phẳng và 4 mặt bên để tạo khối kín watertight
    z_bottom = -base_height
    bot_verts = np.column_stack([xv.ravel(), yv.ravel(), np.full_like(zv.ravel(), z_bottom)])
    bot_colors = np.full_like(top_colors, 210)

    n_top = len(top_verts)
    bot_faces = np.empty_like(top_faces)
    bot_faces[0::2] = np.column_stack([v0 + n_top, v2 + n_top, v1 + n_top])
    bot_faces[1::2] = np.column_stack([v0 + n_top, v3 + n_top, v2 + n_top])

    # 4 cạnh viền
    side_faces = []
    # Cạnh trên (r = 0):
    for c in range(w - 1):
        t1, t2 = c, c + 1
        b1, b2 = t1 + n_top, t2 + n_top
        side_faces.append([t1, b1, t2])
        side_faces.append([t2, b1, b2])

    # Cạnh dưới (r = h - 1):
    r_last = (h - 1) * w
    for c in range(w - 1):
        t1, t2 = r_last + c, r_last + c + 1
        b1, b2 = t1 + n_top, t2 + n_top
        side_faces.append([t1, t2, b1])
        side_faces.append([t2, b2, b1])

    # Cạnh trái (c = 0):
    for r in range(h - 1):
        t1, t2 = r * w, (r + 1) * w
        b1, b2 = t1 + n_top, t2 + n_top
        side_faces.append([t1, t2, b1])
        side_faces.append([t2, b2, b1])

    # Cạnh phải (c = w - 1):
    for r in range(h - 1):
        t1, t2 = r * w + (w - 1), (r + 1) * w + (w - 1)
        b1, b2 = t1 + n_top, t2 + n_top
        side_faces.append([t1, b1, t2])
        side_faces.append([t2, b1, b2])

    all_verts = np.vstack([top_verts, bot_verts])
    all_colors = np.vstack([top_colors, bot_colors])
    all_faces = np.vstack([top_faces, bot_faces, np.array(side_faces, dtype=np.int64)])

    mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, vertex_colors=all_colors, process=False)
    mesh.fix_normals()
    return mesh
