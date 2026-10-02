import numpy as np
import trimesh


def orient(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """TripoSR xuất trong hệ toạ độ riêng -> xoay về Y-up, mặt chính quay ra +Z."""
    m = mesh.copy()
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    return m


def normalize(mesh: trimesh.Trimesh, target_size: float = 1.0,
              floor: bool = True) -> trimesh.Trimesh:
    m = mesh.copy()
    m.apply_translation(-m.bounding_box.centroid)
    scale = target_size / max(m.extents.max(), 1e-8)
    m.apply_scale(scale)
    if floor:
        m.apply_translation([0, -m.bounds[0][1], 0])
    return m


def clean(mesh: trimesh.Trimesh, min_component_ratio: float = 0.05,
          fill: bool = True) -> trimesh.Trimesh:
    """Xoá mảnh rác bay lơ lửng, hàn lỗ, chuẩn hoá pháp tuyến."""
    m = mesh.copy()
    # Tương thích trimesh phiên bản mới (4.x / 5.x) lẫn cũ (3.x)
    if hasattr(m, "update_faces"):
        try:
            if hasattr(m, "unique_faces"):
                m.update_faces(m.unique_faces())
            if hasattr(m, "nondegenerate_faces"):
                m.update_faces(m.nondegenerate_faces())
        except Exception:
            pass
    elif hasattr(m, "remove_duplicate_faces"):
        m.remove_duplicate_faces()
        if hasattr(m, "remove_degenerate_faces"):
            m.remove_degenerate_faces()

    if hasattr(m, "remove_unreferenced_vertices"):
        try:
            m.remove_unreferenced_vertices()
        except Exception:
            pass

    try:
        parts = m.split(only_watertight=False)
        if len(parts) > 1:
            biggest = max(len(p.faces) for p in parts)
            keep = [p for p in parts if len(p.faces) >= biggest * min_component_ratio]
            m = trimesh.util.concatenate(keep) if keep else m
    except Exception:
        pass

    if fill:
        try:
            m.fill_holes()
        except Exception:
            pass
    try:
        m.fix_normals()
    except Exception:
        pass
    return m


def decimate(mesh: trimesh.Trimesh, target_faces: int = 0) -> trimesh.Trimesh:
    """Giảm số mặt. 0 = giữ nguyên."""
    if target_faces <= 0 or len(mesh.faces) <= target_faces:
        return mesh

    # 1. Thử fast_simplification nếu có (nhanh, giữ vertex color)
    try:
        import fast_simplification
        v_simp, f_simp = fast_simplification.simplify(
            mesh.vertices, mesh.faces, target_count=target_faces
        )
        simp_colors = None
        if (
            mesh.visual
            and hasattr(mesh.visual, "vertex_colors")
            and mesh.visual.vertex_colors is not None
            and len(mesh.visual.vertex_colors) == len(mesh.vertices)
        ):
            from scipy.spatial import cKDTree
            tree = cKDTree(mesh.vertices)
            _, idxs = tree.query(v_simp)
            simp_colors = mesh.visual.vertex_colors[idxs]
        return trimesh.Trimesh(vertices=v_simp, faces=f_simp, vertex_colors=simp_colors, process=False)
    except Exception:
        pass

    # 2. Thử simplify_quadric_decimation của trimesh
    try:
        return mesh.simplify_quadric_decimation(target_faces)
    except Exception:
        pass

    # 3. Thử pymeshlab
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces, preserveboundary=True)
        cur = ms.current_mesh()
        new_v = cur.vertex_matrix()
        new_f = cur.face_matrix()
        simp_colors = None
        if (
            mesh.visual
            and hasattr(mesh.visual, "vertex_colors")
            and mesh.visual.vertex_colors is not None
            and len(mesh.visual.vertex_colors) == len(mesh.vertices)
        ):
            from scipy.spatial import cKDTree
            tree = cKDTree(mesh.vertices)
            _, idxs = tree.query(new_v)
            simp_colors = mesh.visual.vertex_colors[idxs]
        return trimesh.Trimesh(new_v, new_f, vertex_colors=simp_colors, process=False)
    except Exception as e:
        print(f"[WARN] Không giảm mặt được: {e}")
        return mesh


def smooth(mesh: trimesh.Trimesh, iterations: int = 0) -> trimesh.Trimesh:
    if iterations <= 0:
        return mesh
    m = mesh.copy()
    try:
        trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=-0.53, iterations=iterations)
    except Exception:
        pass
    return m


def to_uv_textured(mesh: trimesh.Trimesh, tex_size: int = 1024):
    """Chuyển vertex color -> UV + ảnh texture (cần cho một số phần mềm CAD/game)."""
    try:
        import xatlas
        from PIL import Image

        vmap, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
        v = mesh.vertices[vmap]
        colors = (mesh.visual.vertex_colors[vmap][:, :3]
                  if mesh.visual.kind == "vertex" else None)
        if colors is None:
            return mesh

        tex = np.full((tex_size, tex_size, 3), 255, np.uint8)
        px = np.clip((uvs[:, 0] * (tex_size - 1)).astype(int), 0, tex_size - 1)
        py = np.clip(((1 - uvs[:, 1]) * (tex_size - 1)).astype(int), 0, tex_size - 1)
        tex[py, px] = colors

        import cv2
        mask = np.zeros((tex_size, tex_size), np.uint8)
        mask[py, px] = 255
        tex = cv2.inpaint(tex, cv2.bitwise_not(mask), 4, cv2.INPAINT_TELEA)

        mat = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(tex), metallicFactor=0.0, roughnessFactor=0.9)
        return trimesh.Trimesh(
            vertices=v, faces=indices, process=False,
            visual=trimesh.visual.TextureVisuals(uv=uvs, material=mat))
    except Exception as e:
        print(f"[WARN] Bake UV thất bại, giữ vertex color: {e}")
        return mesh


def export(mesh: trimesh.Trimesh, out_path: str) -> str:
    mesh.export(out_path)
    return out_path
