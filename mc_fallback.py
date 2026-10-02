"""Thay MarchingCubeHelper của TripoSR bằng PyMCubes/skimage
để không phải build torchmcubes (hay lỗi trên Windows)."""
import sys
import types
import numpy as np
import torch

_PATCHED = False


def apply_patch():
    global _PATCHED
    if _PATCHED:
        return

    has_torchmcubes = False
    try:
        import torchmcubes
        if hasattr(torchmcubes, "marching_cubes") and torchmcubes.marching_cubes is not None:
            has_torchmcubes = True
    except Exception:
        pass

    if not has_torchmcubes:
        if "torchmcubes" not in sys.modules:
            dummy = types.ModuleType("torchmcubes")
            dummy.marching_cubes = None
            sys.modules["torchmcubes"] = dummy

    from tsr.models import isosurface as iso

    if has_torchmcubes:
        _PATCHED = True
        return

    def _forward(self, level: torch.FloatTensor):
        level = -level.view(self.resolution, self.resolution, self.resolution)
        vol = level.detach().cpu().numpy().astype(np.float32)

        try:
            import mcubes
            v, f = mcubes.marching_cubes(vol, 0.0)
        except Exception:
            from skimage import measure
            v, f, _, _ = measure.marching_cubes(vol, 0.0)

        if len(v) == 0:
            return torch.zeros((0, 3), dtype=torch.float32, device=level.device), torch.zeros((0, 3), dtype=torch.int64, device=level.device)

        v = torch.from_numpy(np.ascontiguousarray(v)).float()
        f = torch.from_numpy(np.ascontiguousarray(f)).long()
        v = v[..., [2, 1, 0]]
        v = v / (self.resolution - 1.0)
        return v.to(level.device), f.to(level.device)

    iso.MarchingCubeHelper.forward = _forward
    _PATCHED = True
    print("[INFO] Dùng marching cubes fallback (PyMCubes/skimage).")
