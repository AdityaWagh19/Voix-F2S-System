"""Select the best available ONNX Runtime installation.

The project venv can contain CPU-only ``onnxruntime`` while the user site has
``onnxruntime-gpu``.  Python normally resolves the venv package first, which
silently disables CUDA for InsightFace.  This helper temporarily puts the user
site ahead of the venv just long enough to import the GPU build, then removes
the path entry so other packages keep resolving from the venv.
"""
from __future__ import annotations

import site
import sys
from pathlib import Path


def _user_site_candidates() -> list[Path]:
    candidates: list[Path] = []
    try:
        user_site = site.getusersitepackages()
        if isinstance(user_site, str):
            candidates.append(Path(user_site))
        else:
            candidates.extend(Path(p) for p in user_site)
    except Exception:
        pass

    # Common Windows location when site.getusersitepackages() is unavailable.
    home = Path.home()
    candidates.append(home / "AppData" / "Roaming" / "Python" / "Python312" / "site-packages")
    candidates.append(home / ".local" / "lib" / "python3.12" / "site-packages")
    return candidates


def prefer_gpu_onnxruntime() -> list[str]:
    """Import ONNX Runtime, preferring a CUDA-capable user-site install.

    Returns the available provider names.  If ONNX Runtime was already imported
    before this call, its provider set is returned unchanged.
    """
    if "onnxruntime" in sys.modules:
        import onnxruntime as ort

        return list(ort.get_available_providers())

    for candidate in _user_site_candidates():
        if not (candidate / "onnxruntime").exists():
            continue
        # The user site is often already on sys.path *after* the venv, so
        # ``str(candidate) in sys.path`` is not enough: we must move it to the
        # front for the duration of the import.
        sys.path.insert(0, str(candidate))
        try:
            import onnxruntime as ort

            providers = list(ort.get_available_providers())
        finally:
            try:
                sys.path.remove(str(candidate))
            except ValueError:
                pass
        if "CUDAExecutionProvider" in providers:
            return providers

    import onnxruntime as ort

    return list(ort.get_available_providers())
