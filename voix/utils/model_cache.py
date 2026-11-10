"""Model-cache discovery for serving preflight checks.

The serving stack has two cache roots that are independent of the repository:

* AudioSeal uses ``AUDIOSEAL_CACHE_DIR`` (or ``XDG_CACHE_HOME``), then
  appends ``audioseal/``.
* StyleTTS 2 uses ``cached_path``'s ``CACHED_PATH_CACHE_ROOT`` (or
  ``~/.cache/cached_path``).

These helpers only inspect paths.  They never import the model packages and
never trigger a download.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


def cache_has_files(directory: Path, minimum: int = 1) -> bool:
    """Return True when a cache directory contains at least ``minimum`` files."""
    try:
        return directory.is_dir() and sum(
            1 for path in directory.rglob("*") if path.is_file()
        ) >= minimum
    except OSError:
        return False


def cache_file_count(directory: Path) -> int:
    """Return the number of files below a cache directory, or 0."""
    try:
        if not directory.is_dir():
            return 0
        return sum(1 for path in directory.rglob("*") if path.is_file())
    except OSError:
        return 0


def audioseal_cache_dir() -> Path:
    """Resolve the AudioSeal cache directory the same way its loader does."""
    root = os.environ.get("AUDIOSEAL_CACHE_DIR")
    if root:
        return Path(root).expanduser() / "audioseal"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "audioseal"
    return Path.home() / ".cache" / "audioseal"


def styletts2_cache_dir() -> Path:
    """Resolve the StyleTTS 2 ``cached_path`` directory."""
    root = os.environ.get("CACHED_PATH_CACHE_ROOT")
    if root:
        return Path(root).expanduser()
    return Path.home() / ".cache" / "cached_path"


def cache_status() -> Dict[str, Dict[str, object]]:
    """Return a JSON-serializable status for both serving caches."""
    status: Dict[str, Dict[str, object]] = {}
    for name, path in (
        ("styletts2", styletts2_cache_dir()),
        ("audioseal", audioseal_cache_dir()),
    ):
        status[name] = {
            "path": str(path),
            "exists": path.is_dir(),
            "files": cache_file_count(path),
        }
    return status
