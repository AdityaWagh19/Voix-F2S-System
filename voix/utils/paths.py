"""Project-root path resolution.

The codebase grew on one Windows workstation with ``D:/voix`` hard-coded in
several modules.  Those paths break on any other checkout, so all runtime
defaults should resolve relative to the repository root instead.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def project_path(*parts: Union[str, Path]) -> Path:
    """Return a path rooted at the repository, independent of the cwd."""
    return PROJECT_ROOT.joinpath(*(str(part) for part in parts))


def resolve_path(value: Union[str, Path], *, base: Union[str, Path, None] = None) -> Path:
    """Expand ``~`` and resolve relative paths against the repository root."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    root = Path(base) if base is not None else PROJECT_ROOT
    return (root / path).resolve()
