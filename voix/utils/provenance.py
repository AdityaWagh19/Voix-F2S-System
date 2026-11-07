"""Structured provenance for learned artifacts and runtime backends.

Every reported metric should be traceable to the exact checkpoints, scaler, and
backend that produced it.  Hashing is streaming and bounded-memory.
"""
from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

from voix.utils.paths import PROJECT_ROOT


def sha256_file(path: Union[str, Path], chunk_size: int = 1 << 20) -> Optional[str]:
    """Return the SHA-256 of a file, or ``None`` when it does not exist."""
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(
    path: Union[str, Path],
    role: str,
) -> Dict[str, Any]:
    """Describe one artifact with a content hash and size."""
    path = Path(path)
    exists = path.is_file()
    return {
        "role": role,
        "path": str(path),
        "exists": exists,
        "bytes": path.stat().st_size if exists else None,
        "sha256": sha256_file(path) if exists else None,
    }


def runtime_provenance(
    *,
    device: Optional[str] = None,
    backend: Optional[str] = None,
    schema_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Describe the interpreter, hardware, backend, and representation schema."""
    record: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": device,
        "backend": backend,
        "schema_version": schema_version,
        "torch": None,
        "cuda_available": None,
        "cuda_version": None,
        "gpu": None,
    }
    try:
        import torch

        record["torch"] = torch.__version__
        record["cuda_available"] = bool(torch.cuda.is_available())
        record["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            record["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return record


def checkpoint_provenance(
    paths: Dict[str, Union[str, Path]],
    *,
    device: Optional[str] = None,
    backend: Optional[str] = None,
    schema_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the full provenance block stored with synthesis results."""
    return {
        "project_root": str(PROJECT_ROOT),
        "runtime": runtime_provenance(
            device=device, backend=backend, schema_version=schema_version
        ),
        "artifacts": {
            role: artifact_record(path, role) for role, path in paths.items()
        },
    }
