"""Read ZIP archives over HTTP range requests.

The VoxCeleb2 dev shards are ~50 GB each.  Downloading a whole shard just to
inspect its central directory is wasteful, and extracting every clip is
unnecessary when the manifest only needs a balanced subset.  This module lets
the index builder read the ZIP directory and pull selected members directly.

Only the standard library ``zipfile`` is used on top of a small seekable HTTP
file object, so the same code works against the Hugging Face CDN or any server
that supports ``Range`` requests.
"""
from __future__ import annotations

import io
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Optional

import requests


class RemoteZipError(RuntimeError):
    """Raised when a remote ZIP cannot be read."""


class RemoteFile(io.RawIOBase):
    """A minimal seekable read-only file backed by HTTP range requests."""

    def __init__(
        self,
        url: str,
        *,
        token: Optional[str] = None,
        timeout_sec: float = 120.0,
        max_retries: int = 4,
    ) -> None:
        super().__init__()
        self.url = str(url)
        self.timeout_sec = float(timeout_sec)
        self.max_retries = max(1, int(max_retries))
        self.session = requests.Session()
        self.headers = {"Accept-Encoding": "identity"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        self.pos = 0
        self.size = self._discover_size()
        self._chunk_size = 8 * 1024 * 1024
        # Single-window read-ahead.  Parquet row groups keep all metadata
        # columns in one small contiguous span before the audio chunk, so one
        # prefetch serves every metadata column read for that row group.
        self._cache_start = 0
        self._cache_end = 0
        self._cache = b""

    def _request(self, headers: dict) -> requests.Response:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    self.url,
                    headers={**self.headers, **headers},
                    stream=True,
                    timeout=self.timeout_sec,
                )
                if response.status_code in (200, 206):
                    return response
                if response.status_code in (408, 429, 500, 502, 503, 504):
                    last_error = RemoteZipError(
                        f"HTTP {response.status_code} for {self.url}"
                    )
                    response.close()
                    time.sleep(min(8.0, 0.5 * (2**attempt)))
                    continue
                text = response.text[:200]
                response.close()
                raise RemoteZipError(
                    f"HTTP {response.status_code} for {self.url}: {text}"
                )
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(min(8.0, 0.5 * (2**attempt)))
        raise RemoteZipError(f"request failed after {self.max_retries} tries: {last_error}")

    def _discover_size(self) -> int:
        response = self._request({"Range": "bytes=0-0"})
        try:
            content_range = response.headers.get("Content-Range", "")
            if "/" not in content_range:
                raise RemoteZipError(
                    f"Server did not return a Content-Range header for {self.url}"
                )
            return int(content_range.rsplit("/", 1)[1])
        finally:
            response.close()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return int(self.pos)

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            new_pos = int(offset)
        elif whence == os.SEEK_CUR:
            new_pos = int(self.pos + offset)
        elif whence == os.SEEK_END:
            new_pos = int(self.size + offset)
        else:
            raise ValueError(f"invalid whence: {whence}")
        self.pos = max(0, min(new_pos, self.size))
        return self.pos

    def _read_range(self, start: int, end: int) -> bytes:
        response = self._request({"Range": f"bytes={start}-{end}"})
        try:
            if response.status_code != 206:
                raise RemoteZipError(
                    f"Expected HTTP 206 for a range request, got {response.status_code}"
                )
            return response.content
        finally:
            response.close()

    def prefetch(self, offset: int, size: int) -> int:
        """Fetch one byte range and serve subsequent reads from it.

        This does not move the file position.  It is intended for callers that
        know they will read a small contiguous region, such as a Parquet row
        group's metadata columns.
        """
        start = max(0, min(int(offset), self.size))
        if size <= 0 or start >= self.size:
            self._cache_start = self._cache_end = start
            self._cache = b""
            return 0
        end = min(start + int(size), self.size) - 1
        data = self._read_range(start, end)
        self._cache = data
        self._cache_start = start
        self._cache_end = start + len(data)
        return len(data)

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self.pos
        size = int(size)
        if size <= 0 or self.pos >= self.size:
            return b""
        if self._cache_start <= self.pos and self.pos + size <= self._cache_end:
            offset = self.pos - self._cache_start
            data = self._cache[offset : offset + size]
            self.pos += len(data)
            return data
        end = min(self.pos + size, self.size) - 1
        data = self._read_range(self.pos, end)
        if not data:
            return b""
        self.pos += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def close(self) -> None:
        try:
            self.session.close()
        finally:
            super().close()


class RemoteZip:
    """A ``zipfile.ZipFile`` wrapper over a remote HTTP ZIP."""

    def __init__(
        self,
        url: str,
        *,
        token: Optional[str] = None,
        timeout_sec: float = 120.0,
    ) -> None:
        self.url = str(url)
        self._file = RemoteFile(url, token=token, timeout_sec=timeout_sec)
        try:
            self._zip = zipfile.ZipFile(self._file)
        except Exception:
            self._file.close()
            raise
        self._names: Optional[list[str]] = None

    @property
    def size(self) -> int:
        return int(self._file.size)

    def namelist(self) -> list[str]:
        if self._names is None:
            self._names = self._zip.namelist()
        return list(self._names)

    def infolist(self) -> list[zipfile.ZipInfo]:
        return self._zip.infolist()

    def open(self, name: str):
        return self._zip.open(name)

    def read(self, name: str) -> bytes:
        with self._zip.open(name) as handle:
            return handle.read()

    def download(self, name: str, destination: str | Path) -> Path:
        """Extract one member to ``destination``, creating parent directories."""
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._zip.open(name) as source, open(destination, "wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        return destination

    def close(self) -> None:
        try:
            self._zip.close()
        finally:
            self._file.close()

    def __enter__(self) -> "RemoteZip":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def hf_dataset_url(repo_id: str, filename: str) -> str:
    """Return the Hugging Face resolve URL for a dataset file."""
    from huggingface_hub import hf_hub_url

    return hf_hub_url(repo_id=repo_id, filename=filename, repo_type="dataset")
