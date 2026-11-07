"""Read Parquet metadata over HTTP range requests.

GLOBE_V3 ships 165.5 GB of Parquet, but the audio lives in one column and the
metadata lives in a small contiguous span at the start of every row group.  A
full download is unnecessary for indexing: read the footer, prefetch the
metadata span, and decode only those columns.  Audio is fetched later, and
only for row groups that contain selected clips.

The implementation deliberately uses the existing :class:`RemoteFile` (the
same range-request reader used by the VoxCeleb ZIP pipeline) so there is one
HTTP path to maintain.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .remote_zip import RemoteFile, hf_dataset_url

HF_API = "https://huggingface.co/api/datasets"


def list_repo_files(
    repo_id: str,
    *,
    revision: str = "main",
    token: Optional[str] = None,
) -> List[Dict[str, object]]:
    """List every file in a Hugging Face dataset repo, following pagination."""
    url = f"{HF_API}/{repo_id}/tree/{revision}?recursive=true&expand=true"
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    files: List[Dict[str, object]] = []
    cursor: Optional[str] = None
    while True:
        page_url = url
        if cursor:
            page_url += "&cursor=" + urllib.parse.quote(cursor)
        request = urllib.request.Request(page_url, headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            batch = json.load(response)
            link = response.headers.get("Link", "")
        files.extend(batch)
        cursor = None
        if 'rel="next"' in link:
            target = link.split(";")[0].strip("<>")
            query = urllib.parse.parse_qs(urllib.parse.urlparse(target).query)
            cursor = query.get("cursor", [None])[0]
        if not cursor:
            break
    return files


def list_parquet_files(
    repo_id: str,
    *,
    revision: str = "main",
    token: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Return only the Parquet entries, with ``path`` and ``size``."""
    return [
        entry
        for entry in list_repo_files(repo_id, revision=revision, token=token)
        if str(entry.get("path", "")).endswith(".parquet")
    ]


class RemoteParquet:
    """A Parquet file whose bytes are fetched lazily over HTTP ranges."""

    def __init__(
        self,
        url: str,
        *,
        token: Optional[str] = None,
        timeout_sec: float = 180.0,
    ) -> None:
        self.url = str(url)
        self.file = RemoteFile(url, token=token, timeout_sec=timeout_sec)
        try:
            self.parquet = pq.ParquetFile(self.file)
        except Exception:
            self.file.close()
            raise
        self.metadata_columns = [
            name
            for name in self.parquet.schema_arrow.names
            if name != "audio" and not name.startswith("audio.")
        ]

    @property
    def num_rows(self) -> int:
        return int(self.parquet.metadata.num_rows)

    @property
    def num_row_groups(self) -> int:
        return int(self.parquet.metadata.num_row_groups)

    def _span(
        self,
        row_group: int,
        columns: Sequence[str],
    ) -> tuple[int, int]:
        metadata = self.parquet.metadata.row_group(int(row_group))
        wanted = set(columns)
        starts: List[int] = []
        ends: List[int] = []
        for index in range(metadata.num_columns):
            column = metadata.column(index)
            name = column.path_in_schema
            root = name.split(".", 1)[0]
            if root not in wanted:
                continue
            start = column.dictionary_page_offset or column.data_page_offset
            starts.append(int(start))
            ends.append(int(start) + int(column.total_compressed_size))
        if not starts:
            raise ValueError(f"no columns {list(columns)} in row group {row_group}")
        return min(starts), max(ends)

    def prefetch_row_group(
        self,
        row_group: int,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> int:
        """Fetch the byte span for ``columns`` in one range request."""
        names = list(columns) if columns is not None else list(self.metadata_columns)
        start, end = self._span(row_group, names)
        return self.file.prefetch(start, end - start)

    def read_metadata_row_group(
        self,
        row_group: int,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> pa.Table:
        """Read only metadata columns from one row group."""
        names = list(columns) if columns is not None else list(self.metadata_columns)
        self.prefetch_row_group(row_group, columns=names)
        return self.parquet.read_row_group(int(row_group), columns=names)

    def read_audio_row_group(self, row_group: int) -> pa.Table:
        """Read the audio column for one row group (fetches the audio chunk)."""
        self.prefetch_row_group(row_group, columns=["audio"])
        return self.parquet.read_row_group(int(row_group), columns=["audio"])

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "RemoteParquet":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def open_dataset_parquet(
    repo_id: str,
    path: str,
    *,
    revision: str = "main",
    token: Optional[str] = None,
    timeout_sec: float = 180.0,
) -> RemoteParquet:
    """Open ``repo_id``/``path`` as a :class:`RemoteParquet`."""
    url = hf_dataset_url(repo_id, path)
    if revision and revision != "main":
        url = url.replace("/resolve/main/", f"/resolve/{revision}/")
    return RemoteParquet(url, token=token, timeout_sec=timeout_sec)
