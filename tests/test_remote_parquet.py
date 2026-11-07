"""Tests for the remote Parquet read-ahead cache.

The cache exists so one HTTP range request can serve a row group's entire
metadata span.  These tests use a fake range reader; no network is touched.
"""
from __future__ import annotations

import pytest

from voix.data.remote_zip import RemoteFile


def test_prefetch_serves_reads_and_does_not_move_position(monkeypatch):
    calls: list[tuple[int, int]] = []

    monkeypatch.setattr(RemoteFile, "_discover_size", lambda self: 100)

    def fake_read_range(self, start: int, end: int) -> bytes:
        calls.append((start, end))
        return bytes((value % 256) for value in range(start, end + 1))

    monkeypatch.setattr(RemoteFile, "_read_range", fake_read_range)

    remote = RemoteFile("https://example.invalid/file.parquet")
    try:
        assert remote.tell() == 0
        assert remote.prefetch(10, 5) == 5
        assert remote.tell() == 0
        assert calls == [(10, 14)]

        remote.seek(10)
        assert remote.read(5) == bytes(range(10, 15))
        assert calls == [(10, 14)]

        assert remote.read(3) == bytes(range(15, 18))
        assert calls[-1] == (15, 17)
    finally:
        remote.close()


def test_prefetch_empty_range_is_a_noop(monkeypatch):
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(RemoteFile, "_discover_size", lambda self: 100)
    monkeypatch.setattr(
        RemoteFile,
        "_read_range",
        lambda self, start, end: calls.append((start, end)) or b"",
    )
    remote = RemoteFile("https://example.invalid/file.parquet")
    try:
        assert remote.prefetch(10, 0) == 0
        assert calls == []
    finally:
        remote.close()
