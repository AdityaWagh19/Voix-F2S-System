"""Tests for merging N1 metric parts."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "evaluation" / "merge_n1_eval.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("merge_n1_eval", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_merge_parts_preserves_rows(tmp_path, monkeypatch):
    module = _load_module()
    first = tmp_path / "part0.json"
    second = tmp_path / "part1.json"
    first.write_text(
        json.dumps({"rows": [{"clip_id": "a", "condition": "base"}]}),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps({"rows": [{"clip_id": "a", "condition": "lora"}]}),
        encoding="utf-8",
    )
    output = tmp_path / "merged.json"
    monkeypatch.setattr(
        "sys.argv",
        ["merge", "--parts", str(first), str(second), "--output", str(output)],
    )
    assert module.main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 2


def test_merge_rejects_duplicate_rows(tmp_path, monkeypatch):
    module = _load_module()
    part = tmp_path / "part.json"
    part.write_text(
        json.dumps({"rows": [{"clip_id": "a", "condition": "base"}]}),
        encoding="utf-8",
    )
    output = tmp_path / "merged.json"
    monkeypatch.setattr(
        "sys.argv",
        ["merge", "--parts", str(part), str(part), "--output", str(output)],
    )
    try:
        module.main()
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate rows were accepted")
