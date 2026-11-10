#!/usr/bin/env python3
"""Prefetch VOIX runtime model caches for an offline-capable demo.

The serving stack downloads two external model families on first use:

* StyleTTS 2 weights, managed by ``cached_path`` (``CACHED_PATH_CACHE_ROOT``).
* AudioSeal generator/detector weights, managed by ``AUDIOSEAL_CACHE_DIR``.

Plus the NLTK tokenizer data used by StyleTTS 2's text front-end.

This script loads each component on CPU, which downloads the weights without
allocating GPU memory.  It never writes into the repository.  Use ``--check``
to inspect caches without importing or downloading anything.

Examples:
    python scripts/setup/prefetch_models.py
    python scripts/setup/prefetch_models.py --check --json
    python scripts/setup/prefetch_models.py --force --only styletts2
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from voix.utils.model_cache import (  # noqa: E402
    audioseal_cache_dir,
    cache_file_count,
    cache_has_files,
    styletts2_cache_dir,
)

CACHE_CHOICES = ("styletts2", "audioseal", "nltk")


def _nltk_status() -> Dict[str, Any]:
    """Return availability of the NLTK tokenizer packages StyleTTS2 may use."""
    try:
        import nltk
    except ImportError:
        return {
            "cached": False,
            "packages": {},
            "error": "nltk is not installed; install the serving extra",
        }

    packages: Dict[str, bool] = {}
    for package in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{package}")
            packages[package] = True
        except LookupError:
            packages[package] = False
    # ``punkt`` is the package the validated NLTK front-end uses.  ``punkt_tab``
    # is required by newer NLTK releases, so it is prefetched but not made a
    # hard failure when the installed NLTK predates it.
    return {"cached": packages.get("punkt", False), "packages": packages}


def _cache_report(only: List[str]) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    if "styletts2" in only:
        path = styletts2_cache_dir()
        report["styletts2"] = {
            "path": str(path),
            "cached": cache_has_files(path, minimum=6),
            "files": cache_file_count(path),
            "minimum_files": 6,
        }
    if "audioseal" in only:
        path = audioseal_cache_dir()
        report["audioseal"] = {
            "path": str(path),
            "cached": cache_has_files(path, minimum=2),
            "files": cache_file_count(path),
            "minimum_files": 2,
        }
    if "nltk" in only:
        status = _nltk_status()
        status["path"] = "nltk_data/tokenizers"
        report["nltk"] = status
    return report


def _prefetch_styletts2(force: bool) -> Dict[str, Any]:
    path = styletts2_cache_dir()
    if not force and cache_has_files(path, minimum=6):
        return {"status": "skipped", "reason": "cache already populated", "path": str(path)}

    try:
        import torch
        from styletts2 import tts
    except ImportError as exc:
        raise RuntimeError(
            "StyleTTS2 is not installed; install the serving extra with "
            "`pip install -e '.[serving]'`"
        ) from exc

    original_is_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        with contextlib.redirect_stdout(sys.stderr):
            model = tts.StyleTTS2(model_checkpoint_path=None, config_path=None)
        del model
    finally:
        torch.cuda.is_available = original_is_available

    return {
        "status": "downloaded" if force else "fetched",
        "path": str(path),
        "cached": cache_has_files(path, minimum=6),
        "files": cache_file_count(path),
    }


def _prefetch_audioseal(force: bool) -> Dict[str, Any]:
    path = audioseal_cache_dir()
    if not force and cache_has_files(path, minimum=2):
        return {"status": "skipped", "reason": "cache already populated", "path": str(path)}

    try:
        from audioseal import AudioSeal
    except ImportError as exc:
        raise RuntimeError(
            "AudioSeal is not installed; install the serving extra or pass "
            "`--no-watermark` during synthesis"
        ) from exc

    with contextlib.redirect_stdout(sys.stderr):
        generator = AudioSeal.load_generator("audioseal_wm_16bits")
        detector = AudioSeal.load_detector("audioseal_detector_16bits")
    del generator, detector

    return {
        "status": "downloaded" if force else "fetched",
        "path": str(path),
        "cached": cache_has_files(path, minimum=2),
        "files": cache_file_count(path),
    }


def _prefetch_nltk(force: bool) -> Dict[str, Any]:
    try:
        import nltk
    except ImportError as exc:
        raise RuntimeError(
            "nltk is not installed; install the serving extra with "
            "`pip install -e '.[serving]'`"
        ) from exc

    downloaded = {}
    for package in ("punkt", "punkt_tab"):
        with contextlib.redirect_stdout(sys.stderr):
            downloaded[package] = bool(
                nltk.download(package, quiet=True, force=force)
            )

    status = _nltk_status()
    return {
        "status": "downloaded" if any(downloaded.values()) else "skipped",
        "path": "nltk_data/tokenizers",
        "cached": status["cached"],
        "packages": status["packages"],
        "download_results": downloaded,
    }


def _print_human(report: Dict[str, Any], check_only: bool) -> None:
    for name, entry in report.items():
        if not isinstance(entry, dict):
            continue
        if name == "ok":
            continue
        if name == "error":
            continue
        cached = entry.get("cached")
        state = "OK" if cached else "MISSING"
        path = entry.get("path", "")
        files = entry.get("files")
        suffix = f" ({files} files)" if files is not None else ""
        print(f"[voix] {name:<10} {state:<7} {path}{suffix}")
        if entry.get("packages"):
            packages = ", ".join(
                f"{package}={'yes' if present else 'no'}"
                for package, present in entry["packages"].items()
            )
            print(f"       packages: {packages}")
        if entry.get("error"):
            print(f"       error: {entry['error']}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prefetch StyleTTS2, AudioSeal, and NLTK caches on CPU."
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=CACHE_CHOICES,
        default=None,
        help="Prefetch only this cache (repeatable). Default: all",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report cache status without importing or downloading models.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when the cache already looks populated.",
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable output.")
    args = parser.parse_args(argv)

    selected = args.only or list(CACHE_CHOICES)

    if args.check:
        report = _cache_report(selected)
        report["ok"] = all(entry.get("cached", False) for entry in report.values())
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_human(report, check_only=True)
            print("\nCaches ready." if report["ok"] else "\nSome caches are missing.")
        return 0 if report["ok"] else 1

    results: Dict[str, Any] = {}
    failures: Dict[str, str] = {}
    workers = {
        "styletts2": _prefetch_styletts2,
        "audioseal": _prefetch_audioseal,
        "nltk": _prefetch_nltk,
    }

    for name in selected:
        try:
            results[name] = workers[name](force=args.force)
        except Exception as exc:  # noqa: BLE001 - report all caches, then fail
            failures[name] = f"{type(exc).__name__}: {exc}"
            results[name] = {"status": "failed", "error": failures[name], "cached": False}

    payload = {
        "ok": not failures,
        "check_only": False,
        "results": results,
        "failures": failures,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _print_human(results, check_only=False)
        for name, message in failures.items():
            print(f"[voix] {name} failed: {message}", file=sys.stderr)
        print("\nPrefetch complete." if not failures else "\nPrefetch failed.")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
