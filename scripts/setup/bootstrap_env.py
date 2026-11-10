#!/usr/bin/env python3
"""Create an isolated, reproducible VOIX environment.

The audited workstation had ``include-system-site-packages = true`` and a
polluted user site, so ``pip check`` could not be trusted.  This bootstrap
creates a normal isolated venv, installs a pinned PyTorch wheel from the
official index, then installs the project extras under the validated
constraints file.

Examples:
    python scripts/setup/bootstrap_env.py
    python scripts/setup/bootstrap_env.py --device cpu --no-lock
    python scripts/setup/bootstrap_env.py --with-evaluation --with-dev
    python scripts/setup/bootstrap_env.py --dry-run --recreate
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VENV = PROJECT_ROOT / ".venv"
DEFAULT_LOCK = PROJECT_ROOT / "constraints" / "serving-py312-cu121.txt"
TORCH_VERSION = "2.4.1"
TORCHAUDIO_VERSION = "2.4.1"
TORCH_INDEXES = {
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cpu": "https://download.pytorch.org/whl/cpu",
}


class BootstrapError(RuntimeError):
    """A setup step failed in a way the user must fix."""


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _run(
    command: List[str],
    *,
    dry_run: bool,
    check: bool = True,
) -> subprocess.CompletedProcess:
    printable = " ".join(str(part) for part in command)
    print(f"+ {printable}", flush=True)
    if dry_run:
        return subprocess.CompletedProcess(command, 0)
    env = dict(os.environ)
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    return subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=check)


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if shutil.which("nvidia-smi"):
        return "cu121"
    return "cpu"


def _check_venv_isolation(venv_dir: Path) -> None:
    config = venv_dir / "pyvenv.cfg"
    if not config.is_file():
        return
    text = config.read_text(encoding="utf-8", errors="replace").lower()
    if "include-system-site-packages = true" in text:
        raise BootstrapError(
            f"{venv_dir} includes the system site packages. Recreate it with "
            "`--recreate` so the environment is isolated."
        )


def _install_extras(args: argparse.Namespace) -> str:
    extras = {item.strip() for item in args.extras.split(",") if item.strip()}
    if args.with_evaluation:
        extras.add("evaluation")
    if args.with_data:
        extras.add("data")
    if args.with_dev:
        extras.add("dev")
    return ".[" + ",".join(sorted(extras)) + "]"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create an isolated VOIX environment with pinned serving deps."
    )
    parser.add_argument(
        "--venv",
        default=str(DEFAULT_VENV),
        help=f"Virtual environment directory (default: {DEFAULT_VENV})",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Base interpreter used to create the venv (default: current Python)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cu121", "cpu", "skip"],
        default="auto",
        help="PyTorch wheel index. 'skip' leaves PyTorch installation to you.",
    )
    parser.add_argument(
        "--extras",
        default="serving",
        help="Comma-separated project extras (default: serving)",
    )
    parser.add_argument("--with-evaluation", action="store_true", help="Add the evaluation extra")
    parser.add_argument("--with-data", action="store_true", help="Add the data-pipeline extra")
    parser.add_argument("--with-dev", action="store_true", help="Add the dev/test extra")
    parser.add_argument(
        "--lock",
        default=None,
        help="Constraints file. Default: the cu121 lock when device=cu121",
    )
    parser.add_argument("--no-lock", action="store_true", help="Do not use a constraints file")
    parser.add_argument("--recreate", action="store_true", help="Clear and rebuild the venv")
    parser.add_argument("--dry-run", action="store_true", help="Print commands, change nothing")
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Do not run `voix preflight` after installation.",
    )
    args = parser.parse_args(argv)

    device = _resolve_device(args.device)
    venv_dir = Path(args.venv).expanduser()
    if not venv_dir.is_absolute():
        venv_dir = (PROJECT_ROOT / venv_dir).resolve()
    python = _venv_python(venv_dir)

    lock_path: Optional[Path] = None
    if not args.no_lock:
        if args.lock:
            lock_path = Path(args.lock).expanduser()
            if not lock_path.is_absolute():
                lock_path = (PROJECT_ROOT / lock_path).resolve()
            if not lock_path.is_file():
                raise BootstrapError(f"Constraints file not found: {lock_path}")
        elif device == "cu121" and DEFAULT_LOCK.is_file():
            lock_path = DEFAULT_LOCK

    if lock_path and device != "cu121":
        raise BootstrapError(
            f"{lock_path.name} pins torch=={TORCH_VERSION}+cu121 and is only valid "
            "for device=cu121. Pass --no-lock for a CPU environment."
        )

    print(f"Project: {PROJECT_ROOT}")
    print(f"Venv:    {venv_dir}")
    print(f"Device:  {device}")
    print(f"Extras:  {_install_extras(args)}")
    print(f"Lock:    {lock_path or 'none'}")
    print()

    if venv_dir.exists() and not args.recreate and not python.exists():
        raise BootstrapError(
            f"{venv_dir} exists but has no interpreter. Pass --recreate to rebuild it."
        )

    if not venv_dir.exists() or args.recreate:
        command = [args.python, "-m", "venv"]
        if args.recreate and venv_dir.exists():
            command.append("--clear")
        command.append(str(venv_dir))
        _run(command, dry_run=args.dry_run)
    elif args.dry_run is False:
        _check_venv_isolation(venv_dir)

    if not args.dry_run and not python.exists():
        raise BootstrapError(f"Venv interpreter was not created: {python}")

    _run(
        [str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        dry_run=args.dry_run,
    )

    if device in TORCH_INDEXES:
        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                f"torch=={TORCH_VERSION}",
                f"torchaudio=={TORCHAUDIO_VERSION}",
                "--index-url",
                TORCH_INDEXES[device],
            ],
            dry_run=args.dry_run,
        )

    install_command = [
        str(python),
        "-m",
        "pip",
        "install",
        "-e",
        _install_extras(args),
    ]
    if lock_path is not None:
        install_command.extend(["-c", str(lock_path)])
    _run(install_command, dry_run=args.dry_run)

    _run([str(python), "-m", "pip", "check"], dry_run=args.dry_run)

    if not args.skip_preflight:
        report = _run(
            [str(python), "-m", "voix", "preflight", "--json"],
            dry_run=args.dry_run,
            check=False,
        )
        if report.returncode not in (0, None):
            print(
                "\nEnvironment installed, but preflight did not pass. This is expected "
                "on a clean checkout until configs/artifacts.yaml files are restored.",
                file=sys.stderr,
            )

    print("\nEnvironment ready.")
    if os.name == "nt":
        print(rf"  Activate: {venv_dir}\Scripts\activate")
    else:
        print(f"  Activate: source {venv_dir}/bin/activate")
    print(f"  Verify artifacts: {python} scripts/setup/verify_artifacts.py")
    print(f"  Prefetch caches:  {python} scripts/setup/prefetch_models.py --check")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BootstrapError as exc:
        print(f"Bootstrap error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}: {exc.cmd}", file=sys.stderr)
        raise SystemExit(exc.returncode or 1)
