"""Data firewall for the canonical face-to-voice training path.

AMI, CREMA-D, and GLOBE may train the naturalness/emotion/accent experiments,
but they must never train the CFVN, CVAE, or style adapter.  This module scans
the core training code for references to those corpora and fails loudly.

It is a static, code-level guard.  It does not replace a manifest-level
provenance check, and it does not scan data files.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

from voix.utils.paths import project_path

FIREWALL_SCHEMA = "voix.data_firewall.v1"

# Tokens that identify an expression/naturalness corpus, not a generic word.
FORBIDDEN_PATTERNS: Dict[str, re.Pattern[str]] = {
    "crema_d": re.compile(r"crema[-_ ]?d", re.IGNORECASE),
    "globe": re.compile(
        r"\bglobe(?:[-_ ]?(?:audio|accent|metadata|with_speakers|selected))?\b",
        re.IGNORECASE,
    ),
    "ami": re.compile(
        r"\bami(?:[-_ ]?(?:lora|corpus|meeting|audio|headset))?\b",
        re.IGNORECASE,
    ),
    "msp_podcast": re.compile(r"msp[-_ ]?podcast", re.IGNORECASE),
}

CORE_TRAINING_GLOBS: Sequence[str] = (
    "voix/training/trainer_cfvn.py",
    "voix/training/trainer_cvae_v3_diversity.py",
    "voix/training/trainer_adapter_cfvn.py",
    "voix/training/data.py",
    "voix/inference/pipeline.py",
    "scripts/train_cfvn*.py",
    "scripts/train_cvae*.py",
    "scripts/train_adapter*.py",
    "scripts/train_style_adapter*.py",
    "scripts/train_face*.py",
)


@dataclass(frozen=True)
class FirewallViolation:
    path: str
    line: int
    dataset: str
    text: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "line": self.line,
            "dataset": self.dataset,
            "text": self.text,
        }


class DataFirewallError(RuntimeError):
    """Raised when core training code references a forbidden corpus."""


def core_training_files(root: Union[str, Path] = project_path()) -> List[Path]:
    """Return the existing core training files in deterministic order."""
    base = Path(root)
    found: List[Path] = []
    for pattern in CORE_TRAINING_GLOBS:
        for path in sorted(base.glob(pattern)):
            if path.is_file() and path not in found:
                found.append(path)
    return found


def scan_file(path: Union[str, Path]) -> List[FirewallViolation]:
    source = Path(path)
    violations: List[FirewallViolation] = []
    try:
        lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return violations
    for line_number, line in enumerate(lines, start=1):
        for dataset, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(line):
                violations.append(
                    FirewallViolation(
                        path=str(source),
                        line=line_number,
                        dataset=dataset,
                        text=line.strip()[:240],
                    )
                )
    return violations


def scan_core_training(
    root: Union[str, Path] = project_path(),
    *,
    files: Optional[Iterable[Union[str, Path]]] = None,
) -> Dict[str, object]:
    """Scan core training files and return a machine-readable report."""
    selected = [Path(path) for path in files] if files is not None else core_training_files(root)
    violations: List[FirewallViolation] = []
    for path in selected:
        violations.extend(scan_file(path))
    return {
        "schema_version": FIREWALL_SCHEMA,
        "passed": not violations,
        "scanned": [str(path) for path in selected],
        "violations": [violation.to_dict() for violation in violations],
    }


def assert_no_core_contamination(
    root: Union[str, Path] = project_path(),
    *,
    files: Optional[Iterable[Union[str, Path]]] = None,
) -> None:
    report = scan_core_training(root, files=files)
    if not report["passed"]:
        details = "; ".join(
            f"{item['path']}:{item['line']} ({item['dataset']})"
            for item in report["violations"]
        )
        raise DataFirewallError(
            "core training code references expression/naturalness corpora: " + details
        )
