"""Demonstration Showcase Generator for VOIX (Wrapper for voix.inference.cli).

Executes the full showcase demonstration generation using the universal VOIX pipeline runner.
Usage:
    python scripts/generate_demo.py
    python scripts/generate_demo.py --face demo/subject_face.png --output-dir demo
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voix.inference.cli import main

if __name__ == "__main__":
    # If no arguments provided, default to showcase mode on demo folder
    if len(sys.argv) == 1:
        sys.argv.extend(["--mode", "showcase", "--face", "demo/subject_face.png", "--output-dir", "demo"])
    main()
