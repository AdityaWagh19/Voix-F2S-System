# VOIX Unified Command-Line Interface (CLI).
from __future__ import annotations

import sys

# Configure UTF-8 stdout on Windows consoles to prevent charmap errors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    console = Console(safe_box=True)
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None

from voix import __version__
from voix.utils.hardware import detect_hardware


ASCII_BANNER = r'''
 __      __   ____    _____  __   __
 \ \    / /  / __ \  |_   _| \ \ / /
  \ \  / /  | |  | |   | |    \ V / 
   \ \/ /   | |  | |   | |     > <  
    \  /    | |__| |  _| |_   / . \ 
     \/      \____/  |_____| /_/ \_\
'''

DEFAULT_DEMO_TEXT = (
    "The human voice reflects both physical anatomy and lived culture, "
    "carrying the unique cadence of personal identity."
)

PRESET_FACES = [
    ("1", "Zendaya (Young Female)", "data/test_subjects/C1_01_young_female_Zendaya.jpg"),
    ("2", "Timothee Chalamet (Young Male)", "data/test_subjects/C1_02_young_male_Timothee_Chalamet.jpg"),
    ("3", "Barack Obama (Adult Male)", "data/test_subjects/C2_11_black_male_Barack_Obama.jpg"),
    ("4", "Priyanka Chopra (South Asian Female)", "data/test_subjects/C2_08_south_asian_female_Priyanka_Chopra.jpg"),
    ("5", "Shah Rukh Khan (South Asian Male)", "data/test_subjects/C2_07_south_asian_male_Shah_Rukh_Khan.jpg"),
    ("6", "Default Showcase Subject", "demo/subject_face.png"),
]


def print_banner() -> None:
    if HAS_RICH and console:
        title_text = Text(ASCII_BANNER, style="bold cyan")
        subtitle = Text(
            f"VOIX v{__version__} | Zero-Shot Probabilistic Face-to-Voice Synthesis\n"
            "Biometrics -> CVAE Latent Manifold -> StyleTTS2 Neural Audio",
            style="dim white",
        )
        console.print(Panel(Text.assemble(title_text, subtitle), border_style="cyan", expand=False))
    else:
        print(ASCII_BANNER)
        print(f"VOIX v{__version__} - Zero-Shot Probabilistic Face-to-Voice Synthesis")
        print("=" * 65)


def cmd_info(args: argparse.Namespace) -> None:
    print_banner()
    profile = detect_hardware()
    
    root_dir = Path(__file__).resolve().parent.parent
    checkpoints = [
        ("ArcFace (InsightFace)", "checkpoints/models/buffalo_l/w600k_r50.onnx"),
        ("MediaPipe FaceLandmarker", "checkpoints/mediapipe/face_landmarker.task"),
        ("ECAPA-TDNN (SpeechBrain)", "checkpoints/speechbrain/spkrec-ecapa/embedding_model.ckpt"),
        ("CVAE Diversity Model", "checkpoints/cvae_cfvn_face192_latent128_diversity.pt"),
        ("StyleAdapter Checkpoint", "checkpoints/style_adapter_best.pt"),
        ("Environment Config", "configs/environment.yaml"),
    ]

    if HAS_RICH and console:
        table = Table(title="System & Hardware Status", border_style="cyan")
        table.add_column("Property", style="bold white")
        table.add_column("Status / Detail", style="green")

        table.add_row("PyTorch Version", torch.__version__)
        table.add_row("Hardware Profile", profile.name)
        table.add_row("Accelerator", profile.accelerator.upper())
        table.add_row("Target Device", str(profile.device))
        table.add_row("Memory Budget", f"{profile.total_memory_gb:.1f} GB")
        table.add_row("Precision", profile.precision.upper())
        table.add_row("Pipeline Mode", profile.pipeline_mode)
        console.print(table)

        ckpt_table = Table(title="Checkpoint & Model Verification", border_style="cyan")
        ckpt_table.add_column("Component", style="bold white")
        ckpt_table.add_column("Path", style="dim white")
        ckpt_table.add_column("Status", style="bold")

        for name, rel_path in checkpoints:
            full_p = root_dir / rel_path
            if full_p.exists():
                sz_mb = full_p.stat().st_size / (1024 * 1024)
                ckpt_table.add_row(name, rel_path, f"[green]OK ({sz_mb:.1f} MB)[/green]")
            else:
                ckpt_table.add_row(name, rel_path, "[red]MISSING[/red]")
        console.print(ckpt_table)
    else:
        print("\n--- Hardware & System Status ---")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  Profile: {profile.name}")
        print(f"  Accelerator: {profile.accelerator}")
        print(f"  Device: {profile.device}")
        print(f"  Memory: {profile.total_memory_gb:.1f} GB")
        print("\n--- Checkpoints ---")
        for name, rel_path in checkpoints:
            full_p = root_dir / rel_path
            status = f"OK ({full_p.stat().st_size / (1024*1024):.1f} MB)" if full_p.exists() else "MISSING"
            print(f"  {name:<25}: {status}")


def cmd_generate(args: argparse.Namespace) -> None:
    from voix.inference.pipeline import VoixPipeline

    face_p = Path(args.face)
    if not face_p.exists():
        if HAS_RICH and console:
            console.print(f"[bold red]Error:[/bold red] Face image not found at '[cyan]{face_p}[/cyan]'")
        else:
            print(f"Error: Face image not found at '{face_p}'")
        sys.exit(1)

    out_p = Path(args.out)
    num_voices = max(1, args.voices)
    
    if HAS_RICH and console:
        console.print(f"[bold cyan]Loading VOIX Pipeline on {args.device or 'auto'}...[/bold cyan]")
    else:
        print(f"Loading VOIX Pipeline on {args.device or 'auto'}...")

    pipeline = VoixPipeline(device=args.device, enable_watermarking=args.watermark)

    t0 = time.time()
    results = pipeline.generate(
        face_input=str(face_p),
        text=args.text,
        num_voices=num_voices,
        temperature=args.temp,
        accent=args.accent,
        emotion=args.emotion,
        watermark=args.watermark,
    )
    latency = time.time() - t0

    out_files = []
    if num_voices == 1:
        if out_p.suffix.lower() != ".wav":
            out_p.mkdir(parents=True, exist_ok=True)
            out_file = out_p / "generated_voice.wav"
        else:
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_file = out_p
        
        wav = results[0]["waveform"].numpy().squeeze()
        sf.write(str(out_file), wav, 24000)
        out_files.append((out_file, len(wav) / 24000, results[0]))
    else:
        out_p.mkdir(parents=True, exist_ok=True)
        for i, res in enumerate(results):
            out_file = out_p / f"candidate_{i+1}.wav"
            wav = res["waveform"].numpy().squeeze()
            sf.write(str(out_file), wav, 24000)
            out_files.append((out_file, len(wav) / 24000, res))

    if HAS_RICH and console:
        table = Table(title="Synthesis Output Summary", border_style="green")
        table.add_column("Voice", style="bold white")
        table.add_column("Duration", style="cyan")
        table.add_column("File Path", style="bold yellow")

        for i, (fpath, dur, _) in enumerate(out_files):
            table.add_row(f"Candidate #{i+1}", f"{dur:.2f}s", str(fpath))

        console.print(table)
        console.print(f"[bold green]Success![/bold green] Total inference time: [bold cyan]{latency:.2f}s[/bold cyan]")
    else:
        print(f"\nSynthesis successful in {latency:.2f}s:")
        for i, (fpath, dur, _) in enumerate(out_files):
            print(f"  Candidate #{i+1}: {fpath} ({dur:.2f}s)")


def cmd_interactive(args: argparse.Namespace) -> None:
    print_banner()

    root_dir = Path(__file__).resolve().parent.parent

    if HAS_RICH and console:
        console.print("[bold yellow]Step 1: Choose a Face Subject[/bold yellow]")
        for key, name, rel_path in PRESET_FACES:
            console.print(f"  [cyan][{key}][/cyan] {name}")
        console.print("  [cyan][C][/cyan] Custom Image Path (enter your own)")
    else:
        print("Step 1: Choose a Face Subject")
        for key, name, _ in PRESET_FACES:
            print(f"  [{key}] {name}")
        print("  [C] Custom Image Path")

    choice = input("\nSelect option [1-6 or C, default=1]: ").strip() or "1"
    
    face_path = None
    if choice.upper() == "C":
        custom_p = input("Enter absolute or relative path to image: ").strip()
        face_path = Path(custom_p)
    else:
        matched = [p for k, _, p in PRESET_FACES if k == choice]
        if matched:
            face_path = root_dir / matched[0]
        else:
            face_path = root_dir / PRESET_FACES[0][2]

    if not face_path.exists():
        if HAS_RICH and console:
            console.print(f"[bold red]Error:[/bold red] File not found: '{face_path}'")
        else:
            print(f"Error: File not found: '{face_path}'")
        return

    if HAS_RICH and console:
        console.print("\n[bold yellow]Step 2: Enter Text Prompt[/bold yellow]")
        console.print(f"[dim]Default: \"{DEFAULT_DEMO_TEXT}\"[/dim]")
    else:
        print("\nStep 2: Enter Text Prompt")
        print(f"Default: \"{DEFAULT_DEMO_TEXT}\"")
    
    text_input = input("Text [press Enter for default]: ").strip()
    text = text_input if text_input else DEFAULT_DEMO_TEXT

    voices_input = input("Number of candidate voices [1-5, default=1]: ").strip()
    try:
        num_voices = int(voices_input) if voices_input else 1
    except ValueError:
        num_voices = 1

    print("\nStep 3: Optional Expressivity")
    accent_opts = ["none", "general_american", "british_rp", "australian", "indian_english", "irish"]
    for idx, acc in enumerate(accent_opts):
        print(f"  [{idx}] {acc}")
    acc_idx = input("Select accent [default=0]: ").strip() or "0"
    try:
        accent = None if acc_idx == "0" else accent_opts[int(acc_idx)]
    except (ValueError, IndexError):
        accent = None

    emotion_opts = ["none", "neutral", "happy", "sad", "angry", "whispered"]
    for idx, emo in enumerate(emotion_opts):
        print(f"  [{idx}] {emo}")
    emo_idx = input("Select emotion [default=0]: ").strip() or "0"
    try:
        emotion = None if emo_idx == "0" else emotion_opts[int(emo_idx)]
    except (ValueError, IndexError):
        emotion = None

    out_dir = root_dir / "demo" / "interactive_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    class DummyArgs:
        face = str(face_path)
        text = text
        out = str(out_dir)
        voices = num_voices
        temp = 1.0
        accent = accent
        emotion = emotion
        device = None
        watermark = True

    cmd_generate(DummyArgs())


def cmd_showcase(args: argparse.Namespace) -> None:
    from voix.inference.cli import run_showcase_suite
    from voix.inference.pipeline import VoixPipeline

    print_banner()
    face_p = Path(args.face)
    if not face_p.exists():
        print(f"Error: Face image not found at '{face_p}'")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    print(f"Initializing pipeline for full 6-category showcase suite on {face_p}...")
    pipeline = VoixPipeline(device=args.device, enable_watermarking=True)
    run_showcase_suite(
        pipeline=pipeline,
        face_image_path=face_p,
        output_dir=out_dir,
        text=args.text,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="voix",
        description="VOIX: Zero-Shot Probabilistic Face-to-Voice Synthesis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    p_gen = subparsers.add_parser("generate", help="Synthesize voice from a face image")
    p_gen.add_argument("-f", "--face", type=str, required=True, help="Path to input face image (.jpg / .png)")
    p_gen.add_argument("-t", "--text", type=str, default=DEFAULT_DEMO_TEXT, help="Text prompt to synthesize")
    p_gen.add_argument("-o", "--out", type=str, default="output.wav", help="Output WAV file or directory")
    p_gen.add_argument("-v", "--voices", type=int, default=1, help="Number of candidate voice hypotheses (1-5)")
    p_gen.add_argument("--temp", type=float, default=1.0, help="CVAE sampling temperature for diversity")
    p_gen.add_argument("--accent", type=str, default=None, choices=["general_american", "british_rp", "australian", "indian_english", "irish", "scottish"], help="Optional accent modulation")
    p_gen.add_argument("--emotion", type=str, default=None, choices=["neutral", "happy", "sad", "angry", "whispered", "surprised"], help="Optional emotional register")
    p_gen.add_argument("--device", type=str, default=None, help="Hardware device ('cuda', 'cpu')")
    p_gen.add_argument("--no-watermark", dest="watermark", action="store_false", help="Disable AudioSeal watermarking")

    subparsers.add_parser("interactive", help="Launch interactive showcase wizard for live demos")

    p_show = subparsers.add_parser("showcase", help="Generate complete 27-asset demonstration showcase")
    p_show.add_argument("-f", "--face", type=str, default="demo/subject_face.png", help="Path to face image")
    p_show.add_argument("-o", "--output-dir", type=str, default="demo/showcase", help="Output directory")
    p_show.add_argument("-t", "--text", type=str, default=DEFAULT_DEMO_TEXT, help="Text prompt")
    p_show.add_argument("--device", type=str, default=None, help="Hardware device ('cuda', 'cpu')")

    subparsers.add_parser("info", help="Show hardware diagnostics and checkpoint status")

    if len(sys.argv) == 1:
        print_banner()
        parser.print_help()
        print("\nQuickstart Examples:")
        print("  voix interactive                     # Run the live interactive showcase")
        print("  voix generate -f demo/face.jpg       # Generate voice from image")
        print("  voix info                            # Check GPU and model status")
        sys.exit(0)

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "interactive":
        cmd_interactive(args)
    elif args.command == "showcase":
        cmd_showcase(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
