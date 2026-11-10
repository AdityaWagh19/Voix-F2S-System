"""VOIX Production-Grade Command-Line Interface.

Fully decoupled, professional CLI layer built on top of the VoixService abstraction.
Exposes deep parametric control without exposing internal neural model implementations.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure UTF-8 on Windows consoles to prevent charmap errors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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
from voix.models.accent_control import AccentType
from voix.models.emotion_control import EmotionType
from voix.service import SynthesisRequest, SynthesisResponse, VoixService


ASCII_BANNER = r"""
 __      __   ____    _____  __   __
 \ \    / /  / __ \  |_   _| \ \ / /
  \ \  / /  | |  | |   | |    \ V / 
   \ \/ /   | |  | |   | |     > <  
    \  /    | |__| |  _| |_   / . \ 
     \/      \____/  |_____| /_/ \_"""

DEFAULT_TEXT = "Hello. This voice was generated from a face image."

ACCENT_CHOICES = tuple(accent.value for accent in AccentType)
EMOTION_CHOICES = tuple(emotion.value for emotion in EmotionType)


def _bounded_int(minimum: int, maximum: int):
    """Argparse type for an integer in a closed range."""

    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
        if not minimum <= number <= maximum:
            raise argparse.ArgumentTypeError(
                f"must be between {minimum} and {maximum}, got {number}"
            )
        return number

    return parse


def _bounded_float(minimum: float, maximum: float):
    """Argparse type for a float in a closed range."""

    def parse(value: str) -> float:
        try:
            number = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from exc
        if not minimum <= number <= maximum:
            raise argparse.ArgumentTypeError(
                f"must be between {minimum:g} and {maximum:g}, got {number:g}"
            )
        return number

    return parse


def _progress_reporter(args: argparse.Namespace):
    """Return a stderr progress callback that keeps stdout machine-readable."""

    def report(message: str) -> None:
        if not getattr(args, "quiet", False) and not getattr(args, "json", False):
            print(f"[voix] {message}", file=sys.stderr, flush=True)

    return report


def _watermark_requested(args: argparse.Namespace) -> bool:
    """Watermarking is opt-in for live demos; --no-watermark is a safe alias."""
    return bool(getattr(args, "watermark", False)) and not bool(
        getattr(args, "no_watermark", False)
    )


def _print_preflight(report: Dict[str, Any], quiet: bool = False) -> None:
    for warning in report.get("warnings", []):
        print(f"Preflight warning: {warning}", file=sys.stderr)
    if not report.get("ok", False):
        for error in report.get("errors", []):
            print(f"Preflight error: {error}", file=sys.stderr)
    elif not quiet:
        print("[voix] Preflight passed.", file=sys.stderr)


def print_banner(quiet: bool = False) -> None:
    """Render the branded terminal header if not in quiet mode."""
    if quiet:
        return
    if HAS_RICH and console:
        title_text = Text(ASCII_BANNER, style="bold cyan")
        subtitle = Text(
            f"VOIX v{__version__} | Zero-Shot Probabilistic Face-to-Voice System\n"
            "Production-Grade Biometric-to-Acoustic Synthesis",
            style="dim white",
        )
        console.print(Panel(Text.assemble(title_text, subtitle), border_style="cyan", expand=False))
    else:
        print(ASCII_BANNER)
        print(f"VOIX v{__version__} - Zero-Shot Probabilistic Face-to-Voice System")
        print("=" * 65)


def handle_synth(args: argparse.Namespace, service: VoixService) -> int:
    """Execute synthesis for single image with full customization."""
    # Resolve input text
    text = args.text
    if args.text_file:
        tf_path = Path(args.text_file)
        if not tf_path.exists():
            print(f"Error: Text file not found at '{args.text_file}'", file=sys.stderr)
            return 1
        text = tf_path.read_text(encoding="utf-8").strip()

    if not text:
        text = DEFAULT_TEXT

    request = SynthesisRequest(
        face_image=args.face,
        text=text,
        output_path=args.out,
        num_voices=args.voices,
        temperature=args.temp,
        seed=args.seed,
        gender=args.gender,
        age=args.age,
        accent=args.accent,
        accent_strength=args.accent_strength,
        emotion=args.emotion,
        emotion_intensity=args.emotion_intensity,
        watermark=_watermark_requested(args),
    )

    try:
        preflight = service.preflight(request)
        if args.json and not preflight["ok"]:
            print(json.dumps(preflight, indent=2))
            return 1
        _print_preflight(preflight, quiet=args.quiet or args.json)
        if not preflight["ok"]:
            return 1

        if not args.quiet and not args.json:
            print_banner()
            if HAS_RICH and console:
                console.print(f"[bold cyan]Synthesizing voice from '[white]{args.face}[/white]'...[/bold cyan]")
            else:
                print(f"Synthesizing voice from '{args.face}'...")

        response: SynthesisResponse = service.synthesize(
            request,
            progress_callback=_progress_reporter(args),
        )

        if response.fallback_reason and not args.quiet and not args.json:
            print(
                f"Warning: {response.fallback_reason}\n"
                "The request was completed on CPU.",
                file=sys.stderr,
            )
        if not args.quiet and not args.json:
            backends = {cand.backend for cand in response.candidates}
            if "fallback" in backends:
                print(
                    "Warning: the neural TTS backend was unavailable; the offline "
                    "formant fallback produced this audio. Do not present it as a "
                    "neural voice demo.",
                    file=sys.stderr,
                )

        # Output formatting
        if args.json:
            print(json.dumps(response.to_dict(), indent=2))
            return 0

        if args.quiet:
            for p in response.saved_files:
                print(p)
            return 0

        if HAS_RICH and console:
            table = Table(title="Synthesis Output Summary", border_style="green")
            table.add_column("Candidate", style="bold white")
            table.add_column("Duration", style="cyan")
            table.add_column("Estimated F0", style="magenta")
            table.add_column("Watermarked", style="dim")
            table.add_column("Output File", style="bold yellow")

            for cand in response.candidates:
                table.add_row(
                    f"Voice #{cand.index}",
                    f"{cand.duration_seconds:.2f}s",
                    f"{cand.f0_mean_hz:.1f} Hz" if cand.f0_mean_hz > 0 else "N/A",
                    "Yes (16-bit)" if cand.is_watermarked else "No",
                    cand.file_path or "(not saved)",
                )

            console.print(table)
            console.print(
                f"[bold green]Success![/bold green] Generated {len(response.candidates)} "
                f"voice(s) in [bold cyan]{response.latency_seconds:.2f}s[/bold cyan] "
                f"on [bold]{response.execution_device or 'unknown'}[/bold]"
            )
        else:
            print(
                f"\nSynthesis successful ({response.latency_seconds:.2f}s, "
                f"device={response.execution_device or 'unknown'}):"
            )
            for cand in response.candidates:
                print(f"  Voice #{cand.index}: {cand.duration_seconds:.2f}s -> {cand.file_path}")

        return 0

    except Exception as e:
        print(f"Synthesis Error: {e}", file=sys.stderr)
        return 2


def handle_batch(args: argparse.Namespace, service: VoixService) -> int:
    """Execute batch synthesis over a directory of face images."""
    in_dir = Path(args.dir)
    if not in_dir.exists() or not in_dir.is_dir():
        print(f"Error: Directory not found at '{args.dir}'", file=sys.stderr)
        return 1

    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in valid_exts])

    if not image_files:
        print(f"Error: No image files found in '{args.dir}'", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    if out_dir.suffix.lower() == ".wav":
        print(
            "Error: batch output must be a directory, not a .wav file path.",
            file=sys.stderr,
        )
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    text = args.text or DEFAULT_TEXT
    if args.text_file:
        tf_path = Path(args.text_file)
        if tf_path.exists():
            text = tf_path.read_text(encoding="utf-8").strip()

    requests = []
    for img in image_files:
        target_wav = out_dir / f"{img.stem}_voice.wav"
        req = SynthesisRequest(
            face_image=str(img),
            text=text,
            output_path=str(target_wav),
            num_voices=args.voices,
            temperature=args.temp,
            accent=args.accent,
            accent_strength=args.accent_strength,
            emotion=args.emotion,
            emotion_intensity=args.emotion_intensity,
            watermark=_watermark_requested(args),
        )
        requests.append(req)

    preflight = service.preflight(requests[0])
    if not args.quiet:
        _print_preflight(preflight, quiet=args.json)
    if not preflight["ok"]:
        if args.json:
            print(json.dumps(preflight, indent=2))
        return 1

    if not args.quiet:
        print_banner()
        print(f"Processing batch of {len(requests)} face images -> '{out_dir}'...")

    def batch_progress(done: int, total: int, _response) -> None:
        if not args.quiet and not args.json:
            print(f"[voix] {done}/{total} subjects complete", file=sys.stderr, flush=True)

    t0 = time.time()
    responses = service.synthesize_batch(requests, progress_callback=batch_progress)
    total_time = time.time() - t0

    if args.json:
        print(json.dumps([r.to_dict() for r in responses], indent=2))
    elif not args.quiet:
        print(f"Batch completed: {len(responses)} subjects synthesized in {total_time:.2f}s.")
        print(f"Results saved to '{out_dir}'.")

    return 0


def handle_interactive(service: VoixService) -> int:
    """Interactive CLI wizard with sensible defaults (no hardcoded presets)."""
    print_banner()

    if HAS_RICH and console:
        console.print("[bold yellow]VOIX Interactive Synthesis Wizard[/bold yellow]")
        console.print("[dim]Press Enter at any prompt to accept the default value.[/dim]\n")
    else:
        print("VOIX Interactive Synthesis Wizard")
        print("Press Enter at any prompt to accept the default value.\n")

    # 1. Face Image or preset ID
    default_face = "young_female_resonant"
    face_input = input(
        f"Input face image path or preset ID [{default_face}]: "
    ).strip() or default_face

    # 2. Text Prompt
    print(f"\nDefault text: \"{DEFAULT_TEXT}\"")
    text_input = input("Text prompt to synthesize [Enter for default]: ").strip()
    text = text_input if text_input else DEFAULT_TEXT

    # 3. Output path
    default_out = "output.wav"
    out_input = input(f"Output destination path [{default_out}]: ").strip() or default_out

    # 4. Number of voices (candidates)
    voices_input = input("Number of candidate voices (1-5) [1]: ").strip() or "1"
    try:
        num_voices = max(1, min(5, int(voices_input)))
    except ValueError:
        num_voices = 1

    # 5. Temperature
    temp_input = input("Diversity sampling temperature (0.5 - 1.5) [1.0]: ").strip() or "1.0"
    try:
        temperature = float(temp_input)
    except ValueError:
        temperature = 1.0

    # 6. Optional Expressivity
    caps = service.get_capabilities()
    
    print(f"\nAvailable Accents: none, {', '.join(caps['accents'])}")
    accent_input = input("Select accent [none]: ").strip() or None
    accent = accent_input if accent_input in caps["accents"] else None

    print(f"Available Emotions: none, {', '.join(caps['emotions'])}")
    emotion_input = input("Select emotion [none]: ").strip() or None
    emotion = emotion_input if emotion_input in caps["emotions"] else None

    wm_input = input("Embed AudioSeal watermark? [y/N]: ").strip().lower()
    watermark = wm_input in {"y", "yes"}

    # Construct request
    request = SynthesisRequest(
        face_image=str(face_input),
        text=text,
        output_path=out_input,
        num_voices=num_voices,
        temperature=temperature,
        accent=accent,
        emotion=emotion,
        watermark=watermark,
    )

    class DummyArgs:
        face = request.face_image
        text = request.text
        text_file = None
        out = request.output_path
        voices = request.num_voices
        temp = request.temperature
        seed = None
        gender = None
        age = None
        accent = request.accent
        accent_strength = request.accent_strength
        emotion = request.emotion
        emotion_intensity = request.emotion_intensity
        watermark = request.watermark
        no_watermark = not request.watermark
        quiet = False
        json = False

    return handle_synth(DummyArgs(), service)


def handle_info(service: VoixService, as_json: bool = False) -> int:
    """Display system diagnostics, hardware profile, and model checkpoints."""
    diag = service.get_diagnostics()

    if as_json:
        print(json.dumps(diag, indent=2))
        return 0

    print_banner()

    if HAS_RICH and console:
        # Hardware table
        hw_table = Table(title="System & Hardware Diagnostic", border_style="cyan")
        hw_table.add_column("Property", style="bold white")
        hw_table.add_column("Value / Status", style="green")

        hw_table.add_row("PyTorch Version", diag["pytorch_version"])
        hw_table.add_row("Python Version", diag["python_version"])
        hw_table.add_row("Hardware Profile", diag["hardware_profile"])
        hw_table.add_row("Accelerator", diag["accelerator"].upper())
        hw_table.add_row("Target Device", diag["device"])
        hw_table.add_row("Memory Budget", f"{diag['memory_budget_gb']:.1f} GB")
        hw_table.add_row("Precision", diag["precision"].upper())
        hw_table.add_row("Pipeline Mode", diag["pipeline_mode"])
        hw_table.add_row("Offline Mode", "ON" if diag["offline"] else "off")
        console.print(hw_table)

        # Checkpoints table
        ckpt_table = Table(title="Neural Backbones & Checkpoints", border_style="cyan")
        ckpt_table.add_column("Component", style="bold white")
        ckpt_table.add_column("Relative Path", style="dim white")
        ckpt_table.add_column("Role", style="dim")
        ckpt_table.add_column("Status", style="bold")

        for ckpt in diag["checkpoints"]:
            if ckpt["exists"]:
                status_str = f"[green]OK ({ckpt['size_mb']:.1f} MB)[/green]"
            elif ckpt.get("required", True):
                status_str = "[red]MISSING[/red]"
            else:
                status_str = "[yellow]MISSING (optional)[/yellow]"
            ckpt_table.add_row(
                ckpt["component"],
                ckpt["relative_path"],
                ckpt.get("role", ""),
                status_str,
            )

        console.print(ckpt_table)

        cache_table = Table(title="Model Caches", border_style="cyan")
        cache_table.add_column("Cache", style="bold white")
        cache_table.add_column("Path", style="dim white")
        cache_table.add_column("Files", style="green")
        for name, cache in diag["caches"].items():
            cache_table.add_row(name, cache["path"], str(cache["files"]))
        console.print(cache_table)
    else:
        print("\n--- System Diagnostics ---")
        print(f"  PyTorch   : {diag['pytorch_version']}")
        print(f"  Python    : {diag['python_version']}")
        print(f"  Device    : {diag['device']} ({diag['accelerator']})")
        print(f"  Memory    : {diag['memory_budget_gb']:.1f} GB ({diag['precision']})")
        print(f"  Offline   : {'ON' if diag['offline'] else 'off'}")
        print("\n--- Checkpoints ---")
        for ckpt in diag["checkpoints"]:
            if ckpt["exists"]:
                stat = f"OK ({ckpt['size_mb']:.1f} MB)"
            elif ckpt.get("required", True):
                stat = "MISSING"
            else:
                stat = "MISSING (optional)"
            print(f"  {ckpt['component']:<30}: {stat}")
        print("\n--- Model Caches ---")
        for name, cache in diag["caches"].items():
            print(f"  {name:<30}: {cache['files']} files at {cache['path']}")

    return 0


def handle_capabilities(service: VoixService, as_json: bool = False) -> int:
    """Display or return system capabilities."""
    caps = service.get_capabilities()
    if as_json:
        print(json.dumps(caps, indent=2))
        return 0

    print_banner()
    if HAS_RICH and console:
        table = Table(title="VOIX System Capabilities", border_style="magenta")
        table.add_column("Dimension", style="bold white")
        table.add_column("Available Options", style="cyan")

        table.add_row("Regional Accents", ", ".join(caps["accents"]))
        table.add_row("Affective Emotions", ", ".join(caps["emotions"]))
        console.print(table)
    else:
        print("\n--- VOIX Capabilities ---")
        for k, v in caps.items():
            print(f"  {k:<22}: {', '.join(v)}")
    return 0



def handle_preflight(args: argparse.Namespace, service: VoixService) -> int:
    """Check artifacts, caches, device, and output paths without synthesizing."""
    request = None
    if getattr(args, "face", None):
        request = SynthesisRequest(
            face_image=args.face,
            text=args.text or DEFAULT_TEXT,
            output_path=getattr(args, "out", None),
            num_voices=getattr(args, "voices", 1),
            accent=getattr(args, "accent", None),
            emotion=getattr(args, "emotion", None),
            watermark=_watermark_requested(args),
        )

    report = service.preflight(request)

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1

    print_banner()
    if HAS_RICH and console:
        table = Table(title="VOIX Preflight", border_style="cyan")
        table.add_column("Check", style="bold white")
        table.add_column("Status", style="bold")
        table.add_column("Detail", style="dim")
        for check in report["checks"]:
            status = "[green]OK[/green]" if check["exists"] else (
                "[red]MISSING[/red]" if check.get("required", True)
                else "[yellow]MISSING[/yellow]"
            )
            detail = check["path"]
            if check.get("cached") is not None:
                detail += " (cached)" if check["cached"] else " (not cached)"
            table.add_row(check["name"], status, detail)
        console.print(table)
    else:
        for check in report["checks"]:
            status = "OK" if check["exists"] else "MISSING"
            print(f"  {check['name']:<35}: {status} ({check['path']})")

    for warning in report["warnings"]:
        print(f"Warning: {warning}", file=sys.stderr)
    for error in report["errors"]:
        print(f"Error: {error}", file=sys.stderr)

    print("\nPreflight passed." if report["ok"] else "\nPreflight failed.", file=sys.stderr)
    return 0 if report["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    """Build production-grade CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="voix",
        description="VOIX: Professional Zero-Shot Probabilistic Face-to-Voice System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--device", type=str, default=None, help="Target device ('cuda', 'cpu', 'mps')")

    subparsers = parser.add_subparsers(dest="command", help="Operational commands")

    # 1. Subcommand: synth (aliases: generate)
    p_synth = subparsers.add_parser("synth", aliases=["generate"], help="Synthesize voice from a face image")
    p_synth.add_argument("-f", "--face", type=str, required=True, help="Input face photograph (.jpg, .png, etc.)")
    p_synth.add_argument("--device", type=str, default=argparse.SUPPRESS, help="Execution device ('cuda', 'cpu', 'mps')")
    p_synth.add_argument("-t", "--text", type=str, default=None, help="Text string to synthesize")
    p_synth.add_argument("--text-file", type=str, default=None, help="Path to text file containing prompt")
    p_synth.add_argument("-o", "--out", type=str, default="output.wav", help="Output WAV path or directory")
    p_synth.add_argument(
        "-v",
        "--voices",
        type=_bounded_int(1, 10),
        default=1,
        metavar="N",
        help="Number of candidate hypotheses (1-10)",
    )
    p_synth.add_argument(
        "--temp",
        type=_bounded_float(0.1, 3.0),
        default=1.0,
        metavar="T",
        help="CVAE latent diversity temperature (0.1 - 3.0)",
    )
    p_synth.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling")
    p_synth.add_argument("--gender", type=int, choices=[0, 1], default=None, help="Force gender (0=Female, 1=Male)")
    p_synth.add_argument(
        "--age",
        type=_bounded_float(0.0, 120.0),
        default=None,
        metavar="YEARS",
        help="Force estimated age in years (0-120)",
    )

    # Expressivity
    p_synth.add_argument(
        "--accent",
        type=str,
        choices=ACCENT_CHOICES,
        default=None,
        help="Accent: us, england, indian_english, or australian",
    )
    p_synth.add_argument(
        "--accent-strength",
        type=_bounded_float(0.0, 1.0),
        default=1.0,
        metavar="X",
        help="Accent intensity (0.0 to 1.0)",
    )
    p_synth.add_argument(
        "--emotion",
        type=str,
        choices=EMOTION_CHOICES,
        default=None,
        help="Emotion: neutral, happy, sad, or angry",
    )
    p_synth.add_argument(
        "--emotion-intensity",
        type=_bounded_float(0.0, 1.0),
        default=1.0,
        metavar="X",
        help="Emotion intensity (0.0 to 1.0)",
    )

    # Output options.  Watermarking is opt-in for live demos; the neural
    # AudioSeal weights are an extra download and an extra GPU allocation.
    wm_group = p_synth.add_mutually_exclusive_group()
    wm_group.add_argument(
        "--watermark",
        action="store_true",
        help="Embed the AudioSeal synthetic-speech watermark (opt-in)",
    )
    wm_group.add_argument(
        "--no-watermark",
        action="store_true",
        help="Explicitly disable watermarking (the default)",
    )
    p_synth.add_argument("--json", action="store_true", help="Output machine-readable JSON metadata to stdout")
    p_synth.add_argument("-q", "--quiet", action="store_true", help="Suppress terminal banners and progress info")

    # 2. Subcommand: batch
    p_batch = subparsers.add_parser("batch", help="Batch synthesize voices for all face images in a directory")
    p_batch.add_argument("-d", "--dir", type=str, required=True, help="Input directory containing face images")
    p_batch.add_argument("--device", type=str, default=argparse.SUPPRESS, help="Execution device ('cuda', 'cpu', 'mps')")
    p_batch.add_argument("-o", "--out", type=str, default="batch_output", help="Output directory for audio files")
    p_batch.add_argument("-t", "--text", type=str, default=None, help="Common text string for all subjects")
    p_batch.add_argument("--text-file", type=str, default=None, help="Path to text prompt file")
    p_batch.add_argument(
        "-v",
        "--voices",
        type=_bounded_int(1, 10),
        default=1,
        metavar="N",
        help="Candidates per subject (1-10)",
    )
    p_batch.add_argument(
        "--temp",
        type=_bounded_float(0.1, 3.0),
        default=1.0,
        metavar="T",
        help="CVAE diversity temperature (0.1 - 3.0)",
    )
    p_batch.add_argument("--accent", type=str, choices=ACCENT_CHOICES, default=None, help="Accent")
    p_batch.add_argument(
        "--accent-strength",
        type=_bounded_float(0.0, 1.0),
        default=1.0,
        metavar="X",
        help="Accent intensity (0.0 to 1.0)",
    )
    p_batch.add_argument("--emotion", type=str, choices=EMOTION_CHOICES, default=None, help="Emotion")
    p_batch.add_argument(
        "--emotion-intensity",
        type=_bounded_float(0.0, 1.0),
        default=1.0,
        metavar="X",
        help="Emotion intensity (0.0 to 1.0)",
    )
    wm_group = p_batch.add_mutually_exclusive_group()
    wm_group.add_argument("--watermark", action="store_true", help="Embed the AudioSeal watermark")
    wm_group.add_argument("--no-watermark", action="store_true", help="Disable watermarking (the default)")
    p_batch.add_argument("--json", action="store_true", help="Output JSON results")
    p_batch.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")

    # 3. Subcommand: interactive
    p_interactive = subparsers.add_parser("interactive", help="Launch interactive step-by-step synthesis wizard")
    p_interactive.add_argument("--device", type=str, default=argparse.SUPPRESS, help="Execution device ('cuda', 'cpu', 'mps')")

    # 4. Subcommand: info (aliases: doctor)
    p_info = subparsers.add_parser("info", aliases=["doctor"], help="System hardware diagnostics and checkpoints")
    p_info.add_argument("--json", action="store_true", help="Output diagnostics as JSON")

    # 5. Subcommand: capabilities
    p_caps = subparsers.add_parser("capabilities", help="List supported accents and emotions")
    p_caps.add_argument("--json", action="store_true", help="Output capabilities as JSON")

    # 6. Subcommand: preflight
    p_pre = subparsers.add_parser(
        "preflight",
        aliases=["check"],
        help="Check artifacts, model caches, device, and output paths without synthesizing",
    )
    p_pre.add_argument("-f", "--face", type=str, default=None, help="Optional face image to validate")
    p_pre.add_argument("--device", type=str, default=argparse.SUPPRESS, help="Execution device ('cuda', 'cpu', 'mps')")
    p_pre.add_argument("-t", "--text", type=str, default=None, help="Optional synthesis text")
    p_pre.add_argument("-o", "--out", type=str, default=None, help="Optional output path to validate")
    p_pre.add_argument(
        "-v",
        "--voices",
        type=_bounded_int(1, 10),
        default=1,
        metavar="N",
        help="Candidate count to validate (1-10)",
    )
    p_pre.add_argument("--accent", type=str, choices=ACCENT_CHOICES, default=None, help="Accent")
    p_pre.add_argument("--emotion", type=str, choices=EMOTION_CHOICES, default=None, help="Emotion")
    wm_group = p_pre.add_mutually_exclusive_group()
    wm_group.add_argument("--watermark", action="store_true", help="Require the AudioSeal cache")
    wm_group.add_argument("--no-watermark", action="store_true", help="Do not require the AudioSeal cache")
    p_pre.add_argument("--json", action="store_true", help="Output the preflight report as JSON")

    return parser


def main() -> None:
    """CLI front-door router."""
    parser = build_parser()

    if len(sys.argv) == 1:
        print_banner()
        parser.print_help()
        print("\nQuickstart Examples:")
        print("  voix synth -f photo.jpg -t \"Hello world\" -o output.wav")
        print("  voix synth -f photo.jpg -v 3 --temp 1.0 -o candidates_dir/")
        print("  voix synth -f photo.jpg --accent england --emotion happy")
        print("  voix batch -d faces_folder/ -o output_folder/")
        print("  voix preflight -f photo.jpg -o output.wav")
        print("  voix interactive")
        print("  voix info")
        sys.exit(0)

    args = parser.parse_args()

    # Instantiate Application Service with target device
    service = VoixService(device=getattr(args, "device", None))

    if args.command in ["synth", "generate"]:
        sys.exit(handle_synth(args, service))
    elif args.command == "batch":
        sys.exit(handle_batch(args, service))
    elif args.command == "interactive":
        sys.exit(handle_interactive(service))
    elif args.command in ["info", "doctor"]:
        sys.exit(handle_info(service, as_json=getattr(args, "json", False)))
    elif args.command == "capabilities":
        sys.exit(handle_capabilities(service, as_json=getattr(args, "json", False)))
    elif args.command in ["preflight", "check"]:
        sys.exit(handle_preflight(args, service))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
