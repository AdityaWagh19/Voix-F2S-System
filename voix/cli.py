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
from voix.service import SynthesisRequest, SynthesisResponse, VoixService


ASCII_BANNER = r"""
 __      __   ____    _____  __   __
 \ \    / /  / __ \  |_   _| \ \ / /
  \ \  / /  | |  | |   | |    \ V / 
   \ \/ /   | |  | |   | |     > <  
    \  /    | |__| |  _| |_   / . \ 
     \/      \____/  |_____| /_/ \_"""

DEFAULT_TEXT = (
    "The human voice reflects both physical anatomy and lived culture, "
    "carrying the unique cadence of personal identity."
)


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

    # Resolve craniofacial sliders
    sliders: Dict[str, float] = {}
    if args.vtl is not None:
        sliders["face_height_to_ioc"] = float(args.vtl)
    if args.jaw is not None:
        sliders["jaw_width_to_ioc"] = float(args.jaw)
    if args.face_width is not None:
        sliders["face_width_to_ioc"] = float(args.face_width)
    if args.mandible_angle is not None:
        sliders["mandible_angle_deg"] = float(args.mandible_angle)

    request = SynthesisRequest(
        face_image=args.face,
        text=text,
        output_path=args.out,
        num_voices=max(1, args.voices),
        temperature=args.temp,
        seed=args.seed,
        gender=args.gender,
        age=args.age,
        craniofacial_sliders=sliders if sliders else None,
        accent=args.accent,
        accent_strength=args.accent_strength,
        emotion=args.emotion,
        emotion_intensity=args.emotion_intensity,
        language=args.lang,
        speech_difference=args.clinical,
        speech_severity=args.clinical_severity,
        watermark=not args.no_watermark,
    )

    try:
        pet_ctrl = None
        if getattr(args, "pet", False):
            try:
                from voix.pet import PetCompanionController
                pet_ctrl = PetCompanionController()
                pet_ctrl.launch(
                    initial_state="think",
                    initial_speech="Analyzing biometric face...",
                    scale=0.9,
                )
            except Exception:
                pet_ctrl = None

        if not args.quiet and not args.json:
            print_banner()
            if HAS_RICH and console:
                console.print(f"[bold cyan]Synthesizing voice from '[white]{args.face}[/white]'...[/bold cyan]")
            else:
                print(f"Synthesizing voice from '{args.face}'...")

        if pet_ctrl:
            pet_ctrl.set_state("speak")
            pet_ctrl.say("Synthesizing voice acoustics...", 2.5)

        response: SynthesisResponse = service.synthesize(request)

        if pet_ctrl:
            pet_ctrl.set_state("jump")
            pet_ctrl.say(f"Synthesis Complete ({response.latency_seconds:.2f}s)!", 4.0)

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
                f"voice(s) in [bold cyan]{response.latency_seconds:.2f}s[/bold cyan]"
            )
        else:
            print(f"\nSynthesis successful ({response.latency_seconds:.2f}s):")
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
            num_voices=max(1, args.voices),
            temperature=args.temp,
            language=args.lang,
            accent=args.accent,
            emotion=args.emotion,
            watermark=not args.no_watermark,
        )
        requests.append(req)

    if not args.quiet:
        print_banner()
        print(f"Processing batch of {len(requests)} face images -> '{out_dir}'...")

    t0 = time.time()
    responses = service.synthesize_batch(requests)
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

    # 1. Face Image
    default_face = "demo/subject_face.png"
    face_input = input(f"Input face image path [{default_face}]: ").strip() or default_face
    face_p = Path(face_input)
    if not face_p.exists():
        print(f"Error: Image '{face_input}' does not exist.", file=sys.stderr)
        return 1

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

    print(f"Available Languages: {', '.join(caps['languages'])}")
    lang_input = input("Select language [en]: ").strip() or "en"
    lang = lang_input if lang_input in caps["languages"] else "en"

    # Construct request
    request = SynthesisRequest(
        face_image=str(face_p),
        text=text,
        output_path=out_input,
        num_voices=num_voices,
        temperature=temperature,
        accent=accent,
        emotion=emotion,
        language=lang,
        watermark=True,
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
        vtl = None
        jaw = None
        face_width = None
        mandible_angle = None
        accent = request.accent
        accent_strength = 0.5
        emotion = request.emotion
        emotion_intensity = 0.5
        lang = request.language
        clinical = None
        clinical_severity = 0.5
        no_watermark = False
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
        hw_table.add_row("Hardware Profile", diag["hardware_profile"])
        hw_table.add_row("Accelerator", diag["accelerator"].upper())
        hw_table.add_row("Target Device", diag["device"])
        hw_table.add_row("Memory Budget", f"{diag['memory_budget_gb']:.1f} GB")
        hw_table.add_row("Precision", diag["precision"].upper())
        hw_table.add_row("Pipeline Mode", diag["pipeline_mode"])
        console.print(hw_table)

        # Checkpoints table
        ckpt_table = Table(title="Neural Backbones & Checkpoints", border_style="cyan")
        ckpt_table.add_column("Component", style="bold white")
        ckpt_table.add_column("Relative Path", style="dim white")
        ckpt_table.add_column("Status", style="bold")

        for ckpt in diag["checkpoints"]:
            status_str = f"[green]OK ({ckpt['size_mb']:.1f} MB)[/green]" if ckpt["exists"] else "[red]MISSING[/red]"
            ckpt_table.add_row(ckpt["component"], ckpt["relative_path"], status_str)

        console.print(ckpt_table)
    else:
        print("\n--- System Diagnostics ---")
        print(f"  PyTorch   : {diag['pytorch_version']}")
        print(f"  Device    : {diag['device']} ({diag['accelerator']})")
        print(f"  Memory    : {diag['memory_budget_gb']:.1f} GB ({diag['precision']})")
        print("\n--- Checkpoints ---")
        for ckpt in diag["checkpoints"]:
            stat = f"OK ({ckpt['size_mb']:.1f} MB)" if ckpt["exists"] else "MISSING"
            print(f"  {ckpt['component']:<25}: {stat}")

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

        table.add_row("Languages", ", ".join(caps["languages"]))
        table.add_row("Regional Accents", ", ".join(caps["accents"]))
        table.add_row("Affective Emotions", ", ".join(caps["emotions"]))
        table.add_row("Clinical Differences", ", ".join(caps["speech_differences"]))
        table.add_row("Craniofacial Sliders", ", ".join(caps["craniofacial_sliders"]))
        console.print(table)
    else:
        print("\n--- VOIX Capabilities ---")
        for k, v in caps.items():
            print(f"  {k:<22}: {', '.join(v)}")
    return 0



def handle_pet(args: argparse.Namespace) -> int:
    """Launch the VOIX floating companion pet or render terminal companion card."""
    from voix.pet import VoixPetWindow, render_terminal_pet, HAS_GUI

    if args.terminal or not HAS_GUI:
        render_terminal_pet(state=args.state, speech=args.speech)
        return 0

    if not args.quiet:
        print_banner()
        if HAS_RICH and console:
            console.print("[bold cyan]🐱 Launching VOIX Floating Companion on Desktop...[/bold cyan]")
            console.print("[dim]  • Left-click and drag: move anywhere on screen[/dim]")
            console.print("[dim]  • Double-click: cycle animation states[/dim]")
            console.print("[dim]  • Right-click: context menu (states, size scaling, close)[/dim]")
            console.print("[dim]Press Ctrl+C or Right-click -> Close Companion to exit.[/dim]\n")
        else:
            print("Launching VOIX Floating Companion on Desktop...")
            print("  • Left-click and drag: move anywhere on screen")
            print("  • Double-click: cycle animation states")
            print("  • Right-click: context menu (states, size scaling, close)")
            print("Press Ctrl+C or Right-click -> Close Companion to exit.\n")

    try:
        pet = VoixPetWindow(
            scale=args.scale,
            initial_state=args.state,
            initial_speech=args.speech,
        )
        pet.run()
    except KeyboardInterrupt:
        pass
    return 0


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
    p_synth.add_argument("-t", "--text", type=str, default=None, help="Text string to synthesize")
    p_synth.add_argument("--text-file", type=str, default=None, help="Path to text file containing prompt")
    p_synth.add_argument("-o", "--out", type=str, default="output.wav", help="Output WAV path or directory")
    p_synth.add_argument("-v", "--voices", type=int, default=1, help="Number of candidate hypotheses (1-10)")
    p_synth.add_argument("--temp", type=float, default=1.0, help="CVAE latent diversity temperature (0.1 - 2.0)")
    p_synth.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling")
    p_synth.add_argument("--gender", type=int, choices=[0, 1], default=None, help="Force gender (0=Female, 1=Male)")
    p_synth.add_argument("--age", type=float, default=None, help="Force estimated age in years")
    p_synth.add_argument("--lang", type=str, default="en", help="Target synthesis language (en, es, fr, de, it)")
    
    # Expressivity
    p_synth.add_argument("--accent", type=str, default=None, help="Regional English accent modulation")
    p_synth.add_argument("--accent-strength", type=float, default=0.5, help="Accent intensity (0.0 to 1.0)")
    p_synth.add_argument("--emotion", type=str, default=None, help="Affective emotional register")
    p_synth.add_argument("--emotion-intensity", type=float, default=0.5, help="Emotion intensity (0.0 to 1.0)")
    p_synth.add_argument("--clinical", type=str, default=None, help="Clinical speech difference profile")
    p_synth.add_argument("--clinical-severity", type=float, default=0.5, help="Clinical difference severity")

    # Craniofacial Sliders
    p_synth.add_argument("--vtl", type=float, default=None, help="Vocal tract length proxy / face height ratio")
    p_synth.add_argument("--jaw", type=float, default=None, help="Mandibular jaw width ratio")
    p_synth.add_argument("--face-width", type=float, default=None, help="Bizygomatic face width ratio")
    p_synth.add_argument("--mandible-angle", type=float, default=None, help="Mandibular vertex angle in degrees")

    # Output options
    p_synth.add_argument("--no-watermark", action="store_true", help="Disable AudioSeal synthetic speech watermark")
    p_synth.add_argument("--json", action="store_true", help="Output machine-readable JSON metadata to stdout")
    p_synth.add_argument("-q", "--quiet", action="store_true", help="Suppress terminal banners and progress info")
    p_synth.add_argument("--pet", action="store_true", help="Spawn the floating desktop companion pet during synthesis")

    # 2. Subcommand: batch
    p_batch = subparsers.add_parser("batch", help="Batch synthesize voices for all face images in a directory")
    p_batch.add_argument("-d", "--dir", type=str, required=True, help="Input directory containing face images")
    p_batch.add_argument("-o", "--out", type=str, default="batch_output", help="Output directory for audio files")
    p_batch.add_argument("-t", "--text", type=str, default=None, help="Common text string for all subjects")
    p_batch.add_argument("--text-file", type=str, default=None, help="Path to text prompt file")
    p_batch.add_argument("-v", "--voices", type=int, default=1, help="Candidates per subject")
    p_batch.add_argument("--temp", type=float, default=1.0, help="CVAE diversity temperature")
    p_batch.add_argument("--lang", type=str, default="en", help="Language code")
    p_batch.add_argument("--accent", type=str, default=None, help="Accent")
    p_batch.add_argument("--emotion", type=str, default=None, help="Emotion")
    p_batch.add_argument("--no-watermark", action="store_true", help="Disable watermarking")
    p_batch.add_argument("--json", action="store_true", help="Output JSON results")
    p_batch.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")
    p_batch.add_argument("--pet", action="store_true", help="Spawn the floating desktop companion pet during batch processing")

    # 3. Subcommand: interactive
    subparsers.add_parser("interactive", help="Launch interactive step-by-step synthesis wizard")

    # 4. Subcommand: info (aliases: doctor)
    p_info = subparsers.add_parser("info", aliases=["doctor"], help="System hardware diagnostics and checkpoints")
    p_info.add_argument("--json", action="store_true", help="Output diagnostics as JSON")

    # 5. Subcommand: capabilities
    p_caps = subparsers.add_parser("capabilities", help="List supported accents, emotions, languages, and sliders")
    p_caps.add_argument("--json", action="store_true", help="Output capabilities as JSON")

    # 6. Subcommand: pet
    p_pet = subparsers.add_parser("pet", help="Launch the floating desktop companion pet or terminal card")
    p_pet.add_argument(
        "--state",
        choices=["idle", "look", "walk", "run", "jump", "think", "speak", "sleep", "alert"],
        default="idle",
        help="Initial animation state (default: idle)",
    )
    p_pet.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Companion window scale factor (default: 1.0, e.g. 0.8 for compact, 1.25 for large)",
    )
    p_pet.add_argument(
        "--speech",
        type=str,
        default="VOIX Biometric Acoustic Companion Ready!",
        help="Initial speech bubble message",
    )
    p_pet.add_argument(
        "--terminal",
        action="store_true",
        help="Render ASCII terminal pet card instead of floating desktop window",
    )
    p_pet.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress terminal banner",
    )

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
        print("  voix synth -f photo.jpg --accent british_rp --emotion happy")
        print("  voix batch -d faces_folder/ -o output_folder/")
        print("  voix interactive")
        print("  voix info")
        sys.exit(0)

    args = parser.parse_args()

    if args.command == "pet":
        sys.exit(handle_pet(args))

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
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
