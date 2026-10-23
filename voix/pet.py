"""VOIX Floating Desktop Companion Pet.

An interactive, animated biometric acoustic companion for VOIX.
Inspired by Codex Pets, this companion floats seamlessly on the user's desktop,
rendering transparent, borderless pixel animations directly above all windows and CLI terminals.
Supports real-time state reactions (idle, look, walk, run, jump, think, speak, sleep, alert),
speech bubbles, drag-and-drop positioning, and live CLI integration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import tkinter as tk
    from tkinter import Menu
    from PIL import Image, ImageTk
    HAS_GUI = True
except ImportError:
    HAS_GUI = False


# Default asset resolution paths
DEFAULT_ASSET_PATHS = [
    Path("D:/voix/voix/assets/pets/voix-pet"),
    Path(__file__).resolve().parent / "assets" / "pets" / "voix-pet",
    Path.home() / ".codex" / "pets" / "voix-pet",
    Path("c:/Users/omen/OneDrive/Desktop/voix/voix/assets/pets/voix-pet"),
]


def resolve_pet_assets() -> Tuple[Path, Path]:
    """Locate the pet.json metadata and spritesheet image files."""
    for asset_dir in DEFAULT_ASSET_PATHS:
        json_path = asset_dir / "pet.json"
        sheet_path = asset_dir / "spritesheet.png"
        webp_path = asset_dir / "spritesheet.webp"

        target_sheet = sheet_path if sheet_path.exists() else webp_path
        if json_path.exists() and target_sheet.exists():
            return json_path, target_sheet

    raise FileNotFoundError(
        "Could not locate VOIX pet assets. Looked in: "
        + ", ".join(str(p) for p in DEFAULT_ASSET_PATHS)
    )


class VoixPetWindow:
    """Floating transparent desktop companion window."""

    CHROMA_KEY = "#010101"  # Almost black color used for transparency cutout

    def __init__(
        self,
        scale: float = 1.0,
        initial_state: str = "idle",
        initial_speech: Optional[str] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
    ) -> None:
        if not HAS_GUI:
            raise RuntimeError("Tkinter and Pillow are required for VoixPetWindow.")

        self.scale = max(0.4, min(2.5, scale))
        self.current_state = initial_state
        self.speech_text: Optional[str] = initial_speech
        self.speech_expire_time: float = time.time() + 6.0 if initial_speech else 0.0

        # Load metadata and sprites
        self.json_path, self.sheet_path = resolve_pet_assets()
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.grid_cols = self.metadata["grid"]["columns"]
        self.grid_rows = self.metadata["grid"]["rows"]
        self.frame_w = self.metadata["grid"]["frameWidth"]
        self.frame_h = self.metadata["grid"]["frameHeight"]
        self.states_config = self.metadata.get("states", {})

        # Window dimensions based on scale
        self.win_w = int(self.frame_w * self.scale)
        self.win_h = int(self.frame_h * self.scale)

        # Pre-process sprite frames for fast playback
        self.frames_by_state: Dict[str, List[ImageTk.PhotoImage]] = {}
        self._raw_sheet = Image.open(self.sheet_path).convert("RGBA")
        self._load_and_process_frames()

        # Initialize Tkinter transparent root
        self.root = tk.Tk()
        self.root.title("VOIX Pet Companion")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.wm_attributes("-transparentcolor", self.CHROMA_KEY)
        self.root.config(bg=self.CHROMA_KEY)

        # Position on screen
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        pos_x = x if x is not None else max(50, sw - self.win_w - 60)
        pos_y = y if y is not None else max(50, sh - self.win_h - 100)
        self.root.geometry(f"{self.win_w}x{self.win_h}+{pos_x}+{pos_y}")

        # Main display canvas
        self.canvas = tk.Canvas(
            self.root,
            width=self.win_w,
            height=self.win_h,
            bg=self.CHROMA_KEY,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Internal animation tracking
        self.frame_idx = 0
        self._anim_timer = None
        self._running = True

        # Mouse drag interactions
        self._drag_start_x = 0
        self._drag_start_y = 0
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_motion)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)
        self.canvas.bind("<Button-3>", self._on_right_click)

        # Context Menu
        self._create_context_menu()

        # Render first frame
        self._render_current_frame()
        self._schedule_next_frame()

    def _load_and_process_frames(self) -> None:
        """Extract and prepare transparent PhotoImage frames for all states."""
        for state_name, s_info in self.states_config.items():
            row_idx = s_info.get("row", 0)
            num_frames = s_info.get("frames", self.grid_cols)
            state_frames = []

            for c in range(num_frames):
                crop_box = (
                    c * self.frame_w,
                    row_idx * self.frame_h,
                    (c + 1) * self.frame_w,
                    (row_idx + 1) * self.frame_h,
                )
                raw_frame = self._raw_sheet.crop(crop_box)

                if self.scale != 1.0:
                    raw_frame = raw_frame.resize(
                        (self.win_w, self.win_h),
                        Image.Resampling.NEAREST if self.scale > 1.0 else Image.Resampling.BILINEAR,
                    )

                # Composite with Chroma key for crisp cutout on Windows
                arr = raw_frame.load()
                w, h = raw_frame.size
                rgb_bg = Image.new("RGB", (w, h), (1, 1, 1))

                # Mask alpha < 50 to chroma key
                for px in range(w):
                    for py in range(h):
                        r, g, b, a = arr[px, py]
                        if a >= 50:
                            rgb_bg.putpixel((px, py), (r, g, b))

                photo = ImageTk.PhotoImage(rgb_bg)
                state_frames.append(photo)

            self.frames_by_state[state_name] = state_frames

    def _create_context_menu(self) -> None:
        """Build the right-click popup menu."""
        self.menu = Menu(self.root, tearoff=0, bg="#1e1e2e", fg="#cdd6f4", activebackground="#89b4fa")
        self.menu.add_command(label="VOIX Acoustic Companion", state="disabled")
        self.menu.add_separator()

        # Animation state submenu
        state_menu = Menu(self.menu, tearoff=0, bg="#1e1e2e", fg="#cdd6f4", activebackground="#89b4fa")
        for state_name in ["idle", "look", "walk", "run", "jump", "think", "speak", "sleep", "alert"]:
            state_menu.add_command(
                label=f"Play: {state_name.capitalize()}",
                command=lambda s=state_name: self.set_state(s),
            )
        self.menu.add_cascade(label="Animation State", menu=state_menu)

        # Scale submenu
        size_menu = Menu(self.menu, tearoff=0, bg="#1e1e2e", fg="#cdd6f4", activebackground="#89b4fa")
        size_menu.add_command(label="Compact (75%)", command=lambda: self.set_scale(0.75))
        size_menu.add_command(label="Normal (100%)", command=lambda: self.set_scale(1.0))
        size_menu.add_command(label="Large (125%)", command=lambda: self.set_scale(1.25))
        self.menu.add_cascade(label="Scale Size", menu=size_menu)

        self.menu.add_command(label="Say Hello", command=lambda: self.say("Meow! VOIX Voice Ready.", 4.0))
        self.menu.add_separator()
        self.menu.add_command(label="Close Companion", command=self.close)

    def _on_drag_start(self, event: tk.Event) -> None:
        self._drag_start_x = event.x
        self._drag_start_y = event.y

    def _on_drag_motion(self, event: tk.Event) -> None:
        x = self.root.winfo_x() + (event.x - self._drag_start_x)
        y = self.root.winfo_y() + (event.y - self._drag_start_y)
        self.root.geometry(f"+{x}+{y}")

    def _on_double_click(self, event: tk.Event) -> None:
        """Cycle through animation states on double click."""
        states = list(self.states_config.keys())
        if not states:
            states = ["idle", "look", "walk", "run", "jump", "think", "speak", "sleep", "alert"]
        curr_idx = states.index(self.current_state) if self.current_state in states else 0
        next_state = states[(curr_idx + 1) % len(states)]
        self.set_state(next_state)
        self.say(f"{next_state.capitalize()} mode", 2.0)

    def _on_right_click(self, event: tk.Event) -> None:
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()

    def set_state(self, state: str) -> None:
        """Switch current animation state cleanly."""
        if state in self.frames_by_state:
            self.current_state = state
            self.frame_idx = 0
            self._render_current_frame()

    def set_scale(self, scale: float) -> None:
        """Re-render companion at a new scale."""
        self.scale = max(0.4, min(2.5, scale))
        self.win_w = int(self.frame_w * self.scale)
        self.win_h = int(self.frame_h * self.scale)
        x = self.root.winfo_x()
        y = self.root.winfo_y()
        self.root.geometry(f"{self.win_w}x{self.win_h}+{x}+{y}")
        self.canvas.config(width=self.win_w, height=self.win_h)
        self._load_and_process_frames()
        self._render_current_frame()

    def say(self, message: str, duration_sec: float = 4.0) -> None:
        """Display a speech bubble above the companion."""
        self.speech_text = message
        self.speech_expire_time = time.time() + duration_sec
        self._render_current_frame()

    def _render_current_frame(self) -> None:
        """Draw active frame and optional speech bubble on transparent canvas."""
        self.canvas.delete("all")
        frames = self.frames_by_state.get(self.current_state, [])
        if not frames:
            return

        photo = frames[self.frame_idx % len(frames)]
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)

        # Check speech bubble expiration
        if self.speech_text and time.time() < self.speech_expire_time:
            bubble_text = self.speech_text
            bx = self.win_w // 2
            by = max(18, int(26 * self.scale))
            # Text shadow and speech text
            self.canvas.create_text(
                bx + 1, by + 1,
                text=bubble_text,
                font=("Segoe UI", max(8, int(9 * self.scale)), "bold"),
                fill="#11111b",
            )
            self.canvas.create_text(
                bx, by,
                text=bubble_text,
                font=("Segoe UI", max(8, int(9 * self.scale)), "bold"),
                fill="#f9e2af",
            )

    def _schedule_next_frame(self) -> None:
        """Schedule next animation tick according to state duration."""
        if not self._running:
            return

        s_info = self.states_config.get(self.current_state, {})
        duration_ms = s_info.get("duration", 800)
        frames = self.frames_by_state.get(self.current_state, [])
        num_frames = max(1, len(frames))
        delay_ms = max(20, int(duration_ms / num_frames))

        def _step():
            if not self._running:
                return
            frames = self.frames_by_state.get(self.current_state, [])
            if frames:
                self.frame_idx = (self.frame_idx + 1) % len(frames)
                self._render_current_frame()
            self._schedule_next_frame()

        self._anim_timer = self.root.after(delay_ms, _step)

    def run(self) -> None:
        """Start the Tkinter event loop (blocking)."""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.close()

    def close(self) -> None:
        """Cleanly destroy window and stop animation."""
        self._running = False
        if self._anim_timer:
            try:
                self.root.after_cancel(self._anim_timer)
            except Exception:
                pass
        try:
            self.root.destroy()
        except Exception:
            pass


class PetCompanionController:
    """Non-blocking background companion controller for CLI integration."""

    def __init__(self) -> None:
        self.process: Optional[threading.Thread] = None
        self.window: Optional[VoixPetWindow] = None
        self.stop_event = threading.Event()

    def launch(
        self,
        initial_state: str = "idle",
        initial_speech: Optional[str] = "VOIX Ready!",
        scale: float = 0.9,
    ) -> None:
        """Launch the floating companion in a background daemon thread."""
        if not HAS_GUI:
            return

        def _worker():
            try:
                self.window = VoixPetWindow(
                    scale=scale,
                    initial_state=initial_state,
                    initial_speech=initial_speech,
                )
                self.window.run()
            except Exception:
                pass

        self.process = threading.Thread(target=_worker, daemon=True)
        self.process.start()

    def set_state(self, state: str) -> None:
        """Update companion state if running."""
        if self.window and self.window._running:
            try:
                self.window.root.after(0, lambda: self.window.set_state(state))
            except Exception:
                pass

    def say(self, message: str, duration_sec: float = 3.5) -> None:
        """Display speech bubble if running."""
        if self.window and self.window._running:
            try:
                self.window.root.after(0, lambda: self.window.say(message, duration_sec))
            except Exception:
                pass

    def close(self) -> None:
        """Close companion window."""
        if self.window and self.window._running:
            try:
                self.window.root.after(0, self.window.close)
            except Exception:
                pass


def main() -> int:
    """Standalone CLI entry point for voix pet."""
    parser = argparse.ArgumentParser(
        description="VOIX Floating Desktop Companion Pet (Codex-Compatible)"
    )
    parser.add_argument(
        "--state",
        choices=["idle", "look", "walk", "run", "jump", "think", "speak", "sleep", "alert"],
        default="idle",
        help="Initial animation state to display (default: idle)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Companion scale factor (default: 1.0, e.g. 0.8 for compact, 1.25 for large)",
    )
    parser.add_argument(
        "--speech",
        type=str,
        default="VOIX Biometric Voice Synthesizer Ready!",
        help="Initial speech bubble text",
    )
    parser.add_argument(
        "--terminal",
        action="store_true",
        help="Render ASCII terminal pet card instead of floating desktop window",
    )
    args = parser.parse_args()

    if args.terminal or not HAS_GUI:
        render_terminal_pet(args.state, args.speech)
        return 0

    print("Launching VOIX Floating Companion on desktop...")
    print("  - Drag with Left Mouse Button to reposition anywhere on screen.")
    print("  - Double-click to cycle animation states.")
    print("  - Right-click for options and states.")
    print("Press Ctrl+C in terminal or Right-click -> Close to exit.")

    pet = VoixPetWindow(
        scale=args.scale,
        initial_state=args.state,
        initial_speech=args.speech,
    )
    pet.run()
    return 0


def render_terminal_pet(state: str = "idle", speech: Optional[str] = None) -> None:
    """Render terminal companion banner for headless or pure text consoles."""
    speech_line = f'"{speech}"' if speech else '"Biometric voice synthesis ready."'
    pet_art = f"""
      /\\_/\\     VOIX Companion ({state.upper()})
     ( o.o )    {speech_line}
      > ^ <     Headphones: ACTIVE | Equalizer: SYNCED
    """
    try:
        from rich.console import Console
        from rich.panel import Panel
        c = Console()
        c.print(Panel(pet_art.strip(), title="VOIX Companion", border_style="cyan"))
    except ImportError:
        print(f"  /\\_/\\    VOIX Companion ({state.upper()}): {speech_line}")
        print(" ( o.o )")
        print("  > ^ <")


if __name__ == "__main__":
    sys.exit(main())
