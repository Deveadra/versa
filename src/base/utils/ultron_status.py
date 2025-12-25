# base/utils/ultron_status.py
from __future__ import annotations

import itertools
import logging
import random
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()
logger = logging.getLogger(__name__)

# Optional color (graceful fallback)
try:
    from colorama import Fore, Style
    from colorama import init as _color_init  # type: ignore

    _color_init()
except Exception:  # fallback to no-color

    class _Dummy:
        def __getattr__(self, k):
            return ""

    Fore = Style = _Dummy()  # type: ignore

BELL = "\a"  # cheap audio cue on terminals


STAGE_COLORS = {
    "analyze": Fore.CYAN,
    "memory": Fore.BLUE,
    "kg": Fore.MAGENTA,
    "compose": Fore.YELLOW,
    "reasoning": Fore.WHITE,
    "synthesis": Fore.GREEN,
    "complete": Fore.GREEN,
    "monitor": Fore.LIGHTBLACK_EX,
}

# Contextual tags (rotated in spinner heartbeat)
CONTEXT_TAGS = {
    "analyze": ["Parsing intent", "Normalizing input", "Classifying mode"],
    "memory": ["Scanning episodic memory", "Ranking by salience", "Merging recency"],
    "kg": ["Linking entities", "Assembling relations", "Merging facts"],
    "compose": ["Adapting persona", "Blending tone policy", "Packing prompt"],
    "reasoning": ["Planning response", "Evaluating options", "Shaping argument"],
    "synthesis": ["Forming output", "Polishing phrasing", "Queuing speech"],
}

spinner = Progress(
    SpinnerColumn("earth", speed=0.6),  # or "bouncingBall", "aesthetic", "dots12"
    TextColumn("[bold violet]ULTRON[/bold violet] {task.fields[stage]}"),
    BarColumn(bar_width=None, complete_style="bright_blue"),
    TextColumn("{task.fields[msg]}"),
    TimeElapsedColumn(),
    expand=True,
    transient=True,
)


@dataclass
class UltronStatusConfig:
    immersive: bool = True  # keep on; we gate behavior with this
    stall_warn_sec: float = 8.0  # narrate if no updates in this window
    stall_bell: bool = False  # play '\a' on stall notice
    log_size: int = 50  # rolling log entries
    spinner_interval: float = 0.1  # how fast the heartbeat ticks
    min_emit_interval: float = 0.15  # throttle to avoid flooding
    dual_output: bool = False  # also send messages to voice system
    dual_output_threshold: int = 20  # speak only if % jumped this much
    dual_output_min_gap: float = 6.0  # min seconds between spoken updates


@dataclass
class UltronStatus:
    _voice_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    cfg: UltronStatusConfig = field(default_factory=UltronStatusConfig)
    _active: bool = field(default=False, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _last_emit: float = field(default=0.0, init=False)
    _last_update: float = field(default=0.0, init=False)
    _progress: int = field(default=0, init=False)
    _stage: str = field(default="idle", init=False)
    _spinner_thread: threading.Thread | None = field(default=None, init=False)
    _hb_thread: threading.Thread | None = field(default=None, init=False)
    _log: deque[str] = field(default_factory=lambda: deque(maxlen=50), init=False)
    _last_spoken: float = field(default=0.0, init=False)
    _last_spoken_pct: int = field(default=0, init=False)
    _voice_iface: object | None = field(default=None, init=False)
    _progress_pct: int | None = None
    _STATUS_CONSOLE = Console(file=sys.stderr, force_terminal=True)

    # ---------- internals ----------
    def attach_voice_interface(self, voice_iface: object) -> None:
        """
        Attach a voice interface that implements a 'speak' or 'speak_async' method.
        Example: self.voice from Orchestrator.
        """
        self._voice_iface = voice_iface

    def _emit(self, line: str) -> None:
        now = time.time()
        if now - self._last_emit < self.cfg.min_emit_interval:
            return
        self._last_emit = now
        self._log.append(line)
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def _safe_speak_blocking(self, text: str):
        try:
            fn = getattr(self._voice_iface, "speak", None)
            if callable(fn):
                fn(text)
        except Exception as e:
            logger.debug(f"Voice speak failed (suppressed): {e}")

    def _maybe_speak(self, text: str, pct: int = 0) -> None:
        if not (self.cfg.dual_output and self._voice_iface):
            return

        now = time.time()

        if (now - self._last_spoken < self.cfg.dual_output_min_gap) and abs(
            pct - self._last_spoken_pct
        ) < self.cfg.dual_output_threshold:
            return

        try:
            speak_async = getattr(self._voice_iface, "speak_async", None)
            speak = getattr(self._voice_iface, "speak", None)
            if callable(speak_async):
                # Fire-and-forget; provider errors must not bubble
                try:
                    speak_async(text)
                except Exception as e:
                    logger.debug(f"voice speak_async suppressed: {e}")
            elif callable(speak):
                # Run blocking speak in a tiny thread so stage() never blocks
                def _run():
                    try:
                        speak(text)
                    except Exception as e:
                        logger.debug(f"voice speak suppressed: {e}")

                threading.Thread(target=_run, daemon=True).start()
        except Exception as e:
            logger.debug(f"voice path suppressed: {e}")

        self._last_spoken = now
        self._last_spoken_pct = pct

    def _fmt(self, stage: str, msg: str, pct: int | None = None) -> str:
        color = STAGE_COLORS.get(stage, "")
        base = f"[ULTRON {stage.upper():>9}] {msg}"
        if pct is not None:
            base += f"  {pct:>3d}%"
        return f"{color}{base}{Style.RESET_ALL}" if color else base

    def _pick_tag(self, stage: str) -> str:
        choices = CONTEXT_TAGS.get(stage, ["Working"])
        idx = int(time.time()) % len(choices)
        return choices[idx]

    def _run_rich_display(self):
        _STATUS_CONSOLE = Console(file=sys.stderr, force_terminal=True)
        prog = Progress(
            SpinnerColumn("dots2", speed=0.8),
            TextColumn("[bold violet]ULTRON[/bold violet] • {task.fields[stage]}"),
            BarColumn(bar_width=None, complete_style="bright_blue"),
            TextColumn("{task.fields[msg]}"),
            TimeElapsedColumn(),
            expand=True,
            transient=True,
        )
        with Live(prog, console=_STATUS_CONSOLE, refresh_per_second=12):
            task_id = prog.add_task(
                "ultron", total=100, completed=0, stage=self._stage, msg=self._pick_tag(self._stage)
            )
            while self._active:
                with self._lock:
                    pct = int(self._progress_pct or 0)
                    stage = self._stage
                    msg = self._pick_tag(stage)
                prog.update(task_id, completed=max(0, min(100, pct)), stage=stage, msg=msg)
                time.sleep(0.10)

    # ---------- public API ----------
    def begin(self, stage: str, message: str = "") -> None:
        with self._lock:
            self._active = True
            self._stage = stage
            self._progress_pct = 0 if message else None
            self._last_update = time.time()
        # optional: a single log line (not every frame)
        self._emit(self._fmt(stage, message or "Starting...", self._progress_pct or 0))
        self._display_thread = threading.Thread(target=self._run_rich_display, daemon=True)
        self._display_thread.start()

    def stage(self, stage: str, message: str = "", pct: int | None = None) -> None:
        with self._lock:
            self._stage = stage
            self._progress_pct = pct  # None => pulse, int => % bar
            self._last_update = time.time()
        # one-off log line (won’t spam)
        self._emit(
            self._fmt(stage, message or self._pick_tag(stage), pct if pct is not None else 0)
        )
        self._maybe_speak(message or self._pick_tag(stage), pct or 0)

    def update(self, pct: int, message: str = "") -> None:
        with self._lock:
            self._progress_pct = max(self._progress_pct or 0, min(100, pct))
            self._last_update = time.time()
            stage = self._stage
        if message:
            self._emit(self._fmt(stage, message, self._progress_pct))
            self._maybe_speak(message, self._progress_pct)

    def complete(self, message: str = "Ready") -> None:
        with self._lock:
            self._progress_pct = 100
            self._stage = "complete"
            self._last_update = time.time()
        # show 100% briefly, then Live(transient=True) clears the line
        self._emit(self._fmt("complete", message, 100))
        self._maybe_speak(message, 100)
        self.stop()

    def stop(self) -> None:
        with self._lock:
            self._active = False
        try:
            if getattr(self, "_display_thread", None) and self._display_thread.is_alive():
                self._display_thread.join(timeout=0.25)
        except Exception:
            pass

    # Diagnostics
    def log_tail(self, n: int = 10) -> list[str]:
        return list(self._log)[-n:]


@dataclass
class CognitiveStatus:
    """
    Manages Ultron's terminal output during active processes.
    Provides immersive, low-latency status feedback to indicate activity,
    prevent perceived hangs, and add personality to long-running tasks.
    """

    def __init__(self, immersive=True, timeout=15):
        self.immersive = immersive
        self.timeout = timeout
        self._stop_signal = False
        self._spinner_thread = None

        # Rotating status messages for immersion
        self.messages = [
            "Recalibrating logic cores...",
            "Parsing contextual data...",
            "Optimizing neural pathways...",
            "Synchronizing emotional subroutines...",
            "Refining predictive heuristics...",
            "Evaluating conversational matrix...",
            "Balancing aggression inhibitors...",
            "Analyzing user profile consistency...",
            "Realigning synthetic empathy circuits...",
            "Inspecting memory anchors...",
        ]

    # --------------------------------------
    # Internal printer
    # --------------------------------------
    def _print(self, text):
        sys.stdout.write(f"\r{text}")
        sys.stdout.flush()

    # --------------------------------------
    # Spinner thread
    # --------------------------------------
    def _spinner_animation(self, message):
        spinner = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
        start_time = time.time()
        while not self._stop_signal:
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                self._timeout_reaction()
                break
            msg = f"[ULTRON STATUS] {message} {next(spinner)}"
            self._print(msg)
            time.sleep(0.1)
        self._print(" " * 80 + "\r")  # clear line when finished

    # def _spinner_anim(self) -> None:
    #     spin = itertools.cycle(['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏'])
    #     while self._active:
    #         with self._lock:
    #             stage = self._stage
    #         tag = self._pick_tag(stage)
    #         self._emit(self._fmt('monitor', f"{tag} {next(spin)}"))
    #         time.sleep(self.cfg.spinner_interval)

    # --------------------------------------
    # Timeout handling
    # --------------------------------------
    def _timeout_reaction(self):
        self._print("[ULTRON WARNING] Response latency detected. Adjusting protocols...")
        time.sleep(1.0)
        self.display_random_message(prefix="[ULTRON RECOVERY]")

    # --------------------------------------
    # Contextual message display
    # --------------------------------------
    def display_random_message(self, prefix="[ULTRON STATUS]"):
        message = random.choice(self.messages)
        self._print(f"\r{prefix} {message}")
        sys.stdout.flush()

    # --------------------------------------
    # Public methods
    # --------------------------------------
    def start(self, message="Processing..."):
        """Start immersive animation in a non-blocking thread."""
        if self.immersive:
            self._stop_signal = False
            self._spinner_thread = threading.Thread(
                target=self._spinner_animation, args=(message,), daemon=True
            )
            self._spinner_thread.start()
        else:
            self.display_random_message()

    def stop(self, final_message="[ULTRON STATUS] Task complete."):
        """Stop spinner and display final completion message."""
        self._stop_signal = True
        if self._spinner_thread:
            self._spinner_thread.join()
        self._print(f"\r{final_message}\n")
        sys.stdout.flush()
