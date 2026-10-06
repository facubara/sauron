"""Alert outputs: looping warning sound and the fullscreen popup."""

from __future__ import annotations

import logging
import random
import threading
import tkinter as tk
import traceback

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
class AudioPlayer:
    """Thin wrapper over pygame.mixer.music. Every method is safe to call
    even if the mixer failed to initialise (no sound device, etc.)."""

    def __init__(self, sound_files: list[str]):
        self.sound_files = list(sound_files)
        self.ok = False
        try:
            import pygame
            pygame.mixer.init()
            self._music = pygame.mixer.music
            self.ok = True
        except Exception as exc:  # pragma: no cover - depends on hardware
            log.warning("Audio disabled: %s", exc)
            self._music = None

    def play_loop(self) -> None:
        """Start looping a random warning sound until stop()."""
        if not self.ok or not self.sound_files:
            return
        try:
            self._music.load(random.choice(self.sound_files))
            self._music.play(loops=-1)
        except Exception as exc:
            log.warning("Could not play warning sound: %s", exc)

    def stop(self) -> None:
        if not self.ok:
            return
        try:
            self._music.stop()
        except Exception:
            pass

    def set_volume(self, volume: float) -> None:
        if not self.ok:
            return
        try:
            self._music.set_volume(max(0.0, min(1.0, volume)))
        except Exception:
            pass

    @property
    def playing(self) -> bool:
        if not self.ok:
            return False
        try:
            return bool(self._music.get_busy())
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Popup
# ---------------------------------------------------------------------------
class WarningPopup:
    """Fullscreen alert that stays up until the hand leaves the mouth.

    A single Tk instance lives in one persistent background thread for the
    whole app lifetime. All Tk calls happen inside that thread via a 100ms
    poll loop - show()/hide() from other threads only flip plain flags.
    (Creating/destroying Tk roots across threads is a known crash source.)
    """

    BG = "#1a0000"

    def __init__(self, icon_path: str | None = None, version: str = ""):
        self._icon_path = icon_path
        self._version = version
        self._want_visible = False
        self._shown = False
        self._root = None
        self._count_label = None
        self._phrase_label = None
        self.count = 0
        self.phrase = ""
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="sauron-popup")
        self._thread.start()

    def show(self, count: int, phrase: str = "") -> None:
        self.count = count
        self.phrase = phrase
        self._want_visible = True

    def hide(self) -> None:
        self._want_visible = False

    @property
    def visible(self) -> bool:
        return self._shown

    def _run(self) -> None:
        try:
            root = tk.Tk()
            self._root = root
            root.withdraw()
            # NOTE: -fullscreen and overrideredirect are mutually exclusive in
            # Tk (setting fullscreen with override-redirect raises TclError).
            # -fullscreen already hides the title bar, so use it alone.
            root.attributes("-fullscreen", True)
            root.attributes("-topmost", True)
            root.attributes("-alpha", 0.92)
            root.configure(bg=self.BG)
            if self._icon_path:
                try:
                    root.iconbitmap(self._icon_path)
                except Exception:
                    pass

            wrap = int(root.winfo_screenwidth() * 0.8)
            frame = tk.Frame(root, bg=self.BG)
            frame.place(relx=0.5, rely=0.5, anchor="center")

            tk.Label(frame, text="STOP BITING YOUR NAILS!",
                     font=("Segoe UI", 54, "bold"), fg="#ff3333",
                     bg=self.BG).pack(pady=(0, 16))
            tk.Label(frame, text="Move your hand away from your mouth"
                     " to dismiss this warning.",
                     font=("Segoe UI", 22), fg="#ff9999",
                     bg=self.BG).pack(pady=(0, 24))
            self._phrase_label = tk.Label(frame, text="",
                                          font=("Segoe UI", 20, "italic"),
                                          fg="#d9a66b", bg=self.BG,
                                          wraplength=wrap, justify="center")
            self._phrase_label.pack(pady=(0, 24))
            self._count_label = tk.Label(frame, text="",
                                         font=("Segoe UI", 16), fg="#aa6666",
                                         bg=self.BG)
            self._count_label.pack()

            if self._version:
                tk.Label(root, text=f"Sauron v{self._version}",
                         font=("Segoe UI", 10), fg="#663333",
                         bg=self.BG).place(relx=1.0, rely=1.0, anchor="se",
                                           x=-16, y=-12)

            self._poll()
            root.mainloop()
        except Exception:
            log.error("Popup thread crashed:\n%s", traceback.format_exc())

    def _poll(self) -> None:
        root = self._root
        try:
            if self._want_visible and not self._shown:
                self._shown = True
                if self._count_label is not None:
                    self._count_label.config(text=f"Bites today: {self.count}")
                if self._phrase_label is not None:
                    self._phrase_label.config(text=self.phrase)
                root.deiconify()
                root.attributes("-topmost", True)
                root.focus_force()
            elif not self._want_visible and self._shown:
                self._shown = False
                root.withdraw()
            elif self._shown:
                # Re-assert topmost in case something stole it
                root.attributes("-topmost", True)
        except Exception:
            pass
        try:
            root.after(100, self._poll)
        except Exception:
            pass
