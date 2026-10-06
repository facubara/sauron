"""Paths, user settings, phrase list and daily stats for Sauron.

This module is deliberately free of cv2 / mediapipe / pygame imports so it
can be unit-tested without a camera or a display.

Everything the user may want to tweak lives in %APPDATA%\\Sauron:

    config.json   detection thresholds, camera, volume, snooze length
    phrases.txt   optional override of the bundled popup phrases
    stats.json    per-day bite counts
    sauron.log    rotating diagnostic log (the .exe has no console)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, fields
from datetime import date, timedelta

APP_NAME = "Sauron"
WINDOW_TITLE = "Sauron - Nail Bite Detector"

# PyInstaller bundles data files into sys._MEIPASS; fall back to script dir.
SCRIPT_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))

# Writable user-data dir (SCRIPT_DIR is a read-only temp dir in exe builds).
USER_DATA_DIR = os.path.join(
    os.environ.get("APPDATA") or os.path.expanduser("~"), APP_NAME)
CONFIG_PATH = os.path.join(USER_DATA_DIR, "config.json")
STATS_PATH = os.path.join(USER_DATA_DIR, "stats.json")
LOG_PATH = os.path.join(USER_DATA_DIR, "sauron.log")
PHRASES_FILENAME = "phrases.txt"

log = logging.getLogger(__name__)


def asset_path(name: str) -> str:
    """Path of a file bundled next to the script / inside the exe."""
    return os.path.join(SCRIPT_DIR, name)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
@dataclass
class Settings:
    """All tunables. Defaults favour precision (few false alarms)."""

    # Camera
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480

    # Audio
    volume: float = 0.5
    start_muted: bool = False

    # Fingertip must be within this fraction of the face width from the
    # mouth centre (never less than fingertip_thr_min_px pixels).
    fingertip_thr_face_frac: float = 0.32
    fingertip_thr_min_px: float = 25.0

    # Sustained proximity required before an alert fires (seconds).
    bite_dwell_seconds: float = 1.5
    # Hand must stay away this long for the alert to clear (seconds).
    clear_seconds: float = 1.0
    # Alert is never shorter than this, even if the hand leaves at once.
    min_alert_seconds: float = 2.0

    # Only the user is tracked: the largest face, and only if it spans at
    # least this fraction of the frame width. Background people are smaller.
    min_face_width_frac: float = 0.12
    # Hand size (wrist -> middle knuckle) / face width must be in this band;
    # rejects hands that belong to someone at a different distance.
    hand_face_ratio_min: float = 0.30
    hand_face_ratio_max: float = 1.80

    # Eating suppression: lip gap / mouth width above this = mouth wide open,
    # and detection stays suppressed this long afterwards.
    mouth_open_ratio: float = 0.40
    eating_suppress_seconds: float = 4.0

    # If the face tracker drops out briefly (hand covering the mouth), keep
    # using the last known mouth position for this long.
    face_memory_seconds: float = 1.0

    # Length of a manual snooze (the 's' key).
    snooze_minutes: float = 5.0

    # Show a random phrase from phrases.txt on the warning popup.
    show_phrases: bool = True

    # Logging level for sauron.log: DEBUG, INFO, WARNING, ERROR.
    log_level: str = "INFO"

    def validate(self) -> list[str]:
        """Clamp nonsensical values in place and return a list of warnings."""
        warnings: list[str] = []
        defaults = Settings()

        for name in ("bite_dwell_seconds", "clear_seconds", "min_alert_seconds",
                     "eating_suppress_seconds", "snooze_minutes",
                     "fingertip_thr_face_frac", "fingertip_thr_min_px",
                     "camera_width", "camera_height"):
            if getattr(self, name) <= 0:
                default = getattr(defaults, name)
                warnings.append(f"{name} must be > 0; using {default}")
                setattr(self, name, default)
        if not 0.0 <= self.volume <= 1.0:
            warnings.append("volume must be between 0 and 1; clamping")
            self.volume = min(1.0, max(0.0, self.volume))
        if self.face_memory_seconds < 0:
            warnings.append("face_memory_seconds must be >= 0; using 0")
            self.face_memory_seconds = 0.0
        if self.hand_face_ratio_min > self.hand_face_ratio_max:
            warnings.append("hand_face_ratio_min > max; swapping")
            self.hand_face_ratio_min, self.hand_face_ratio_max = (
                self.hand_face_ratio_max, self.hand_face_ratio_min)
        if self.camera_index < 0:
            warnings.append("camera_index must be >= 0; using 0")
            self.camera_index = 0
        level = str(self.log_level).upper()
        if level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            warnings.append(f"unknown log_level {self.log_level!r}; using INFO")
            level = "INFO"
        self.log_level = level
        return warnings


def _coerce(name: str, value, default):
    """Coerce a JSON value to the type of the default, or raise ValueError."""
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        raise ValueError(f"{name} must be true or false")
    if isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
        return int(value)
    if isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
        return float(value)
    return str(value)


def settings_from_dict(data: dict) -> tuple[Settings, list[str]]:
    """Build Settings from a parsed JSON object. Bad keys are skipped with a
    warning rather than aborting, so a typo never leaves the user alertless."""
    settings = Settings()
    warnings: list[str] = []
    known = {f.name: getattr(settings, f.name) for f in fields(Settings)}
    for key, value in data.items():
        if key not in known:
            warnings.append(f"unknown setting {key!r} ignored")
            continue
        try:
            setattr(settings, key, _coerce(key, value, known[key]))
        except ValueError as exc:
            warnings.append(f"{exc}; using default {known[key]!r}")
    warnings.extend(settings.validate())
    return settings, warnings


def load_settings(path: str = CONFIG_PATH) -> tuple[Settings, list[str]]:
    """Load config.json. A missing file is created with the defaults so the
    user has something to edit; a broken file falls back to defaults."""
    if not os.path.exists(path):
        try:
            write_settings(Settings(), path)
            log.info("Wrote default config to %s", path)
        except OSError as exc:
            log.warning("Could not write default config: %s", exc)
        return Settings(), []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        return Settings(), [f"could not read {path}: {exc}; using defaults"]
    if not isinstance(data, dict):
        return Settings(), [f"{path} must contain a JSON object; using defaults"]
    return settings_from_dict(data)


def write_settings(settings: Settings, path: str = CONFIG_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(settings), f, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# Phrases shown on the popup
# ---------------------------------------------------------------------------
DEFAULT_PHRASES = [
    "The Eye of Sauron sees every finger that strays.",
    "The hand that bites is the hand that loses.",
    "You shall not chew.",
    "One does not simply gnaw into Mordor.",
    "Hands in your lap, traveler. The road is long.",
]


def parse_phrases(text: str) -> list[str]:
    """One phrase per line; blank lines and '#' comments are ignored."""
    phrases = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            phrases.append(line)
    return phrases


def load_phrases(user_dir: str = USER_DATA_DIR,
                 bundled_dir: str = SCRIPT_DIR) -> list[str]:
    """User override in %APPDATA%\\Sauron wins, then the bundled file, then
    a small built-in list."""
    for candidate in (os.path.join(user_dir, PHRASES_FILENAME),
                      os.path.join(bundled_dir, PHRASES_FILENAME)):
        try:
            with open(candidate, encoding="utf-8") as f:
                phrases = parse_phrases(f.read())
        except OSError:
            continue
        if phrases:
            log.info("Loaded %d phrases from %s", len(phrases), candidate)
            return phrases
    return list(DEFAULT_PHRASES)


# ---------------------------------------------------------------------------
# Daily stats
# ---------------------------------------------------------------------------
STATS_VERSION = 2


def _today() -> str:
    return date.today().isoformat()


def load_stats(path: str = STATS_PATH) -> dict:
    """Return {"version": 2, "days": {"YYYY-MM-DD": count, ...}}.

    Transparently upgrades the v1 format ({"date": ..., "count": ...}) and
    tolerates a missing or corrupt file.
    """
    empty = {"version": STATS_VERSION, "days": {}}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return empty
    if not isinstance(data, dict):
        return empty
    if isinstance(data.get("days"), dict):
        days = {}
        for day, count in data["days"].items():
            try:
                days[str(day)] = max(0, int(count))
            except (TypeError, ValueError):
                continue
        return {"version": STATS_VERSION, "days": days}
    if "date" in data:  # v1 format
        try:
            count = max(0, int(data.get("count", 0)))
        except (TypeError, ValueError):
            return empty
        return {"version": STATS_VERSION, "days": {str(data["date"]): count}}
    return empty


def save_stats(stats: dict, path: str = STATS_PATH) -> None:
    """Atomic write (tmp file + rename) so a crash never corrupts the file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except OSError as exc:
        log.warning("Could not save stats: %s", exc)


def today_count(stats: dict, today: str | None = None) -> int:
    return int(stats.get("days", {}).get(today or _today(), 0))


def record_bite(stats: dict, today: str | None = None) -> int:
    """Increment today's count in place and return the new value."""
    today = today or _today()
    days = stats.setdefault("days", {})
    days[today] = int(days.get(today, 0)) + 1
    return days[today]


def clean_streak(stats: dict, today: str | None = None) -> int:
    """Consecutive bite-free days ending yesterday, counting back to the
    earliest recorded day. 0 if yesterday had a bite or there is no history."""
    days = stats.get("days", {})
    if not days:
        return 0
    today_d = date.fromisoformat(today or _today())
    earliest = min(date.fromisoformat(d) for d in days)
    streak = 0
    cursor = today_d - timedelta(days=1)
    while cursor >= earliest:
        if days.get(cursor.isoformat(), 0) > 0:
            break
        streak += 1
        cursor -= timedelta(days=1)
    return streak
