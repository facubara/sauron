"""Self-update from GitHub releases.

Flow (only for released, frozen builds):
1. A background thread polls the latest GitHub release every few hours.
2. If its version is newer, Sauron.exe is downloaded next to the running
   exe as Sauron.exe.new and verified (size, plus SHA-256 when GitHub
   provides a digest).
3. The main loop calls apply() when no alert is active. Windows allows
   renaming a running exe, so the swap is: Sauron.exe -> Sauron.exe.old,
   Sauron.exe.new -> Sauron.exe.
4. The app shuts down cleanly and relaunch() starts the new exe. The next
   start deletes Sauron.exe.old.

The startup shortcut keeps pointing at the same path, so Windows always
launches the newest version.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
import urllib.request

log = logging.getLogger(__name__)

REPO = "facubara/sauron"
# SAURON_UPDATE_URL lets an end-to-end test point a frozen build at a local
# fake release instead of GitHub.
LATEST_URL = os.environ.get(
    "SAURON_UPDATE_URL", f"https://api.github.com/repos/{REPO}/releases/latest")
ASSET_NAME = "Sauron.exe"
USER_AGENT = "Sauron-updater"
INITIAL_DELAY_SECONDS = float(os.environ.get("SAURON_UPDATE_DELAY", 60))
HTTP_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
_VERSION_RE = re.compile(r"^v?(\d+(?:\.\d+)*)$")


def parse_version(text: str) -> tuple[int, ...] | None:
    """'v0.3.12' -> (0, 3, 12). Anything else ('0.3.0-dev', '') -> None."""
    m = _VERSION_RE.match(str(text).strip())
    if not m:
        return None
    return tuple(int(p) for p in m.group(1).split("."))


def is_newer(remote: str, local: str) -> bool:
    r, loc = parse_version(remote), parse_version(local)
    if r is None or loc is None:
        return False
    width = max(len(r), len(loc))
    return r + (0,) * (width - len(r)) > loc + (0,) * (width - len(loc))


def pick_asset(release: dict, name: str = ASSET_NAME) -> dict | None:
    for asset in release.get("assets") or []:
        if asset.get("name") == name and asset.get("browser_download_url"):
            return asset
    return None


def expected_sha256(asset: dict) -> str | None:
    digest = asset.get("digest") or ""
    if digest.startswith("sha256:"):
        return digest.split(":", 1)[1].lower()
    return None


def updates_supported(version: str) -> tuple[bool, str]:
    """Only frozen release builds update themselves."""
    if not getattr(sys, "frozen", False):
        return False, "running from source"
    if parse_version(version) is None:
        return False, f"version {version!r} is a local build"
    return True, ""


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------
def new_path(exe_path: str) -> str:
    return exe_path + ".new"


def old_path(exe_path: str) -> str:
    return exe_path + ".old"


def cleanup_old(exe_path: str) -> None:
    """Delete leftovers of a previous update. Silently retried later if the
    previous process still has the old exe open."""
    for path in (old_path(exe_path),):
        try:
            os.remove(path)
            log.info("Removed %s", path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            log.debug("Could not remove %s yet: %s", path, exc)


def swap_in(exe_path: str) -> bool:
    """Replace exe_path with exe_path.new, keeping the running image as
    .old. Rolls back if the second rename fails."""
    new, old = new_path(exe_path), old_path(exe_path)
    if not os.path.exists(new):
        return False
    try:
        if os.path.exists(old):
            os.remove(old)
    except OSError as exc:
        log.warning("Update not applied, cannot remove %s: %s", old, exc)
        return False
    try:
        os.replace(exe_path, old)
    except OSError as exc:
        log.warning("Update not applied, cannot move running exe: %s", exc)
        return False
    try:
        os.replace(new, exe_path)
    except OSError as exc:
        log.error("Update swap failed, rolling back: %s", exc)
        try:
            os.replace(old, exe_path)
        except OSError:
            log.critical("Rollback failed; restore %s manually", old)
        return False
    return True


def relaunch(exe_path: str, extra_args: list[str] | None = None) -> None:
    """Start exe_path as an independent process.

    PYINSTALLER_RESET_ENVIRONMENT makes the child PyInstaller app unpack
    into its own temp dir instead of reusing ours, which is deleted as soon
    as this process exits.
    """
    env = dict(os.environ)
    env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.Popen([exe_path, *(extra_args or [])], env=env,
                     cwd=os.path.dirname(exe_path), close_fds=True,
                     creationflags=flags)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
def fetch_latest_release(url: str = LATEST_URL) -> dict:
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json", "User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return json.load(resp)


def download_asset(asset: dict, dest: str) -> None:
    """Download to dest, verifying size and SHA-256 (when known). On any
    mismatch dest is removed and ValueError raised."""
    req = urllib.request.Request(asset["browser_download_url"],
                                 headers={"User-Agent": USER_AGENT})
    sha = hashlib.sha256()
    size = 0
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp, \
                open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                sha.update(chunk)
                size += len(chunk)
        verify_download(asset, size, sha.hexdigest())
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def verify_download(asset: dict, size: int, sha_hex: str) -> None:
    expected_size = asset.get("size")
    if expected_size is not None and size != int(expected_size):
        raise ValueError(f"size mismatch: got {size}, expected {expected_size}")
    expected = expected_sha256(asset)
    if expected is not None and sha_hex.lower() != expected:
        raise ValueError("SHA-256 mismatch")


# ---------------------------------------------------------------------------
# Background checker
# ---------------------------------------------------------------------------
class Updater:
    """Downloads newer releases in the background. Thread-safe surface:
    ``ready`` (bool), ``new_version`` (str) and apply()."""

    def __init__(self, current_version: str, exe_path: str, check_hours: float,
                 fetch=fetch_latest_release, download=download_asset):
        self.current_version = current_version
        self.exe_path = exe_path
        self.check_seconds = max(60.0, check_hours * 3600)
        self._fetch = fetch
        self._download = download
        self._ready = threading.Event()
        self._stop = threading.Event()
        self.new_version = ""
        self._thread: threading.Thread | None = None

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="sauron-updater")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        if self._stop.wait(INITIAL_DELAY_SECONDS):
            return
        while not self._stop.is_set() and not self.ready:
            cleanup_old(self.exe_path)
            try:
                self.check_once()
            except Exception as exc:  # network down, rate limit, etc.
                log.info("Update check failed: %s", exc)
            if self._stop.wait(self.check_seconds):
                return

    def check_once(self) -> bool:
        """One poll. Returns True when a verified update is staged."""
        release = self._fetch()
        tag = str(release.get("tag_name", ""))
        if not is_newer(tag, self.current_version):
            log.debug("Up to date (%s, latest %s)", self.current_version, tag)
            return False
        asset = pick_asset(release)
        if asset is None:
            log.warning("Release %s has no %s asset", tag, ASSET_NAME)
            return False
        log.info("Downloading update %s -> %s", self.current_version, tag)
        self._download(asset, new_path(self.exe_path))
        self.new_version = tag.lstrip("v")
        self._ready.set()
        log.info("Update %s downloaded and verified", self.new_version)
        return True

    def apply(self) -> bool:
        """Swap the staged exe in. On failure, stop offering this update."""
        if not self.ready:
            return False
        if swap_in(self.exe_path):
            log.info("Installed %s; restarting", self.new_version)
            return True
        self._ready.clear()
        self.stop()
        return False
