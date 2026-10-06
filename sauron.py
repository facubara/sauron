"""
Sauron - Nail Biting Detection System
Uses webcam + MediaPipe (Hand + Face landmarkers) to detect nail biting,
then shows a fullscreen popup and plays a warning sound until you stop.

Module map:
    config.py    settings file, phrases, daily stats (pure Python)
    detector.py  face/hand geometry and the dwell state machine (pure Python)
    alerts.py    looping sound + fullscreen Tk popup
    hud.py       OpenCV overlay, clickable controls, keyboard hints
    sauron.py    this file: camera loop and glue

Hotkeys in the webcam window:
    q / Esc  quit          m  mute/unmute       c  camera on/off
    s        snooze/unsnooze (length in config.json)   - / +  volume
"""

from __future__ import annotations

import ctypes
import logging
import logging.handlers
import os
import random
import sys
import time
import traceback
from datetime import date

import cv2
import numpy as np

import hud
from alerts import AudioPlayer, WarningPopup
from config import (
    APP_NAME,
    LOG_PATH,
    USER_DATA_DIR,
    WINDOW_TITLE,
    Settings,
    asset_path,
    clean_streak,
    load_phrases,
    load_settings,
    load_stats,
    record_bite,
    save_stats,
    today_count,
)
from detector import (
    HAND_FINGERTIPS,
    BiteStateMachine,
    FaceMemory,
    fingertip_proximity,
    fingertip_threshold,
    select_user_face,
)

log = logging.getLogger("sauron")

ICON_PATH = asset_path("sauron-icon.ico")
WARNING_SOUNDS = [asset_path("isengard.mp3"), asset_path("sauron-sound.mp3")]
MODEL_FILES = {
    "hand": asset_path("hand_landmarker.task"),
    "face": asset_path("face_landmarker.task"),
}
MUTEX_NAME = "Local\\SauronNailBiteDetector"
IS_WINDOWS = sys.platform == "win32"


# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------
def setup_logging(level: str = "INFO") -> None:
    """Rotating file log in %APPDATA%\\Sauron plus stderr when there is one
    (the windowed .exe has no console, so the file is the only trace)."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    try:
        os.makedirs(USER_DATA_DIR, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            LOG_PATH, maxBytes=512 * 1024, backupCount=2, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        pass
    if sys.stderr is not None:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
        root.addHandler(sh)


def fatal_dialog(message: str) -> None:
    """Show a native error box (the exe has no console to print to)."""
    log.error(message)
    if IS_WINDOWS:
        try:
            MB_ICONERROR = 0x10
            ctypes.windll.user32.MessageBoxW(0, message, APP_NAME, MB_ICONERROR)
            return
        except Exception:
            pass
    print(f"ERROR: {message}", file=sys.stderr)


def acquire_single_instance() -> bool:
    """Return False if another Sauron is already running (two instances
    would fight over the webcam). The mutex is released when the process
    exits."""
    if not IS_WINDOWS:
        return True
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.CreateMutexW(None, False, MUTEX_NAME)
        ERROR_ALREADY_EXISTS = 183
        if not handle:
            return True
        if kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
            return False
        acquire_single_instance.handle = handle  # keep it alive
        return True
    except Exception:
        return True


def set_window_icon(title: str, icon_path: str) -> None:
    """OpenCV windows have no icon API; set it through Win32."""
    if not IS_WINDOWS:
        return
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, title)
        if not hwnd:
            return
        IMAGE_ICON, LR_LOADFROMFILE, LR_DEFAULTSIZE = 1, 0x0010, 0x0040
        icon = user32.LoadImageW(0, icon_path, IMAGE_ICON, 0, 0,
                                 LR_LOADFROMFILE | LR_DEFAULTSIZE)
        if icon:
            WM_SETICON, ICON_SMALL, ICON_BIG = 0x0080, 0, 1
            user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, icon)
            user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, icon)
    except Exception:
        pass


def window_closed(title: str) -> bool:
    """True once the user closed the webcam window with the title-bar X.
    Without this check cv2.imshow would silently recreate the window."""
    try:
        return cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


# ---------------------------------------------------------------------------
# Camera / models
# ---------------------------------------------------------------------------
def open_camera(settings: Settings):
    cap = cv2.VideoCapture(settings.camera_index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.camera_height)
        log.info("Camera %d opened at %dx%d", settings.camera_index,
                 cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                 cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        log.warning("Camera %d could not be opened", settings.camera_index)
    return cap


def create_landmarkers():
    """Hand + face landmarkers in VIDEO mode (temporal smoothing)."""
    import mediapipe as mp

    missing = [p for p in MODEL_FILES.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing model file(s): " + ", ".join(missing))

    vision = mp.tasks.vision
    base = mp.tasks.BaseOptions
    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=base(model_asset_path=MODEL_FILES["hand"]),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ))
    # num_faces=3 so we can pick the user (largest face) even when other
    # people are in frame, instead of locking onto whoever appears first.
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=base(model_asset_path=MODEL_FILES["face"]),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=3,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ))
    return mp, hand, face


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main(settings: Settings) -> None:
    mp, hand_landmarker, face_landmarker = create_landmarkers()

    phrases = load_phrases() if settings.show_phrases else []
    stats = load_stats()
    streak_days = clean_streak(stats)
    log.info("Bites today so far: %d, clean streak: %d days",
             today_count(stats), streak_days)

    audio = AudioPlayer(WARNING_SOUNDS)
    popup = WarningPopup(ICON_PATH)
    controls = hud.Controls(muted=settings.start_muted, volume=settings.volume)
    audio.set_volume(controls.effective_volume())

    machine = BiteStateMachine(settings)
    face_memory = FaceMemory(settings.face_memory_seconds)

    cap = open_camera(settings)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open webcam {settings.camera_index}. Is another app using "
            f"it? You can change camera_index in\n{USER_DATA_DIR}\\config.json")
    time.sleep(1.0)  # let the camera settle before the first read

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
    set_window_icon(WINDOW_TITLE, ICON_PATH)
    cv2.setMouseCallback(WINDOW_TITLE, hud.make_mouse_callback(controls, audio.set_volume))

    blank = np.zeros((settings.camera_height, hud.HUD_WIDTH, 3), dtype=np.uint8)
    t0 = time.monotonic()
    prev_t = t0
    last_ts_ms = -1
    last_log = 0.0
    failed_reads = 0
    detect_errors = 0
    current_day = date.today().isoformat()
    quit_requested = False

    def stop_alert(reason: str) -> None:
        if machine.force_clear():
            log.info("<<< CLEAR (%s)", reason)
        popup.hide()
        audio.stop()

    def handle_key(key: int, now: float) -> None:
        nonlocal quit_requested
        if key in (ord("q"), 27):
            quit_requested = True
        elif key == ord("m"):
            controls.muted = not controls.muted
            audio.set_volume(controls.effective_volume())
        elif key == ord("c"):
            controls.camera_off = not controls.camera_off
        elif key == ord("s"):
            if controls.snoozed(now):
                controls.snooze_until = 0.0
                log.info("Snooze cancelled")
            else:
                controls.snooze_until = now + settings.snooze_minutes * 60
                machine.reset_dwell()
                log.info("Snoozed for %.1f min", settings.snooze_minutes)
        elif key in (ord("+"), ord("=")):
            controls.volume = min(1.0, controls.volume + 0.1)
            audio.set_volume(controls.effective_volume())
        elif key in (ord("-"), ord("_")):
            controls.volume = max(0.0, controls.volume - 0.1)
            audio.set_volume(controls.effective_volume())

    log.info("Sauron is watching. Press q in the webcam window to quit.")

    try:
        while not quit_requested:
            now = time.monotonic()
            dt = min(now - prev_t, 0.25)  # cap dt across stalls
            prev_t = now

            # ---------------- Camera off: release the device ----------------
            if controls.camera_off:
                if cap is not None:
                    cap.release()
                    cap = None
                    face_memory.reset()
                    stop_alert("camera off")
                    log.info("Camera released")
                frame = blank.copy()
                hud.draw_camera_off(frame, controls)
                cv2.imshow(WINDOW_TITLE, frame)
                handle_key(cv2.waitKey(100) & 0xFF, now)
                if window_closed(WINDOW_TITLE):
                    break
                continue

            if cap is None:
                cap = open_camera(settings)
                if cap.isOpened():
                    time.sleep(0.5)

            ret, frame = cap.read() if cap.isOpened() else (False, None)
            if not ret:
                failed_reads += 1
                if failed_reads >= 60:  # unplugged / driver hiccup: reopen
                    failed_reads = 0
                    log.warning("Camera read failing; reopening")
                    cap.release()
                    time.sleep(0.5)
                    cap = open_camera(settings)
                frame = blank.copy()
                cv2.putText(frame, "NO CAMERA SIGNAL", (190, 240), hud.FONT, 0.8,
                            (100, 100, 100), 2)
                hud.draw_controls(frame, controls)
                cv2.imshow(WINDOW_TITLE, frame)
                handle_key(cv2.waitKey(30) & 0xFF, now)
                if window_closed(WINDOW_TITLE):
                    break
                continue
            failed_reads = 0

            frame = hud.fit_width(cv2.flip(frame, 1))
            h, w = frame.shape[:2]

            # Strictly increasing wall-clock timestamps for VIDEO mode, so a
            # stall or a camera reopen is visible to the trackers' smoothing.
            ts_ms = max(int((now - t0) * 1000), last_ts_ms + 1)
            last_ts_ms = ts_ms

            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                hand_result = hand_landmarker.detect_for_video(mp_image, ts_ms)
                face_result = face_landmarker.detect_for_video(mp_image, ts_ms)
            except Exception as exc:
                # Never let a bad frame kill the watcher
                detect_errors += 1
                if detect_errors <= 5 or detect_errors % 100 == 0:
                    log.warning("Detection error #%d (skipping frame): %s",
                                detect_errors, exc)
                cv2.imshow(WINDOW_TITLE, frame)
                handle_key(cv2.waitKey(1) & 0xFF, now)
                if window_closed(WINDOW_TITLE):
                    break
                continue

            # ---------------- Geometry ----------------
            fresh_face = select_user_face(face_result.face_landmarks, w, h, settings)
            face = face_memory.update(fresh_face, now)
            hands = hand_result.hand_landmarks
            prox = fingertip_proximity(hands, hand_result.handedness, face, w, h,
                                       settings)
            mouth_open = face.mouth_open_ratio if (face and not face.remembered) else 0.0

            # ---------------- Temporal state ----------------
            snoozed = controls.snoozed(now)
            event = machine.update(prox.raw_near, mouth_open, dt, now, inhibit=snoozed)

            if event == BiteStateMachine.ALERT:
                today = date.today().isoformat()
                if today != current_day:
                    current_day = today
                    streak_days = clean_streak(stats, today)
                count = record_bite(stats, today)
                save_stats(stats)
                phrase = random.choice(phrases) if phrases else ""
                popup.show(count, phrase)
                if not controls.muted:
                    audio.play_loop()
                log.info(">>> ALERT #%d today  hand=%s dist=%.0f thr=%.0f",
                         count, prox.side, prox.min_dist, prox.threshold)
            elif event == BiteStateMachine.CLEAR:
                popup.hide()
                audio.stop()
                log.info("<<< CLEAR (%.1fs clean)", machine.clear_time)

            # Mute toggled mid-alert: stop/start the loop accordingly
            if machine.alert_active:
                if controls.muted and audio.playing:
                    audio.stop()
                elif not controls.muted and not audio.playing:
                    audio.play_loop()

            # ---------------- Periodic debug line ----------------
            if now - last_log > 0.5 and log.isEnabledFor(logging.DEBUG):
                last_log = now
                parts = [f"face={'Y' if face else 'N'}{'(mem)' if face and face.remembered else ''}",
                         f"hand={'Y' if hands else 'N'}"]
                if prox.min_dist != float("inf"):
                    parts.append(f"dist={prox.min_dist:.0f} thr={prox.threshold:.0f}")
                parts.append(f"bite={machine.bite_time:.1f}/{settings.bite_dwell_seconds}s")
                if machine.eating:
                    parts.append("EATING-SUPPRESSED")
                if snoozed:
                    parts.append("SNOOZED")
                if machine.engaged:
                    parts.append(f"NEAR({prox.side})")
                log.debug(" | ".join(parts))

            # ---------------- Draw ----------------
            threshold = prox.threshold or (
                fingertip_threshold(face.face_w_px, settings) if face else 0.0)
            hud.draw_landmarks(frame, hands, HAND_FINGERTIPS, face, threshold,
                               machine.engaged, prox.raw_near)

            if snoozed:
                status, colour = "SNOOZED", hud.AMBER
            elif machine.engaged:
                status, colour = "BITING!", hud.RED
            elif prox.raw_near and machine.eating:
                status, colour = "EATING (ignored)", hud.AMBER
            else:
                status, colour = "OK", hud.GREEN

            hud.draw_status(
                frame, status=status, colour=colour,
                bite_frac=machine.bite_time / settings.bite_dwell_seconds,
                bites_today=today_count(stats), streak_days=streak_days,
                face_seen=face is not None, hand_seen=bool(hands),
                min_dist=prox.min_dist, engaged=machine.engaged,
                snooze_remaining=controls.snooze_remaining(now))
            hud.draw_controls(frame, controls)
            cv2.imshow(WINDOW_TITLE, frame)

            handle_key(cv2.waitKey(1) & 0xFF, now)
            if window_closed(WINDOW_TITLE):
                break
    finally:
        popup.hide()
        audio.stop()
        for closer in (hand_landmarker.close, face_landmarker.close):
            try:
                closer()
            except Exception:
                pass
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        cv2.destroyAllWindows()

    log.info("Sauron has closed its eye.")


def run() -> int:
    settings, warnings = load_settings()
    setup_logging(settings.log_level)
    for warning in warnings:
        log.warning("config.json: %s", warning)
    log.info("Starting %s (python %s, frozen=%s)", APP_NAME,
             sys.version.split()[0], bool(getattr(sys, "frozen", False)))

    if not acquire_single_instance():
        fatal_dialog("Sauron is already running. Look for its webcam window "
                     "(it may be minimised).")
        return 1
    try:
        main(settings)
        return 0
    except Exception as exc:
        log.error("Fatal error:\n%s", traceback.format_exc())
        fatal_dialog(f"{exc}\n\nDetails were written to:\n{LOG_PATH}")
        return 1


if __name__ == "__main__":
    sys.exit(run())
