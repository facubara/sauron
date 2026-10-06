"""On-screen HUD for the webcam window: status overlay, clickable controls
(mute, camera, volume) and keyboard hints.

All drawing assumes a frame HUD_WIDTH pixels wide; main() resizes the
camera frame to that width so the click regions always line up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2

HUD_WIDTH = 640
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Control regions (x1, y1, x2, y2)
CTRL_MUTE = (465, 10, 545, 35)
CTRL_CAM = (555, 10, 630, 35)
CTRL_VOL = (465, 42, 630, 56)

# BGR colours
RED = (0, 0, 255)
GREEN = (0, 200, 0)
AMBER = (0, 200, 255)
GREY = (150, 150, 150)
DIM = (80, 80, 80)
WHITE = (255, 255, 255)
ORANGE = (0, 165, 255)
BLUE_BG = (160, 0, 0)

HINTS = "q quit | m mute | c camera | s snooze | -/+ volume"


@dataclass
class Controls:
    muted: bool = False
    camera_off: bool = False
    volume: float = 0.5
    dragging_volume: bool = False
    snooze_until: float = 0.0  # monotonic timestamp; 0 = not snoozed

    def effective_volume(self) -> float:
        return 0.0 if self.muted else self.volume

    def snoozed(self, now: float) -> bool:
        return self.snooze_until > now

    def snooze_remaining(self, now: float) -> float:
        return max(0.0, self.snooze_until - now)


def _in(region, x, y, pad=0) -> bool:
    x1, y1, x2, y2 = region
    return x1 <= x <= x2 and y1 - pad <= y <= y2 + pad


def _vol_from_x(x: int) -> float:
    x1, _, x2, _ = CTRL_VOL
    return max(0.0, min(1.0, (x - x1) / (x2 - x1)))


def make_mouse_callback(controls: Controls, on_volume_change):
    """Return a cv2 mouse callback. on_volume_change(effective_volume) is
    invoked whenever mute or the slider changes."""

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if _in(CTRL_MUTE, x, y):
                controls.muted = not controls.muted
                on_volume_change(controls.effective_volume())
            elif _in(CTRL_CAM, x, y):
                controls.camera_off = not controls.camera_off
            elif _in(CTRL_VOL, x, y, pad=5):
                controls.dragging_volume = True
                controls.volume = _vol_from_x(x)
                on_volume_change(controls.effective_volume())
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            if controls.dragging_volume:
                controls.volume = _vol_from_x(x)
                on_volume_change(controls.effective_volume())
        elif event == cv2.EVENT_LBUTTONUP:
            controls.dragging_volume = False

    return on_mouse


def draw_controls(frame, controls: Controls) -> None:
    """Mute button, camera button and volume slider (top right)."""
    mx1, my1, mx2, my2 = CTRL_MUTE
    cx1, cy1, cx2, cy2 = CTRL_CAM
    vx1, vy1, vx2, vy2 = CTRL_VOL

    mute_bg = BLUE_BG if controls.muted else (50, 50, 50)
    cv2.rectangle(frame, (mx1, my1), (mx2, my2), mute_bg, -1)
    cv2.rectangle(frame, (mx1, my1), (mx2, my2), GREY, 1)
    cv2.putText(frame, "MUTED" if controls.muted else "MUTE",
                (mx1 + 12, my2 - 8), FONT, 0.45, WHITE, 1)

    cam_bg = BLUE_BG if controls.camera_off else (50, 50, 50)
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), cam_bg, -1)
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), GREY, 1)
    cv2.putText(frame, "CAM OFF" if controls.camera_off else "CAM",
                (cx1 + 5, cy2 - 8), FONT, 0.4, WHITE, 1)

    fill_x = int(vx1 + (vx2 - vx1) * controls.volume)
    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (40, 40, 40), -1)
    cv2.rectangle(frame, (vx1, vy1), (fill_x, vy2), (0, 160, 0), -1)
    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), GREY, 1)
    knob_y = (vy1 + vy2) // 2
    cv2.circle(frame, (fill_x, knob_y), 8, (220, 220, 220), -1)
    cv2.circle(frame, (fill_x, knob_y), 8, GREY, 1)
    cv2.putText(frame, f"VOL {int(controls.volume * 100)}%", (vx1, vy2 + 14),
                FONT, 0.35, (180, 180, 180), 1)


def draw_landmarks(frame, hands, fingertip_ids, face, threshold, engaged,
                   raw_near) -> None:
    """Orange fingertip dots, green mouth dot and the threshold circle."""
    h, w = frame.shape[:2]
    for hand_lm in hands:
        for tip_id in fingertip_ids:
            tip = hand_lm[tip_id]
            cv2.circle(frame, (int(tip.x * w), int(tip.y * h)), 5, ORANGE, -1)
    if face is None:
        return
    centre = (int(face.mouth_x), int(face.mouth_y))
    cv2.circle(frame, centre, 6, GREEN if not face.remembered else AMBER, -1)
    colour = RED if engaged else (AMBER if raw_near else DIM)
    cv2.circle(frame, centre, int(threshold), colour, 1)


def draw_status(frame, *, status: str, colour, bite_frac: float,
                bites_today: int, streak_days: int, face_seen: bool,
                hand_seen: bool, min_dist: float, engaged: bool,
                snooze_remaining: float) -> None:
    """Everything on the HUD except the controls and landmarks."""
    h = frame.shape[0]

    cv2.putText(frame, status, (10, 30), FONT, 0.7, colour, 2)

    bar_w = int(max(0.0, min(1.0, bite_frac)) * 200)
    cv2.rectangle(frame, (10, 45), (10 + bar_w, 60), colour, -1)
    cv2.rectangle(frame, (10, 45), (210, 60), (100, 100, 100), 1)

    counter_colour = RED if bites_today > 0 else GREY
    cv2.putText(frame, f"BITES TODAY: {bites_today}", (225, 35), FONT, 0.7,
                counter_colour, 2)
    if streak_days > 0:
        cv2.putText(frame, f"clean streak: {streak_days}d", (225, 55), FONT,
                    0.45, GREEN, 1)

    if snooze_remaining > 0:
        mins, secs = divmod(int(math.ceil(snooze_remaining)), 60)
        cv2.putText(frame, f"SNOOZED {mins}:{secs:02d}", (10, 85), FONT, 0.6,
                    AMBER, 2)

    y_info = h - 32
    if face_seen:
        cv2.putText(frame, "FACE", (10, y_info), FONT, 0.4, GREEN, 1)
    if hand_seen:
        cv2.putText(frame, "HAND", (60, y_info), FONT, 0.4, GREEN, 1)
    if min_dist < math.inf:
        cv2.putText(frame, f"Dist: {min_dist:.0f}px", (120, y_info), FONT, 0.4,
                    RED if engaged else (200, 200, 200), 1)

    cv2.putText(frame, HINTS, (10, h - 10), FONT, 0.38, (130, 130, 130), 1)


def draw_version(frame, version: str, pending_update: str = "") -> None:
    """Running version in the bottom-right corner, plus a note when an
    update is downloaded and waiting for an idle moment to install."""
    h, w = frame.shape[:2]
    text = f"v{version}"
    (tw, _), _ = cv2.getTextSize(text, FONT, 0.38, 1)
    cv2.putText(frame, text, (w - tw - 10, h - 10), FONT, 0.38, (130, 130, 130), 1)
    if pending_update:
        note = f"update v{pending_update} ready - installs when idle"
        (nw, _), _ = cv2.getTextSize(note, FONT, 0.4, 1)
        cv2.putText(frame, note, (w - nw - 10, h - 32), FONT, 0.4, AMBER, 1)


def draw_camera_off(frame, controls: Controls) -> None:
    h, w = frame.shape[:2]
    frame[:] = 0
    cv2.putText(frame, "CAMERA OFF", (w // 2 - 115, h // 2), FONT, 1.0,
                (100, 100, 100), 2)
    cv2.putText(frame, "camera released - click CAM or press c to resume",
                (w // 2 - 215, h // 2 + 30), FONT, 0.45, (90, 90, 90), 1)
    cv2.putText(frame, HINTS, (10, h - 10), FONT, 0.38, (130, 130, 130), 1)
    draw_controls(frame, controls)


def fit_width(frame, width: int = HUD_WIDTH):
    """Resize to the HUD width (keeping aspect) if the camera gave us
    something else, so click regions and text positions stay valid."""
    h, w = frame.shape[:2]
    if w == width:
        return frame
    new_h = max(1, int(round(h * width / w)))
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)
