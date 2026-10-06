"""Pure detection logic: which face is the user, are their fingertips at
their mouth, and has that lasted long enough to be a bite.

No cv2 / mediapipe imports. Landmarks are any objects with normalised
``.x`` and ``.y`` attributes (MediaPipe's NormalizedLandmark qualifies, and
so does a namedtuple in the tests).

Detection strategy (precision over recall - minimise false positives):
- Only the user is tracked: the largest face in frame, and only if it is
  big enough to be someone sitting at the camera. Background people are
  ignored. Hands must be size-consistent with that face.
- Only fingertips near the mouth count, and they must stay there for a
  sustained dwell time before an alert fires. A quick pass (grabbing food,
  scratching) never accumulates enough.
- Eating suppression: if the mouth opens wide (taking a bite of food),
  detection is suppressed for a few seconds. Nail biting happens with the
  lips barely parted; eating doesn't.
- Face memory: a hand at the mouth can make the face tracker drop out for
  a few frames. The last known mouth position is kept briefly so the dwell
  timer does not reset in the middle of a bite.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from config import Settings

# ---------------------------------------------------------------------------
# Landmark indices
# ---------------------------------------------------------------------------
# Hand (21-point model)
HAND_WRIST = 0
HAND_MIDDLE_MCP = 9
HAND_THUMB_TIP = 4
HAND_INDEX_TIP = 8
HAND_MIDDLE_TIP = 12
HAND_RING_TIP = 16
HAND_PINKY_TIP = 20
HAND_FINGERTIPS = (HAND_THUMB_TIP, HAND_INDEX_TIP, HAND_MIDDLE_TIP,
                   HAND_RING_TIP, HAND_PINKY_TIP)

# Face mesh (478-point model)
FACE_UPPER_LIP = 13
FACE_LOWER_LIP = 14
FACE_MOUTH_LEFT = 61
FACE_MOUTH_RIGHT = 291
FACE_CHEEK_LEFT = 234
FACE_CHEEK_RIGHT = 454


def dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)


# ---------------------------------------------------------------------------
# Per-frame geometry
# ---------------------------------------------------------------------------
@dataclass
class FaceObservation:
    mouth_x: float
    mouth_y: float
    face_w_px: float
    mouth_open_ratio: float
    remembered: bool = False  # True when served from FaceMemory, not a fresh detection


def select_user_face(faces: Sequence[Sequence], w: int, h: int,
                     settings: Settings) -> FaceObservation | None:
    """Pick the largest face that is big enough to be the person at the
    camera. Returns None if nobody qualifies."""
    best = None
    best_w = 0.0
    for lm in faces:
        cl, cr = lm[FACE_CHEEK_LEFT], lm[FACE_CHEEK_RIGHT]
        fw = dist(cl.x * w, cl.y * h, cr.x * w, cr.y * h)
        if fw > best_w:
            best_w = fw
            best = lm
    if best is None or best_w < settings.min_face_width_frac * w:
        return None

    upper, lower = best[FACE_UPPER_LIP], best[FACE_LOWER_LIP]
    mouth_x = (upper.x + lower.x) / 2 * w
    mouth_y = (upper.y + lower.y) / 2 * h
    ml, mr = best[FACE_MOUTH_LEFT], best[FACE_MOUTH_RIGHT]
    mouth_w = dist(ml.x * w, ml.y * h, mr.x * w, mr.y * h)
    lip_gap = dist(upper.x * w, upper.y * h, lower.x * w, lower.y * h)
    open_ratio = lip_gap / mouth_w if mouth_w > 1 else 0.0
    return FaceObservation(mouth_x, mouth_y, best_w, open_ratio)


def fingertip_threshold(face_w_px: float, settings: Settings) -> float:
    return max(face_w_px * settings.fingertip_thr_face_frac,
               settings.fingertip_thr_min_px)


@dataclass
class HandProximity:
    raw_near: bool = False
    min_dist: float = math.inf
    threshold: float = 0.0
    side: str = ""  # "L", "R" or "?" for the closest hand


def fingertip_proximity(hands: Sequence[Sequence], handedness: Sequence,
                        face: FaceObservation | None, w: int, h: int,
                        settings: Settings) -> HandProximity:
    """Closest fingertip of any size-consistent hand to the user's mouth."""
    result = HandProximity()
    if face is None or not hands:
        return result
    result.threshold = fingertip_threshold(face.face_w_px, settings)
    for i, hand_lm in enumerate(hands):
        wr, mcp = hand_lm[HAND_WRIST], hand_lm[HAND_MIDDLE_MCP]
        hand_span = dist(wr.x * w, wr.y * h, mcp.x * w, mcp.y * h)
        ratio = hand_span / face.face_w_px if face.face_w_px > 1 else 0.0
        if not (settings.hand_face_ratio_min <= ratio <= settings.hand_face_ratio_max):
            continue
        for tip_id in HAND_FINGERTIPS:
            tip = hand_lm[tip_id]
            d = dist(tip.x * w, tip.y * h, face.mouth_x, face.mouth_y)
            if d < result.min_dist:
                result.min_dist = d
                result.side = _hand_side(handedness, i)
            if d < result.threshold:
                result.raw_near = True
    return result


def _hand_side(handedness: Sequence, index: int) -> str:
    try:
        return handedness[index][0].category_name[0]
    except (IndexError, AttributeError, TypeError):
        return "?"


# ---------------------------------------------------------------------------
# Temporal state
# ---------------------------------------------------------------------------
class FaceMemory:
    """Bridges short face-tracking dropouts with the last known position."""

    def __init__(self, memory_seconds: float):
        self.memory_seconds = memory_seconds
        self._last: FaceObservation | None = None
        self._last_seen = -math.inf

    def update(self, face: FaceObservation | None, now: float) -> FaceObservation | None:
        if face is not None:
            self._last = face
            self._last_seen = now
            return face
        if self._last is not None and now - self._last_seen <= self.memory_seconds:
            remembered = FaceObservation(self._last.mouth_x, self._last.mouth_y,
                                         self._last.face_w_px,
                                         self._last.mouth_open_ratio, remembered=True)
            return remembered
        self._last = None
        return None

    def reset(self) -> None:
        self._last = None
        self._last_seen = -math.inf


@dataclass
class BiteState:
    bite_time: float = 0.0      # accumulated seconds of sustained proximity
    clear_time: float = 0.0     # accumulated seconds with hand away
    alert_active: bool = False
    alert_started_at: float = 0.0
    last_wide_open: float = -math.inf
    eating: bool = False
    engaged: bool = False
    raw_near: bool = False
    events: list = field(default_factory=list)


class BiteStateMachine:
    """Turns per-frame proximity into ALERT / CLEAR events.

    update() returns "alert" when a new alert should start, "clear" when the
    active alert should end, else None.
    """

    ALERT = "alert"
    CLEAR = "clear"

    def __init__(self, settings: Settings):
        self.s = settings
        self.state = BiteState()

    # Convenience accessors used by the HUD
    @property
    def bite_time(self) -> float:
        return self.state.bite_time

    @property
    def clear_time(self) -> float:
        return self.state.clear_time

    @property
    def alert_active(self) -> bool:
        return self.state.alert_active

    @property
    def eating(self) -> bool:
        return self.state.eating

    @property
    def engaged(self) -> bool:
        return self.state.engaged

    def reset_dwell(self) -> None:
        """Forget accumulated proximity (camera off, snooze)."""
        self.state.bite_time = 0.0

    def update(self, raw_near: bool, mouth_open_ratio: float, dt: float,
               now: float, inhibit: bool = False) -> str | None:
        """Advance by one frame.

        raw_near          a fingertip is inside the mouth threshold
        mouth_open_ratio  lip gap / mouth width (0 if no face)
        dt                seconds since the previous frame (already capped)
        now               monotonic timestamp of this frame
        inhibit           True while snoozed: never start a new alert
        """
        s, st = self.s, self.state

        if mouth_open_ratio > s.mouth_open_ratio:
            st.last_wide_open = now
        st.eating = (now - st.last_wide_open) < s.eating_suppress_seconds
        st.raw_near = raw_near
        st.engaged = raw_near and not st.eating

        if st.engaged and not inhibit:
            st.bite_time = min(st.bite_time + dt, s.bite_dwell_seconds)
        else:
            st.bite_time = max(0.0, st.bite_time - dt * 2)
        if raw_near:
            st.clear_time = 0.0
        else:
            st.clear_time += dt

        if not st.alert_active and not inhibit and st.bite_time >= s.bite_dwell_seconds:
            st.alert_active = True
            st.alert_started_at = now
            return self.ALERT

        if (st.alert_active
                and st.clear_time >= s.clear_seconds
                and now - st.alert_started_at >= s.min_alert_seconds):
            st.alert_active = False
            st.bite_time = 0.0
            return self.CLEAR

        return None

    def force_clear(self) -> bool:
        """End any active alert immediately (camera off, quit). Returns
        True if an alert was active."""
        was_active = self.state.alert_active
        self.state.alert_active = False
        self.state.bite_time = 0.0
        return was_active
