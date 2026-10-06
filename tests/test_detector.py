"""Unit tests for the pure detection logic (no camera, no MediaPipe)."""

import math
from collections import namedtuple

import pytest

from config import Settings
from detector import (
    FACE_CHEEK_LEFT,
    FACE_CHEEK_RIGHT,
    FACE_LOWER_LIP,
    FACE_MOUTH_LEFT,
    FACE_MOUTH_RIGHT,
    FACE_UPPER_LIP,
    HAND_INDEX_TIP,
    HAND_MIDDLE_MCP,
    HAND_WRIST,
    BiteStateMachine,
    FaceMemory,
    fingertip_proximity,
    select_user_face,
)

W, H = 640, 480
LM = namedtuple("LM", "x y")


def make_face(cx=0.5, cy=0.5, width_frac=0.3, lip_gap_frac=0.02):
    """478 dummy landmarks with the few we care about placed sensibly.
    width_frac is the cheek-to-cheek width as a fraction of frame width."""
    lm = [LM(cx, cy)] * 478
    half = width_frac / 2
    lm[FACE_CHEEK_LEFT] = LM(cx - half, cy)
    lm[FACE_CHEEK_RIGHT] = LM(cx + half, cy)
    mouth_y = cy + width_frac * 0.5
    lm[FACE_UPPER_LIP] = LM(cx, mouth_y - lip_gap_frac / 2)
    lm[FACE_LOWER_LIP] = LM(cx, mouth_y + lip_gap_frac / 2)
    lm[FACE_MOUTH_LEFT] = LM(cx - width_frac * 0.2, mouth_y)
    lm[FACE_MOUTH_RIGHT] = LM(cx + width_frac * 0.2, mouth_y)
    return lm


def make_hand(tip_x, tip_y, span_px=60.0, wrist=(0.5, 0.95)):
    """21 dummy landmarks: wrist, middle knuckle (span_px above the wrist)
    and the index fingertip at the given normalised position."""
    lm = [LM(*wrist)] * 21
    lm[HAND_WRIST] = LM(*wrist)
    lm[HAND_MIDDLE_MCP] = LM(wrist[0], wrist[1] - span_px / H)
    lm[HAND_INDEX_TIP] = LM(tip_x, tip_y)
    return lm


class Cat:
    def __init__(self, name):
        self.category_name = name


HANDEDNESS = [[Cat("Right")], [Cat("Left")]]


# ---------------------------------------------------------------------------
# Face selection
# ---------------------------------------------------------------------------
def test_no_faces_returns_none():
    assert select_user_face([], W, H, Settings()) is None


def test_face_too_small_is_ignored():
    s = Settings()
    tiny = make_face(width_frac=s.min_face_width_frac * 0.5)
    assert select_user_face([tiny], W, H, s) is None


def test_largest_face_wins():
    big = make_face(cx=0.3, width_frac=0.4)
    small = make_face(cx=0.8, width_frac=0.15)
    face = select_user_face([small, big], W, H, Settings())
    assert face is not None
    assert face.mouth_x == pytest.approx(0.3 * W)
    assert face.face_w_px == pytest.approx(0.4 * W)


def test_mouth_open_ratio():
    closed = select_user_face([make_face(lip_gap_frac=0.0)], W, H, Settings())
    wide = select_user_face([make_face(width_frac=0.3, lip_gap_frac=0.12)], W, H, Settings())
    assert closed.mouth_open_ratio == 0.0
    # lip gap 0.12*H px over mouth width 0.4*0.3*W px
    assert wide.mouth_open_ratio == pytest.approx((0.12 * H) / (0.12 * W))


# ---------------------------------------------------------------------------
# Fingertip proximity
# ---------------------------------------------------------------------------
def face_at_mouth():
    face = select_user_face([make_face()], W, H, Settings())
    return face, face.mouth_x / W, face.mouth_y / H


def test_fingertip_at_mouth_is_near():
    s = Settings()
    face, mx, my = face_at_mouth()
    hand = make_hand(mx, my, span_px=face.face_w_px * 0.6)
    prox = fingertip_proximity([hand], HANDEDNESS, face, W, H, s)
    assert prox.raw_near
    assert prox.min_dist == pytest.approx(0.0)
    assert prox.side == "R"
    assert prox.threshold == pytest.approx(face.face_w_px * s.fingertip_thr_face_frac)


def test_fingertip_far_from_mouth_is_not_near():
    s = Settings()
    face, _, _ = face_at_mouth()
    hand = make_hand(0.05, 0.05, span_px=face.face_w_px * 0.6)
    prox = fingertip_proximity([hand], HANDEDNESS, face, W, H, s)
    assert not prox.raw_near
    assert prox.min_dist > prox.threshold


def test_hand_with_wrong_scale_is_rejected():
    """A hand much smaller than the face belongs to someone further away."""
    s = Settings()
    face, mx, my = face_at_mouth()
    tiny_hand = make_hand(mx, my, span_px=face.face_w_px * s.hand_face_ratio_min * 0.5)
    prox = fingertip_proximity([tiny_hand], HANDEDNESS, face, W, H, s)
    assert not prox.raw_near
    assert prox.min_dist == math.inf


def test_no_face_means_no_proximity():
    prox = fingertip_proximity([make_hand(0.5, 0.5)], HANDEDNESS, None, W, H, Settings())
    assert not prox.raw_near and prox.threshold == 0.0


def test_closest_hand_decides_side():
    s = Settings()
    face, mx, my = face_at_mouth()
    span = face.face_w_px * 0.6
    far_right = make_hand(0.05, 0.05, span_px=span)
    near_left = make_hand(mx + 0.01, my, span_px=span)
    prox = fingertip_proximity([far_right, near_left], HANDEDNESS, face, W, H, s)
    assert prox.side == "L"


def test_missing_handedness_gives_question_mark():
    face, mx, my = face_at_mouth()
    prox = fingertip_proximity([make_hand(mx, my, span_px=face.face_w_px * 0.6)],
                               [], face, W, H, Settings())
    assert prox.side == "?"


# ---------------------------------------------------------------------------
# Face memory
# ---------------------------------------------------------------------------
def test_face_memory_bridges_short_dropout():
    mem = FaceMemory(memory_seconds=1.0)
    face, _, _ = face_at_mouth()
    assert mem.update(face, now=10.0) is face
    remembered = mem.update(None, now=10.5)
    assert remembered is not None and remembered.remembered
    assert remembered.mouth_x == face.mouth_x
    assert mem.update(None, now=11.5) is None  # expired
    assert mem.update(None, now=11.6) is None  # stays gone


def test_face_memory_zero_disables_it():
    mem = FaceMemory(memory_seconds=0.0)
    face, _, _ = face_at_mouth()
    mem.update(face, now=0.0)
    assert mem.update(None, now=0.001) is None


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------
def run_frames(machine, raw_near, seconds, start, dt=0.05, mouth_open=0.0,
               inhibit=False):
    """Feed identical frames for `seconds`; return (events, end_time)."""
    events = []
    t = start
    for _ in range(int(round(seconds / dt))):
        t += dt
        ev = machine.update(raw_near, mouth_open, dt, t, inhibit=inhibit)
        if ev:
            events.append((ev, t))
    return events, t


def test_alert_fires_only_after_dwell():
    s = Settings(bite_dwell_seconds=1.5)
    m = BiteStateMachine(s)
    events, t = run_frames(m, True, 1.0, start=0.0)
    assert events == []
    assert not m.alert_active
    events, t = run_frames(m, True, 0.6, start=t)
    assert [e for e, _ in events] == ["alert"]
    assert m.alert_active
    assert events[0][1] == pytest.approx(1.5, abs=0.06)


def test_quick_pass_never_alerts():
    s = Settings(bite_dwell_seconds=1.5)
    m = BiteStateMachine(s)
    t = 0.0
    for _ in range(10):  # 0.5s near, 0.5s away, repeated
        ev, t = run_frames(m, True, 0.5, start=t)
        assert ev == []
        ev, t = run_frames(m, False, 0.5, start=t)
        assert ev == []
    assert m.bite_time < s.bite_dwell_seconds


def test_dwell_decays_twice_as_fast():
    m = BiteStateMachine(Settings(bite_dwell_seconds=2.0))
    _, t = run_frames(m, True, 1.0, start=0.0)
    assert m.bite_time == pytest.approx(1.0, abs=0.06)
    run_frames(m, False, 0.5, start=t)
    assert m.bite_time == pytest.approx(0.0, abs=0.06)


def test_clear_requires_hand_away_and_min_duration():
    s = Settings(bite_dwell_seconds=1.0, clear_seconds=1.0, min_alert_seconds=3.0)
    m = BiteStateMachine(s)
    events, t = run_frames(m, True, 1.1, start=0.0)
    assert [e for e, _ in events] == ["alert"]
    alert_t = events[0][1]
    # Hand leaves immediately: clear_time satisfied after 1s but
    # min_alert_seconds (3s) is not, so it must wait.
    events, t = run_frames(m, False, 1.5, start=t)
    assert events == []
    assert m.alert_active
    events, t = run_frames(m, False, 2.0, start=t)
    assert [e for e, _ in events] == ["clear"]
    assert events[0][1] - alert_t == pytest.approx(3.0, abs=0.06)
    assert not m.alert_active
    assert m.bite_time == 0.0


def test_hand_returning_resets_clear_timer():
    s = Settings(bite_dwell_seconds=1.0, clear_seconds=1.0, min_alert_seconds=0.1)
    m = BiteStateMachine(s)
    _, t = run_frames(m, True, 1.1, start=0.0)
    assert m.alert_active
    _, t = run_frames(m, False, 0.8, start=t)
    _, t = run_frames(m, True, 0.1, start=t)  # hand back briefly
    assert m.clear_time == 0.0
    events, t = run_frames(m, False, 0.8, start=t)
    assert events == []
    events, t = run_frames(m, False, 0.3, start=t)
    assert [e for e, _ in events] == ["clear"]


def test_eating_suppresses_detection_for_a_while():
    s = Settings(bite_dwell_seconds=1.0, mouth_open_ratio=0.4,
                 eating_suppress_seconds=4.0)
    m = BiteStateMachine(s)
    # One wide-open frame, then fingers at the (now closed) mouth.
    m.update(False, 0.9, 0.05, 0.0)
    assert m.eating
    events, t = run_frames(m, True, 3.5, start=0.0)
    assert events == []
    assert m.bite_time == 0.0
    # After suppression lapses the dwell accumulates normally.
    events, t = run_frames(m, True, 1.6, start=t)
    assert [e for e, _ in events] == ["alert"]


def test_inhibit_blocks_new_alerts_but_allows_clear():
    s = Settings(bite_dwell_seconds=1.0, clear_seconds=0.5, min_alert_seconds=0.5)
    m = BiteStateMachine(s)
    events, t = run_frames(m, True, 2.0, start=0.0, inhibit=True)
    assert events == []
    assert m.bite_time == 0.0
    events, t = run_frames(m, True, 1.1, start=t)
    assert [e for e, _ in events] == ["alert"]
    events, t = run_frames(m, False, 1.0, start=t, inhibit=True)
    assert [e for e, _ in events] == ["clear"]


def test_force_clear():
    m = BiteStateMachine(Settings(bite_dwell_seconds=1.0))
    run_frames(m, True, 1.1, start=0.0)
    assert m.alert_active
    assert m.force_clear() is True
    assert not m.alert_active and m.bite_time == 0.0
    assert m.force_clear() is False


def test_reset_dwell():
    m = BiteStateMachine(Settings(bite_dwell_seconds=2.0))
    run_frames(m, True, 1.0, start=0.0)
    assert m.bite_time > 0
    m.reset_dwell()
    assert m.bite_time == 0.0
