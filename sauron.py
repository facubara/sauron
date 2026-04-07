"""
Sauron - Nail Biting Detection System
Uses webcam + MediaPipe (Hand + Pose + Face landmarkers) to detect nail biting,
then shows a fullscreen popup and plays a warning sound.

Detection strategy (layered):
1. HandLandmarker fingertips near mouth (precise, tight threshold)
2. PoseLandmarker wrist near mouth (fallback when hand tracker fails)
3. Wrist-was-approaching + tracking lost (catches hand arriving at face)

All landmarkers run in VIDEO mode for temporal smoothing.
FaceLandmarker provides accurate mouth position.
"""

import cv2
import mediapipe as mp
import threading
import time
import tkinter as tk
import math
import os
import sys
import random
import pygame
import ctypes

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# PyInstaller bundles data files into sys._MEIPASS; fall back to script dir
SCRIPT_DIR = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
ICON_PATH = os.path.join(SCRIPT_DIR, "sauron-icon.ico")
WINDOW_TITLE = "Sauron - Nail Bite Detector"


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
WARNING_SOUNDS = [
    os.path.join(SCRIPT_DIR, "isengard.mp3"),
    os.path.join(SCRIPT_DIR, "sauron-sound.mp3"),
]

pygame.mixer.init()


def play_warning():
    try:
        sound_file = random.choice(WARNING_SOUNDS)
        pygame.mixer.music.load(sound_file)
        pygame.mixer.music.play()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Popup
# ---------------------------------------------------------------------------
class WarningPopup:
    def __init__(self):
        self._visible = False
        self._root = None
        self._thread = None

    def show(self, duration=2.5):
        if self._visible:
            return
        self._visible = True
        self._thread = threading.Thread(target=self._run, args=(duration,), daemon=True)
        self._thread.start()

    def _run(self, duration):
        root = tk.Tk()
        self._root = root
        root.attributes("-fullscreen", True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.82)
        root.configure(bg="#1a0000")
        root.overrideredirect(True)
        try:
            root.iconbitmap(ICON_PATH)
        except Exception:
            pass

        frame = tk.Frame(root, bg="#1a0000")
        frame.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(frame, text="STOP BITING YOUR NAILS!",
                 font=("Segoe UI", 54, "bold"), fg="#ff3333", bg="#1a0000").pack(pady=(0, 10))
        tk.Label(frame, text="Hands away from your mouth.",
                 font=("Segoe UI", 24), fg="#ff9999", bg="#1a0000").pack()

        root.after(int(duration * 1000), self._close)
        root.bind("<Button-1>", lambda e: self._close())
        root.bind("<Escape>", lambda e: self._close())
        root.mainloop()

    def _close(self):
        if self._root:
            try:
                self._root.destroy()
            except Exception:
                pass
        self._root = None
        self._visible = False


# ---------------------------------------------------------------------------
# Landmark indices
# ---------------------------------------------------------------------------
# Pose
POSE_NOSE = 0
POSE_MOUTH_LEFT = 9
POSE_MOUTH_RIGHT = 10
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16

# Hand (21-point model)
HAND_THUMB_TIP = 4
HAND_INDEX_TIP = 8
HAND_MIDDLE_TIP = 12
HAND_RING_TIP = 16
HAND_PINKY_TIP = 20
HAND_FINGERTIPS = [HAND_THUMB_TIP, HAND_INDEX_TIP, HAND_MIDDLE_TIP,
                   HAND_RING_TIP, HAND_PINKY_TIP]

# Face mesh (478-point model) — mouth landmarks
FACE_UPPER_LIP = 13
FACE_LOWER_LIP = 14

# Pose skeleton connections for drawing
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
    (17, 19), (18, 20), (11, 23), (12, 24),
]


def dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def lm_visibility(landmark):
    """Get landmark visibility score, defaulting to 0."""
    v = getattr(landmark, 'visibility', None)
    return v if v is not None else 0.0


# HUD control regions (x1, y1, x2, y2) for 640px-wide frame
_CTRL_MUTE = (465, 10, 545, 35)
_CTRL_CAM = (555, 10, 630, 35)
_CTRL_VOL = (465, 42, 630, 56)


def _draw_controls(frame, controls):
    """Draw mute, camera, and volume controls on the HUD."""
    mx1, my1, mx2, my2 = _CTRL_MUTE
    cx1, cy1, cx2, cy2 = _CTRL_CAM
    vx1, vy1, vx2, vy2 = _CTRL_VOL

    # Mute button
    mute_bg = (0, 0, 160) if controls["muted"] else (50, 50, 50)
    cv2.rectangle(frame, (mx1, my1), (mx2, my2), mute_bg, -1)
    cv2.rectangle(frame, (mx1, my1), (mx2, my2), (150, 150, 150), 1)
    mute_label = "MUTED" if controls["muted"] else "MUTE"
    cv2.putText(frame, mute_label, (mx1 + 12, my2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Camera button
    cam_bg = (0, 0, 160) if controls["camera_off"] else (50, 50, 50)
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), cam_bg, -1)
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (150, 150, 150), 1)
    cam_label = "CAM OFF" if controls["camera_off"] else "CAM"
    cv2.putText(frame, cam_label, (cx1 + 5, cy2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Volume slider
    vol_fill_x = int(vx1 + (vx2 - vx1) * controls["volume"])
    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (40, 40, 40), -1)
    cv2.rectangle(frame, (vx1, vy1), (vol_fill_x, vy2), (0, 160, 0), -1)
    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (150, 150, 150), 1)
    knob_y = (vy1 + vy2) // 2
    cv2.circle(frame, (vol_fill_x, knob_y), 8, (220, 220, 220), -1)
    cv2.circle(frame, (vol_fill_x, knob_y), 8, (150, 150, 150), 1)
    vol_pct = int(controls["volume"] * 100)
    cv2.putText(frame, f"VOL {vol_pct}%", (vx1, vy2 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- Create landmarkers (all VIDEO mode for temporal smoothing) ---
    pose_landmarker = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=os.path.join(SCRIPT_DIR, "pose_landmarker.task")
        ),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ))

    hand_landmarker = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=os.path.join(SCRIPT_DIR, "hand_landmarker.task")
        ),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ))

    face_landmarker = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=os.path.join(SCRIPT_DIR, "face_landmarker.task")
        ),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Give the camera time to initialize before reading frames
    time.sleep(1.0)

    # Create OpenCV window and set its icon via Win32 API
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, WINDOW_TITLE)
        if hwnd:
            IMAGE_ICON = 1
            LR_LOADFROMFILE = 0x0010
            LR_DEFAULTSIZE = 0x0040
            icon = user32.LoadImageW(
                0, ICON_PATH, IMAGE_ICON, 0, 0,
                LR_LOADFROMFILE | LR_DEFAULTSIZE)
            if icon:
                ICON_SMALL = 0
                ICON_BIG = 1
                WM_SETICON = 0x0080
                user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, icon)
                user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, icon)
    except Exception:
        pass

    popup = WarningPopup()

    controls = {
        "muted": False,
        "camera_off": False,
        "volume": 0.5,
        "dragging_volume": False,
    }
    pygame.mixer.music.set_volume(controls["volume"])

    def on_mouse(event, x, y, flags, param):
        ctrl = param
        mx1, my1, mx2, my2 = _CTRL_MUTE
        cx1, cy1, cx2, cy2 = _CTRL_CAM
        vx1, vy1, vx2, vy2 = _CTRL_VOL

        if event == cv2.EVENT_LBUTTONDOWN:
            if mx1 <= x <= mx2 and my1 <= y <= my2:
                ctrl["muted"] = not ctrl["muted"]
                pygame.mixer.music.set_volume(
                    0.0 if ctrl["muted"] else ctrl["volume"])
            elif cx1 <= x <= cx2 and cy1 <= y <= cy2:
                ctrl["camera_off"] = not ctrl["camera_off"]
            elif vx1 <= x <= vx2 and vy1 - 5 <= y <= vy2 + 5:
                ctrl["dragging_volume"] = True
                ctrl["volume"] = max(0.0, min(1.0,
                                              (x - vx1) / (vx2 - vx1)))
                if not ctrl["muted"]:
                    pygame.mixer.music.set_volume(ctrl["volume"])
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            if ctrl["dragging_volume"]:
                ctrl["volume"] = max(0.0, min(1.0,
                                              (x - vx1) / (vx2 - vx1)))
                if not ctrl["muted"]:
                    pygame.mixer.music.set_volume(ctrl["volume"])
        elif event == cv2.EVENT_LBUTTONUP:
            ctrl["dragging_volume"] = False

    cv2.setMouseCallback(WINDOW_TITLE, on_mouse, controls)

    bite_frames = 0
    BITE_THRESHOLD_FRAMES = 6
    cooldown_until = 0
    last_log = 0
    frame_count = 0

    # Wrist trajectory tracking (for "approaching mouth then lost" detection)
    prev_wrist_dist = {"L": None, "R": None}   # previous distance to mouth
    wrist_approaching = {"L": False, "R": False}

    # Face/mouth occlusion tracking
    face_present_streak = 0       # consecutive frames with face detected
    prev_mouth_pos = None         # (x, y) of last mouth center
    FACE_STREAK_MIN = 10          # need this many frames of face before "lost" counts

    print("Sauron is watching... Press 'q' in the webcam window to quit.")
    print("--- Debug log (every 0.5s) ---")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Camera off: show black frame with controls only
            if controls["camera_off"]:
                frame[:] = 0
                cv2.putText(frame, "CAMERA OFF", (w // 2 - 115, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
                _draw_controls(frame, controls)
                cv2.imshow(WINDOW_TITLE, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Monotonically increasing timestamp for VIDEO mode
            frame_count += 1
            timestamp_ms = frame_count * 33  # ~30 fps

            # Run all three landmarkers
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

            now = time.time()
            detected = False
            detection_method = ""
            min_d = float("inf")
            threshold = 0
            side = ""

            has_pose = len(pose_result.pose_landmarks) > 0
            has_hands = len(hand_result.hand_landmarks) > 0
            has_face = len(face_result.face_landmarks) > 0

            # ----- Mouth position (face mesh preferred, pose fallback) -----
            mouth_cx, mouth_cy = None, None
            mouth_jump = 0.0  # how far mouth moved vs last frame (pixels)

            if has_face:
                face_lm = face_result.face_landmarks[0]
                upper = face_lm[FACE_UPPER_LIP]
                lower = face_lm[FACE_LOWER_LIP]
                mouth_cx = (upper.x + lower.x) / 2 * w
                mouth_cy = (upper.y + lower.y) / 2 * h
                # Track mouth jump (sudden landmark shift = occlusion)
                if prev_mouth_pos is not None:
                    mouth_jump = dist(mouth_cx, mouth_cy,
                                      prev_mouth_pos[0], prev_mouth_pos[1])
                prev_mouth_pos = (mouth_cx, mouth_cy)
                face_present_streak += 1
            elif has_pose:
                pose_lm = pose_result.pose_landmarks[0]
                ml = pose_lm[POSE_MOUTH_LEFT]
                mr = pose_lm[POSE_MOUTH_RIGHT]
                if lm_visibility(ml) > 0.5 and lm_visibility(mr) > 0.5:
                    mouth_cx = (ml.x + mr.x) / 2 * w
                    mouth_cy = (ml.y + mr.y) / 2 * h

            # ----- Shoulder width (for scale-adaptive thresholds) -----
            shoulder_d = None
            if has_pose:
                pose_lm = pose_result.pose_landmarks[0]
                ls = pose_lm[POSE_LEFT_SHOULDER]
                rs = pose_lm[POSE_RIGHT_SHOULDER]
                if lm_visibility(ls) > 0.5 and lm_visibility(rs) > 0.5:
                    shoulder_d = dist(ls.x * w, ls.y * h, rs.x * w, rs.y * h)

            # =============================================================
            # DETECTION METHOD 1: Hand tracker fingertips near mouth (precise)
            # =============================================================
            if has_hands and mouth_cx is not None:
                hand_thr = 50.0
                if shoulder_d:
                    hand_thr = max(shoulder_d * 0.2, 35.0)

                for i, hand_lm in enumerate(hand_result.hand_landmarks):
                    for tip_id in HAND_FINGERTIPS:
                        tip = hand_lm[tip_id]
                        tx, ty = tip.x * w, tip.y * h
                        d = dist(tx, ty, mouth_cx, mouth_cy)
                        if d < min_d:
                            min_d = d
                            side = hand_result.handedness[i][0].category_name[0]
                        if d < hand_thr and not detected:
                            detected = True
                            detection_method = "hand_fingertip"
                            threshold = hand_thr

            # =============================================================
            # DETECTION METHOD 2 & 3: Pose wrist fallback + wrist-lost
            # Fires when hand fingertips didn't already trigger detection.
            # =============================================================
            if has_pose and mouth_cx is not None and shoulder_d:
                pose_lm = pose_result.pose_landmarks[0]
                wrist_thr = shoulder_d * 0.25

                for wrist_side, wrist_id in [("L", POSE_LEFT_WRIST),
                                              ("R", POSE_RIGHT_WRIST)]:
                    wrist = pose_lm[wrist_id]
                    vis = lm_visibility(wrist)

                    if vis < 0.6:
                        # Low visibility — check if wrist was approaching
                        # OR was already close when tracking was lost
                        if not detected:
                            prev = prev_wrist_dist[wrist_side]
                            if prev is not None and prev < shoulder_d * 0.4:
                                if wrist_approaching[wrist_side] or prev < wrist_thr:
                                    detected = True
                                    detection_method = "wrist_lost"
                                    threshold = shoulder_d * 0.4
                                    side = wrist_side
                                    if prev < min_d:
                                        min_d = prev
                        # Reset trajectory since we can't trust the position
                        prev_wrist_dist[wrist_side] = None
                        wrist_approaching[wrist_side] = False
                        continue

                    wx, wy = wrist.x * w, wrist.y * h
                    d = dist(wx, wy, mouth_cx, mouth_cy)

                    # Update trajectory
                    prev = prev_wrist_dist[wrist_side]
                    if prev is not None:
                        wrist_approaching[wrist_side] = d < prev - 2  # 2px hysteresis
                    else:
                        # First frame seeing this wrist — if already close,
                        # treat as approaching (hand entered frame near mouth)
                        wrist_approaching[wrist_side] = d < shoulder_d * 0.5
                    prev_wrist_dist[wrist_side] = d

                    if d < min_d:
                        min_d = d
                        side = wrist_side

                    # Direct wrist detection when fingertip method didn't fire
                    if not detected and d < wrist_thr:
                        detected = True
                        detection_method = "pose_wrist"
                        threshold = wrist_thr

            # =============================================================
            # DETECTION METHOD 4: Mouth occluded / face lost
            # If face was consistently tracked and suddenly lost while
            # pose still sees you, something is blocking the mouth.
            # Also triggers on large sudden mouth-landmark jumps.
            # =============================================================
            if not detected and has_pose:
                # 4a: Face was tracked, now lost → hand covering face
                if not has_face and face_present_streak >= FACE_STREAK_MIN:
                    detected = True
                    detection_method = "face_occluded"
                    threshold = 0
                    min_d = 0

                # 4b: Mouth landmarks jumped abnormally (face mesh distorted)
                if not detected and has_face and shoulder_d and mouth_jump > shoulder_d * 0.15:
                    detected = True
                    detection_method = "mouth_distorted"
                    threshold = shoulder_d * 0.15
                    min_d = mouth_jump

            # Update face streak (reset AFTER checking method 4)
            if not has_face:
                face_present_streak = 0
                prev_mouth_pos = None

            # =============================================================
            # Draw visualization
            # =============================================================
            # Pose skeleton
            if has_pose:
                pose_lm = pose_result.pose_landmarks[0]
                for c1, c2 in POSE_CONNECTIONS:
                    if c1 < len(pose_lm) and c2 < len(pose_lm):
                        p1 = (int(pose_lm[c1].x * w), int(pose_lm[c1].y * h))
                        p2 = (int(pose_lm[c2].x * w), int(pose_lm[c2].y * h))
                        cv2.line(frame, p1, p2, (180, 180, 180), 1)

            # Hand fingertips (orange dots)
            if has_hands:
                for hand_lm in hand_result.hand_landmarks:
                    for tip_id in HAND_FINGERTIPS:
                        tip = hand_lm[tip_id]
                        cx, cy = int(tip.x * w), int(tip.y * h)
                        cv2.circle(frame, (cx, cy), 5, (0, 165, 255), -1)

            # Mouth position + threshold circle
            if mouth_cx is not None:
                cv2.circle(frame, (int(mouth_cx), int(mouth_cy)), 6, (0, 255, 0), -1)
                draw_thr = threshold if threshold > 0 else (
                    shoulder_d * 0.25 if shoulder_d else 50)
                cv2.circle(frame, (int(mouth_cx), int(mouth_cy)), int(draw_thr),
                           (0, 0, 255) if detected else (80, 80, 80), 1)

            # ----- Consecutive-frame logic -----
            if detected and now > cooldown_until:
                bite_frames += 1
            else:
                bite_frames = max(0, bite_frames - 1)

            if bite_frames >= BITE_THRESHOLD_FRAMES:
                bite_frames = 0
                cooldown_until = now + 5
                if not controls["muted"]:
                    play_warning()
                popup.show(duration=2.5)
                print(f"  >>> ALERT! method={detection_method} hand={side}"
                      f" dist={min_d:.0f} thr={threshold:.0f}")

            # ----- Console log (every 0.5s) -----
            if now - last_log > 0.5:
                last_log = now
                parts = [
                    f"face={'Y' if has_face else 'N'}",
                    f"hand={'Y' if has_hands else 'N'}",
                    f"pose={'Y' if has_pose else 'N'}",
                ]
                if mouth_cx is not None:
                    parts.append(f"dist={min_d:.0f}")
                    if threshold > 0:
                        parts.append(f"thr={threshold:.0f}")
                    parts.append(f"bite={bite_frames}/{BITE_THRESHOLD_FRAMES}")
                    if detected:
                        parts.append(f"DETECTED({detection_method},{side})")
                print("  " + " | ".join(parts))

            # ----- On-screen overlay -----
            status = f"BITING! ({detection_method})" if detected else "OK"
            color = (0, 0, 255) if detected else (0, 200, 0)
            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            bar_w = int((bite_frames / BITE_THRESHOLD_FRAMES) * 200)
            cv2.rectangle(frame, (10, 45), (10 + bar_w, 60), color, -1)
            cv2.rectangle(frame, (10, 45), (210, 60), (100, 100, 100), 1)

            # Detection source indicators
            y_info = h - 15
            if has_face:
                cv2.putText(frame, "FACE", (10, y_info),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
            if has_hands:
                cv2.putText(frame, "HAND", (60, y_info),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
            if has_pose:
                cv2.putText(frame, "POSE", (120, y_info),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
            if min_d < float("inf"):
                d_color = (0, 0, 255) if detected else (200, 200, 200)
                cv2.putText(frame, f"Dist: {min_d:.0f}px", (180, y_info),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, d_color, 1)

            _draw_controls(frame, controls)
            cv2.imshow(WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pose_landmarker.close()
        hand_landmarker.close()
        face_landmarker.close()
        cap.release()
        cv2.destroyAllWindows()

    print("Sauron has closed its eye.")


if __name__ == "__main__":
    main()
