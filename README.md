# Sauron - Nail Bite Detector

Sauron watches your webcam and, when it sees your fingertips rest at your
mouth for more than a moment, covers the screen with a fullscreen warning
and loops a sound until your hand goes back down. It counts bites per day
and shows a clean-day streak.

Detection runs entirely on your machine with MediaPipe hand and face
landmarkers. Nothing leaves your computer.

## Quick start (from source)

```powershell
pip install -r requirements.txt
python sauron.py
```

A webcam window opens with a live overlay. Sit roughly where you normally
sit; Sauron only tracks the largest face in frame and ignores people in the
background.

## Building the .exe

```powershell
pip install -r requirements-dev.txt
.\build.ps1          # lint + tests + PyInstaller -> dist\Sauron.exe
```

`Sauron.exe` is self-contained (models and sounds are bundled). It runs
without a console; look in `%APPDATA%\Sauron\sauron.log` if something goes
wrong. Only one instance can run at a time.

To start it with Windows, put a shortcut to `Sauron.exe` in
`shell:startup` (Win+R, type `shell:startup`).

## Controls

Clickable buttons sit at the top right of the webcam window (mute, camera,
volume slider). Keyboard, with the webcam window focused:

| Key       | Action                                                   |
|-----------|----------------------------------------------------------|
| `q`, Esc  | Quit (closing the window with the X also quits)          |
| `m`       | Mute / unmute the warning sound                          |
| `c`       | Camera off / on. Off really releases the webcam.         |
| `s`       | Snooze detection (default 5 min); press again to cancel  |
| `-`, `+`  | Volume down / up                                         |

The warning popup has no dismiss button on purpose. It closes by itself
once your hand has been away from your mouth for a second (and the popup
has been up for at least two seconds).

## How detection works

Precision is preferred over recall: a missed bite is annoying, a false
alarm while you eat lunch is worse.

1. **Only you.** The largest face is taken as the user, and only if it
   spans at least 12% of the frame width. Hands must be size-consistent
   with that face, so someone else's hand cannot trigger it.
2. **Fingertips at the mouth.** Any of the five fingertips within ~a third
   of the face width from the mouth centre counts as "near".
3. **Dwell time.** "Near" must accumulate for 1.5 s before an alert. Time
   away decays the counter twice as fast, so a quick scratch never adds up.
4. **Eating suppression.** If the mouth opens wide (taking a bite of food)
   detection pauses for 4 s. Nail biting happens with lips barely parted.
5. **Face memory.** If the face tracker drops out for under a second while
   a hand is at the mouth, the last mouth position is kept so the dwell
   timer does not reset mid-bite.

All of these numbers are tunable (see below). The overlay shows the mouth
threshold circle (grey = idle, amber = near, red = counting) and a dwell
bar so you can see what the detector is thinking.

## Configuration

On first run Sauron writes `%APPDATA%\Sauron\config.json` with the
defaults. Edit it and restart; unknown keys or bad values are logged and
ignored, so a typo never silently disables alerts.

| Key                        | Default | Meaning                                             |
|----------------------------|---------|-----------------------------------------------------|
| `camera_index`             | 0       | Which webcam to use                                 |
| `camera_width/height`      | 640x480 | Requested capture size                              |
| `volume`                   | 0.5     | Initial warning volume (0 to 1)                     |
| `start_muted`              | false   | Start with sound muted                              |
| `fingertip_thr_face_frac`  | 0.32    | Mouth radius as a fraction of face width            |
| `fingertip_thr_min_px`     | 25      | Lower bound of that radius in pixels                |
| `bite_dwell_seconds`       | 1.5     | Sustained proximity before an alert                 |
| `clear_seconds`            | 1.0     | Hand must stay away this long to clear              |
| `min_alert_seconds`        | 2.0     | Minimum time the popup stays up                     |
| `min_face_width_frac`      | 0.12    | Faces narrower than this are background people      |
| `hand_face_ratio_min/max`  | 0.3/1.8 | Accepted hand-span : face-width band                |
| `mouth_open_ratio`         | 0.40    | Lip gap / mouth width that counts as eating         |
| `eating_suppress_seconds`  | 4.0     | Pause after the mouth was wide open                 |
| `face_memory_seconds`      | 1.0     | Bridge face-tracking dropouts for this long         |
| `snooze_minutes`           | 5       | Length of the `s` snooze                            |
| `show_phrases`             | true    | Show a random line from `phrases.txt` on the popup  |
| `log_level`                | INFO    | DEBUG adds a detector trace line every 0.5 s        |

Other files in `%APPDATA%\Sauron`:

- `stats.json` - bite count per day. Never deleted automatically.
- `phrases.txt` - drop a copy here to override the bundled popup phrases
  (one per line, `#` comments allowed).
- `sauron.log` - rotating diagnostic log.

## Project layout

```
sauron.py       entry point: camera loop, hotkeys, glue
detector.py     pure detection logic (face choice, proximity, dwell state machine)
config.py       settings file, phrases, daily stats
alerts.py       looping sound (pygame) and fullscreen popup (tkinter)
hud.py          OpenCV overlay and clickable controls
tests/          pytest suite for detector.py and config.py (no camera needed)
Sauron.spec     PyInstaller spec; build.ps1 drives it
*.task          MediaPipe models (hand, face)
*.mp3           warning sounds
```

## Development

```powershell
pip install -r requirements-dev.txt
python -m pytest        # unit tests
python -m ruff check .  # lint
```

`detector.py` and `config.py` import neither OpenCV nor MediaPipe, so the
detection rules can be tested and tuned with synthetic landmarks.
Ideas for making the warnings harder to ignore live in
`possible-improvements.md`.
