# Sauron

Webcam nail-biting detector for Windows. Python 3.13, MediaPipe tasks API,
OpenCV window + tkinter popup + pygame audio, shipped as a one-file
PyInstaller exe.

## Commands

- Run: `python sauron.py`
- Tests: `python -m pytest` (pure-logic tests only; no camera or display)
- Lint: `python -m ruff check .`
- Build exe: `.\build.ps1` (lint, tests, then PyInstaller via `Sauron.spec`)

## Releases

Every push to `main` (except docs-only changes) runs
`.github/workflows/release.yml`: lint, tests, stamps `version.py` with
`BASE_VERSION.<run number>`, builds and publishes a GitHub release. The
installed copy (`%LOCALAPPDATA%\Programs\Sauron`) picks it up via
`updater.py` and restarts when idle. Consequences:

- A push to `main` ships to the user within hours. Keep `main` working.
- Do not commit a stamped `version.py`; the committed one stays `-dev`.
- Bump `BASE_VERSION` in the workflow for notable changes.
- Changing the asset name, repo, or `updater.py` swap logic can strand
  installed copies on an old version. Test the frozen update path first.

## Structure

- `detector.py` and `config.py` must stay free of cv2 / mediapipe / pygame
  imports so they remain unit-testable. Put new detection rules there, not
  in `sauron.py`.
- Every tunable belongs on `config.Settings` (and in the README table), so
  users can change it in `%APPDATA%\Sauron\config.json` without rebuilding.
- Files bundled into the exe are listed in `Sauron.spec` `datas`; add new
  assets there and load them via `config.asset_path()`.
- All Tk calls happen on the popup thread (`alerts.WarningPopup`); other
  threads only flip flags.

## Design rule

Precision over recall. A change that catches more bites but adds false
alarms while eating, drinking or scratching is a regression.
