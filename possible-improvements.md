# Possible Improvements

Ideas to make Sauron's warnings harder to ignore, ordered from easy to nuclear.

1. ~~**Dismiss-on-compliance, not timer** — Keep the popup open until hands actually leave the mouth, instead of auto-closing after 2.5s. Right now you can just wait it out.~~ ✅

2. ~~**Loop the sound** — Play the warning on repeat until hands move away, not just once.~~ ✅

3. **Escalating aggression** — Track repeat offenses. First warning is gentle, but if you bite again within 30s, make it louder, fully opaque, or swap to a more obnoxious sound.

4. **Screen flash/strobe** — Rapidly flash the popup on and off (red/black) a few times. Much harder to tune out than a static overlay.

5. **Steal focus** — Minimize all windows or bring the popup to the absolute foreground. Currently a fullscreen app could cover it.

6. ~~**Violation counter on the HUD** — Show a running tally of how many times you've been caught today. Guilt is a motivator.~~ ✅

7. **Screenshot "hall of shame"** — Snap a photo of you mid-bite and save it. Knowing it's being documented changes behavior.

8. **Windows toast notification** — Persists in the action center even after dismissal, so there's a log you can't escape.

9. **TTS voice** — Use `pyttsx3` or similar to say "Stop biting your nails" out loud. A human voice is harder to ignore than a sound effect.

10. **Cursor hijack** — Move the mouse to the center of the screen during a warning so you can't keep working through it.

11. **Typing challenge gate** — Require typing a randomly-chosen LOTR phrase to dismiss the popup. The popup persists until the phrase is typed correctly. Active friction beats passive friction — you can't ignore something you're forced to interact with. *Implemented in May 2026 (commit 4d4924c), removed again in the June rewrite along with the pose fallback. `phrases.txt` survived: since October 2026 the popup shows a random line from it as flavour text, and `%APPDATA%\Sauron\phrases.txt` still overrides the bundled list. The challenge could be re-added behind a `config.json` flag if wanted.*

## Workshop: extensions to the typing challenge

12. **Phrase length scales with violations** — first bite of the day = 30-char phrase, 5th = 80-char phrase. Combines naturally with idea #3 (escalating aggression).

13. **No-look mode** — require the camera to detect your face during typing; if you look away from the screen, progress freezes. Forces engagement with the warning instead of typing-by-feel while watching something else.

14. **Block bypass keys** — capture and swallow `Alt+F4`, `Win`, `Ctrl+Esc`, `Ctrl+Shift+Esc` in the popup. Right now you might be able to alt-tab the alert into the background.

15. **Cumulative typing penalty** — each bite adds a permanent +10 chars to your "next phrase length" until you go a full hour clean. Long streaks get monstrously long phrases.

16. **Backspace forbidden** — strict mode where you can't fix typos at all; one mistake and the entire phrase resets. Maybe gate this behind violation count > 10 so the early offences stay merciful.

17. **TTS reads the phrase** — pyttsx3 reads each word out loud as you type it. Combines with idea #9 (TTS voice). Bonus: pick a Saruman-ish voice config.

18. **Hall-of-shame integration** — screenshot the moment the popup opens (likely still mid-bite), save with timestamp + which phrase you got. Combines with idea #7 (screenshot photos).

19. **Random capitals** — randomly capitalise letters in the phrase to force shift-key engagement. Harder to autopilot through.

20. **Cooldown timer** — even after passing the challenge, the popup stays for an extra N seconds with a "phrase passed, hold steady" message. Prevents the dopamine of dismissal from being the immediate reward of biting.


## Done in the October 2026 pass (not warning-related, but worth knowing)

- **Snooze** (`s` key, length in `config.json`) for meals and calls. Snooze blocks new alerts but still lets an active one clear.
- **Hotkeys** for mute, camera, volume and quit; closing the webcam window with the X now actually quits instead of respawning the window.
- **Camera off really releases the webcam** (LED goes off) instead of silently still reading frames.
- **`config.json`** in `%APPDATA%\Sauron` exposes every threshold, so tuning no longer needs a rebuild. Bad values are logged and ignored.
- **Per-day history** in `stats.json` plus a clean-day streak on the HUD.
- **Face memory**: a brief face-tracking dropout while the hand covers the mouth no longer resets the dwell timer.
- **Rotating log file** and native error dialogs, since the windowed exe has no console. Single-instance guard so two copies cannot fight over the webcam.
- Detection logic split into `detector.py` / `config.py` with a pytest suite, so the threshold rules can be changed with a safety net.

## Still open, roughly in order of payoff

- **#3 Escalating aggression** is now easy: `stats.json` holds per-day counts and the alert timestamps are in the log, so "second bite within N minutes" is a few lines in `sauron.py` around the `ALERT` event, and the popup could take an `intensity` argument.
- **#8 Windows toast** and **#9 TTS** are self-contained additions to `alerts.py`.
- **#7 Hall of shame**: the frame is already in hand at the `ALERT` event; `cv2.imwrite` into `%APPDATA%\Sauron\shame\` is one line, plus a config flag and a retention cap.
