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
