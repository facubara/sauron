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

11. ~~**Typing challenge gate** — Require typing a randomly-chosen LOTR phrase to dismiss the popup. The popup persists until the phrase is typed correctly. Active friction beats passive friction — you can't ignore something you're forced to interact with.~~ ✅

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
