"""
tts_speaker.py
──────────────
Non-blocking text-to-speech — platform-aware, CPU-efficient.

Backend selection (chosen once at startup):
  macOS   → subprocess `say`
             pyttsx3 on macOS calls pyttsx3.init() which opens a brand-new
             AVAudioSession every time, causing coreaudiod to spike to 40%+
             CPU with thousands of idle wake-ups per minute.  The built-in
             `say` command reuses the system TTS daemon and costs ~0% CPU.

  Windows → pyttsx3 with a fresh engine per utterance.
             Reusing one engine instance causes SAPI5's COM event loop to
             silently return from runAndWait() after the first utterance,
             producing no audio.  A fresh init() per call works correctly.

  Linux   → espeak if present, else pyttsx3, else festival.

Threading model
───────────────
  One daemon worker thread owns all audio I/O.
  _is_speaking + _pending implement an interrupt-safe one-slot buffer:
    - The current utterance always plays to completion — never cut off.
    - If a new message arrives while speaking, it is stored in _pending,
      replacing any previously stored pending message (newest-wins).
    - When the current utterance finishes, _pending is dispatched next.
    - The queue therefore never holds more than 1 waiting item.
"""

from __future__ import annotations
import queue
import subprocess
import sys
import threading
from typing import Optional

_ON_MACOS   = sys.platform == "darwin"
_ON_WINDOWS = sys.platform == "win32"


class TTSSpeaker:
    """
    Interrupt-safe, CPU-efficient fire-and-forget TTS wrapper.

    Usage:
        speaker = TTSSpeaker(rate=165, volume=1.0)
        speaker.speak("Brake now — vehicle 5 metres ahead.")
        ...
        speaker.stop()
    """

    def __init__(self, rate: int = 165, volume: float = 1.0, enabled: bool = True):
        self.enabled      = enabled
        self._rate        = rate
        self._volume      = volume

        # Worker queue — maxsize 2 gives the sentinel a guaranteed slot
        # even when a message is in flight.
        self._queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=2)

        # True while the backend is actively synthesising speech.
        self._is_speaking = False
        # Lock protecting _is_speaking and _pending together.
        self._speak_lock  = threading.Lock()
        # One-slot buffer for messages that arrive during synthesis.
        self._pending: Optional[str] = None

        self._thread = threading.Thread(
            target=self._run, daemon=True, name="TTSSpeaker"
        )
        self._thread.start()

    # ── Public API ────────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """
        Request *text* to be spoken.

        If the engine is idle  -> dispatched immediately (no latency).
        If the engine is busy  -> stored as _pending (newest-wins).
                                  The current utterance is NOT interrupted.
        """
        if not self.enabled or not text:
            return

        with self._speak_lock:
            if self._is_speaking:
                self._pending = text   # replace any stale pending message
                return

        # Engine idle — deliver directly to the worker.
        try:
            self._queue.get_nowait()   # evict any leftover item
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass   # defensive; should not happen after the get above

    @property
    def is_alive(self) -> bool:
        """True if the background thread is still running."""
        return self._thread.is_alive()

    def stop(self) -> None:
        """Shut down the background thread gracefully."""
        with self._speak_lock:
            self._pending = None
        # Ensure the sentinel lands even if both queue slots are full.
        for _ in range(3):
            try:
                self._queue.put_nowait(None)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass

    # ── Backend implementations ───────────────────────────────────────────────

    def _speak_macos(self, text: str) -> None:
        """
        macOS: use the built-in `say` command.

        `say` delegates to speechsynthesisd which keeps a persistent audio
        session — no AVAudioSession open/close churn, coreaudiod stays idle.
        Rate is passed as words-per-minute (say counts differently from
        pyttsx3, so we scale by 1.15 as a calibration factor).
        """
        wpm = max(100, min(300, int(self._rate * 1.15)))
        try:
            subprocess.run(
                ["say", "-r", str(wpm), text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60,
            )
        except FileNotFoundError:
            # `say` not found — fall back to pyttsx3
            self._speak_pyttsx3(text)
        except subprocess.TimeoutExpired:
            pass
        except Exception as exc:
            print(f"[TTSSpeaker] say error: {exc}")

    def _speak_pyttsx3(self, text: str) -> None:
        """
        Windows / Linux: pyttsx3 with a fresh engine per utterance.

        A fresh pyttsx3.init() is used every time because SAPI5 (Windows)
        and some ALSA/PulseAudio drivers leave the engine's internal event
        loop in a broken state after the first runAndWait(), causing all
        subsequent calls to return silently with no audio.
        """
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate",   self._rate)
            engine.setProperty("volume", self._volume)
            engine.say(text)
            engine.runAndWait()
            try:
                engine.stop()     # release COM apartment / audio device
            except Exception:
                pass
        except Exception as exc:
            print(f"[TTSSpeaker] pyttsx3 error: {exc}")
            self._speak_espeak(text)   # last-resort Linux fallback

    def _speak_espeak(self, text: str) -> None:
        """Linux last-resort: espeak or festival."""
        try:
            r = subprocess.run(["which", "espeak"], capture_output=True)
            if r.returncode == 0:
                subprocess.run(
                    ["espeak", "-s", str(self._rate), text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=60,
                )
            else:
                r2 = subprocess.run(["which", "festival"], capture_output=True)
                if r2.returncode == 0:
                    subprocess.run(
                        ["festival", "--tts"],
                        input=text.encode(),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=60,
                    )
        except Exception as exc:
            print(f"[TTSSpeaker] espeak/festival error: {exc}")

    # ── Worker thread ─────────────────────────────────────────────────────────

    def _run(self) -> None:
        while True:
            text = self._queue.get()   # blocks until message or sentinel
            if text is None:
                return                 # shutdown sentinel

            with self._speak_lock:
                self._is_speaking = True
                self._pending     = None

            try:
                if _ON_MACOS:
                    self._speak_macos(text)
                elif _ON_WINDOWS:
                    self._speak_pyttsx3(text)
                else:
                    # Linux: prefer espeak over pyttsx3
                    try:
                        result = subprocess.run(
                            ["which", "espeak"], capture_output=True)
                        if result.returncode == 0:
                            self._speak_espeak(text)
                        else:
                            self._speak_pyttsx3(text)
                    except Exception:
                        self._speak_pyttsx3(text)

            finally:
                with self._speak_lock:
                    self._is_speaking = False
                    next_text = self._pending
                    self._pending = None

                # Dispatch the buffered message (if any) for the next cycle.
                if next_text is not None:
                    try:
                        self._queue.put_nowait(next_text)
                    except queue.Full:
                        try:
                            self._queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self._queue.put_nowait(next_text)
                        except queue.Full:
                            pass