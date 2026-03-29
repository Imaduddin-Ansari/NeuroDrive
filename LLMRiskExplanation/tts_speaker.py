"""
tts_speaker.py
──────────────
Non-blocking text-to-speech using pyttsx3.

A dedicated daemon thread owns the pyttsx3 engine so we never
block the UI or the LLM worker.  Speech requests are queued;
if a new explanation arrives while the engine is still speaking,
the old utterance is replaced (we never queue up a backlog of
warnings that would play out long after they are relevant).
"""

from __future__ import annotations
import queue
import threading
from typing import Optional


class TTSSpeaker:
    """
    Fire-and-forget TTS wrapper.

    Usage:
        speaker = TTSSpeaker(rate=165, volume=1.0)
        speaker.speak("Brake now — vehicle 5 metres ahead.")
        ...
        speaker.stop()
    """

    def __init__(self, rate: int = 165, volume: float = 1.0, enabled: bool = True):
        self.enabled  = enabled
        self._rate    = rate
        self._volume  = volume
        self._queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=1)
        self._thread  = threading.Thread(
            target=self._run, daemon=True, name="TTSSpeaker"
        )
        self._thread.start()

    # ─────────────────────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """
        Queue *text* for speech.  If a previous utterance is waiting
        it is replaced so we always speak the most recent alert.
        """
        if not self.enabled or not text:
            return
        # Evict stale item — only keep latest
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass

    def stop(self) -> None:
        """Shut down the background thread gracefully."""
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    # ── Background thread ─────────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate",   self._rate)
            engine.setProperty("volume", self._volume)
        except Exception as exc:
            print(f"[TTSSpeaker] pyttsx3 init failed ({exc}). TTS disabled.")
            # Drain queue so the thread exits cleanly on stop()
            while True:
                item = self._queue.get()
                if item is None:
                    return
            return

        while True:
            text = self._queue.get()   # blocks until something arrives
            if text is None:           # sentinel — shut down
                return
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:
                print(f"[TTSSpeaker] Speech error: {exc}")