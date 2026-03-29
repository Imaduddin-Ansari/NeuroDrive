"""
llm_risk_explainer.py
─────────────────────
Main orchestrator for the LLM-Based Risk Explanation module.

Architecture
────────────
  NeuroDrive tick loop  →  explainer.submit(snapshot)
                                │
                                ▼  (if cooldown elapsed AND trigger fires)
                         _work_queue  (size 1 — always newest snapshot)
                                │
                       LLM worker thread
                                │
                         LLMClient.generate()   ← Llama 3.2 (local)
                                │
                         _result_queue
                                │
  NeuroDrive tick loop  →  explainer.poll()  →  AlertPanel + TTS + Logger

The worker thread is the only thread that calls LLMClient, so HTTP
requests never touch the UI thread.  poll() is non-blocking and safe
to call every tick.
"""

from __future__ import annotations
import queue
import threading
import time
from typing import Optional

from .risk_snapshot import RiskSnapshot
from .llm_client    import LLMClient
from .tts_speaker   import TTSSpeaker
from .risk_logger   import RiskLogger


class LLMRiskExplainer:
    """
    Public interface used by NeuroDriveUI.

    Parameters
    ──────────
    backend      "ollama"  or  "llamacpp"
    base_url     URL of the local inference server
    model        model name (Ollama) or ignored (llama.cpp)
    cooldown     minimum seconds between consecutive LLM calls
    timeout      HTTP timeout for each LLM request
    tts_enabled  whether to speak explanations aloud
    log_enabled  whether to write a JSON-lines risk log
    """

    def __init__(
        self,
        backend:     str   = "ollama",
        base_url:    str   = "http://localhost:11434",
        model:       str   = "llama3.2",
        cooldown:    float = 8.0,
        timeout:     float = 3.5,
        tts_enabled: bool  = True,
        log_enabled: bool  = True,
    ):
        self._cooldown  = cooldown
        self._last_call = 0.0
        self._running   = True

        # LLM HTTP client
        self._client = LLMClient(
            backend=backend, base_url=base_url,
            model=model, timeout=timeout,
        )

        # TTS speaker (daemon thread)
        self._speaker = TTSSpeaker(enabled=tts_enabled)

        # Risk event logger
        self._logger: Optional[RiskLogger] = RiskLogger() if log_enabled else None

        # Inter-thread queues
        # _work_queue  size 1 — only the newest snapshot matters
        # _result_queue unbounded — UI polls at its own rate
        self._work_queue:   queue.Queue[Optional[RiskSnapshot]] = queue.Queue(maxsize=1)
        self._result_queue: queue.Queue[str]                    = queue.Queue()

        # LLM worker thread
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="LLMRiskWorker"
        )
        self._worker.start()

        # Startup connectivity check (non-blocking, just a log message)
        threading.Thread(
            target=self._check_connectivity, daemon=True, name="LLMHealthCheck"
        ).start()

    # ── Public API ────────────────────────────────────────────────────────────

    def submit(self, snapshot: RiskSnapshot) -> None:
        """
        Offer a snapshot for LLM processing.

        Silently dropped when:
          • the module is not running
          • cooldown has not elapsed since the last call
          • the snapshot does not meet any trigger condition
        """
        if not self._running:
            return
        now = time.time()
        if now - self._last_call < self._cooldown:
            return
        if not snapshot.should_trigger():
            return

        # Evict any stale snapshot still waiting; keep only the newest
        try:
            self._work_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._work_queue.put_nowait(snapshot)
        except queue.Full:
            pass

    def poll(self) -> Optional[str]:
        """
        Return the next pending explanation (if any), or None.
        Safe to call every UI tick — never blocks.
        """
        try:
            return self._result_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        """Shut down all background threads cleanly."""
        self._running = False
        # Send sentinel to unblock the worker
        try:
            self._work_queue.put_nowait(None)
        except queue.Full:
            pass
        self._speaker.stop()
        if self._logger:
            self._logger.close()

    # ── Worker thread ─────────────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        while self._running:
            try:
                item = self._work_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:   # sentinel
                break

            snapshot: RiskSnapshot = item
            self._last_call = time.time()

            print(f"[LLMRisk] Generating explanation "
                  f"(alerts={snapshot.active_alert_count()}) …")

            explanation = self._client.generate(snapshot.to_prompt())

            if explanation:
                print(f"[LLMRisk] → {explanation}")
                self._result_queue.put(explanation)
                self._speaker.speak(explanation)
                if self._logger:
                    self._logger.log(snapshot, explanation)
            else:
                print("[LLMRisk] No explanation returned (timeout or error).")
                if self._logger:
                    self._logger.log(snapshot, None)

    # ── Health check ──────────────────────────────────────────────────────────

    def _check_connectivity(self) -> None:
        if self._client.is_reachable():
            print("[LLMRisk] ✓ Local model server reachable.")
        else:
            print(
                "[LLMRisk] ✗ Cannot reach local model server. "
                "Start Ollama with: ollama serve"
            )