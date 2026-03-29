"""
llm_client.py
─────────────
Thin HTTP wrapper for a locally running Llama model.

Supported backends
  "ollama"   — Ollama server  (default: http://localhost:11434)
               POST /api/generate
  "llamacpp" — llama.cpp server (default: http://localhost:8080)
               POST /completion

Both backends are called with streaming disabled so we get one
clean JSON response.  The call is intentionally fire-and-forget
from the caller's perspective; exceptions are caught and logged
so a LLM failure never crashes the ADAS pipeline.
"""

from __future__ import annotations
from typing import Optional
import requests


class LLMClient:
    """
    Synchronous HTTP client for a local Llama inference server.

    This class is called from a background thread inside
    LLMRiskExplainer, so every method must be thread-safe and
    must not touch any Tkinter widgets.
    """

    # Tokens to generate.  60 is enough for two sentences;
    # staying low keeps latency under ~2 s on a 3 B model.
    MAX_TOKENS   = 60
    TEMPERATURE  = 0.25   # low = deterministic, action-focused
    STOP_TOKENS  = ["\n\n", "###", "\n3.", "\n4."]   # stop after 2 sentences

    def __init__(
        self,
        backend:  str = "ollama",
        base_url: str = "http://localhost:11434",
        model:    str = "llama3.2",
        timeout:  float = 3.5,
    ):
        self.backend  = backend.lower()
        self.model    = model
        self.timeout  = timeout

        # Normalise trailing slash
        self.base_url = base_url.rstrip("/")

        if self.backend == "ollama":
            self._endpoint = f"{self.base_url}/api/generate"
        elif self.backend == "llamacpp":
            self._endpoint = f"{self.base_url}/completion"
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. Use 'ollama' or 'llamacpp'."
            )

    # ─────────────────────────────────────────────────────────────────────────

    def generate(self, prompt: str) -> Optional[str]:
        """
        Send *prompt* to the local model and return the text reply.

        Returns None on any error (timeout, connection refused, bad JSON).
        """
        try:
            if self.backend == "ollama":
                return self._call_ollama(prompt)
            else:
                return self._call_llamacpp(prompt)
        except requests.exceptions.Timeout:
            print("[LLMClient] Request timed out — skipping explanation.")
            return None
        except requests.exceptions.ConnectionError:
            print("[LLMClient] Cannot reach local model server — is it running?")
            return None
        except Exception as exc:
            print(f"[LLMClient] Unexpected error: {exc}")
            return None

    def is_reachable(self) -> bool:
        """
        Quick health-check.  Returns True if the server responds to a
        trivial request within 2 seconds.  Used at startup to decide
        whether to show an offline warning.
        """
        try:
            if self.backend == "ollama":
                r = requests.get(
                    f"{self.base_url}/api/tags", timeout=2.0
                )
            else:
                r = requests.get(
                    f"{self.base_url}/health", timeout=2.0
                )
            return r.status_code < 500
        except Exception:
            return False

    # ── Private helpers ───────────────────────────────────────────────────────

    def _call_ollama(self, prompt: str) -> Optional[str]:
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": self.MAX_TOKENS,
                "temperature": self.TEMPERATURE,
                "stop":        self.STOP_TOKENS,
            },
        }
        r = requests.post(self._endpoint, json=payload, timeout=self.timeout)
        r.raise_for_status()
        text = r.json().get("response", "").strip()
        return text if text else None

    def _call_llamacpp(self, prompt: str) -> Optional[str]:
        payload = {
            "prompt":      prompt,
            "n_predict":   self.MAX_TOKENS,
            "temperature": self.TEMPERATURE,
            "stop":        self.STOP_TOKENS,
            "stream":      False,
        }
        r = requests.post(self._endpoint, json=payload, timeout=self.timeout)
        r.raise_for_status()
        text = r.json().get("content", "").strip()
        return text if text else None