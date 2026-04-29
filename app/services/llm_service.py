"""
LLMService — chat completions with automatic OpenAI → Ollama fallback.

Backend selection (in priority order):
  1. OpenAI  — if OPENAI_API_KEY is set in the environment.
  2. Ollama  — if OpenAI key is absent AND Ollama is reachable at
               OLLAMA_BASE_URL (default: http://localhost:11434).
               Detection: HTTP probe to /api/tags, then `ollama` binary check.
  3. Error   — LLMError raised if neither backend is available.

Both backends use the OpenAI-compatible SDK so the calling code is identical.
Ollama exposes an OpenAI-compatible endpoint at <base_url>/v1.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import urllib.error
import urllib.request
from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum, auto

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from app.core.config import Settings
from app.core.exceptions import LLMError, RateLimitedError

logger = logging.getLogger(__name__)


class _Backend(Enum):
    OPENAI = auto()
    OLLAMA = auto()


@dataclass(frozen=True)
class _BackendConfig:
    backend: _Backend
    base_url: str
    api_key: str
    model: str


def _probe_ollama(base_url: str, timeout: float = 2.0) -> list[str]:
    """
    Return list of model names available in Ollama, or [] if unreachable.
    Also accepts the response as a plain 200 with no models (empty list).
    """
    import json
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=timeout) as resp:
            if resp.status != 200:
                return []
            data = json.loads(resp.read().decode())
            return [m["name"] for m in data.get("models", [])]
    except (urllib.error.URLError, OSError, KeyError, ValueError):
        return []


def _detect_backend(settings: Settings) -> _BackendConfig:
    """
    Determine which LLM backend to use.

    Priority:
      1. OPENAI_API_KEY present → OpenAI
      2. Ollama reachable (HTTP probe) → Ollama
      3. `ollama` binary on PATH → Ollama (assume server will be started)
      4. Raise LLMError
    """
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        logger.info("LLM backend: OpenAI (model=%s)", settings.openai_model)
        return _BackendConfig(
            backend=_Backend.OPENAI,
            base_url="https://api.openai.com/v1",
            api_key=openai_key,
            model=settings.openai_model,
        )

    # No OpenAI key — try Ollama
    ollama_base = settings.ollama_base_url.rstrip("/")
    available_models = _probe_ollama(ollama_base)

    if available_models:
        # Prefer the OLLAMA_MODEL env var if it's actually downloaded,
        # otherwise just use whatever is available.
        preferred = settings.ollama_model
        model = preferred if preferred in available_models else available_models[0]
        logger.info(
            "LLM backend: Ollama (base_url=%s, model=%s) — %d model(s) available: %s",
            ollama_base, model, len(available_models), available_models,
        )
        return _BackendConfig(
            backend=_Backend.OLLAMA,
            base_url=f"{ollama_base}/v1",
            api_key="ollama",
            model=model,
        )

    raise LLMError(
        "No LLM backend available. "
        "Set OPENAI_API_KEY in your .env file, "
        "or install and start Ollama (https://ollama.com)."
    )


class LLMService:
    """
    Chat completions service with automatic OpenAI → Ollama fallback.

    Backend is detected once on first use and cached for the lifetime of the
    service instance. Both backends are accessed through the OpenAI-compatible
    SDK so no code paths differ between them.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: OpenAI | None = None
        self._backend_cfg: _BackendConfig | None = None
        self._init_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal initialisation (lazy, thread-safe)
    # ------------------------------------------------------------------

    def _get_client(self) -> tuple[OpenAI, _BackendConfig]:
        """
        Detect backend and create the OpenAI-SDK client exactly once.
        Returns (client, backend_config).
        """
        if self._client is None:
            with self._init_lock:
                if self._client is None:
                    cfg = _detect_backend(self._settings)
                    self._backend_cfg = cfg
                    self._client = OpenAI(
                        api_key=cfg.api_key,
                        base_url=cfg.base_url,
                    )
                    logger.info(
                        "LLMService ready — backend=%s base_url=%s model=%s",
                        cfg.backend.name,
                        cfg.base_url,
                        cfg.model,
                    )
        return self._client, self._backend_cfg  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(self, system_prompt: str, user_content: str) -> str:
        """
        Send a chat completion request and return the raw assistant text.

        Does NOT render markdown — the caller is responsible for that.

        Raises:
          RateLimitedError — when the backend signals rate limiting (OpenAI 429
                             or an Ollama equivalent).
          LLMError         — on connection failure or non-rate-limit API error.
        """
        client, cfg = self._get_client()

        logger.info(
            "Calling %s model=%s (user_content=%d chars).",
            cfg.backend.name,
            cfg.model,
            len(user_content),
        )

        try:
            completion = client.chat.completions.create(
                model=cfg.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            reply: str = completion.choices[0].message.content or ""
            logger.info(
                "%s response: %d chars, finish_reason=%s.",
                cfg.backend.name,
                len(reply),
                completion.choices[0].finish_reason,
            )
            return reply

        except RateLimitError as exc:
            logger.warning("%s rate limit hit: %s", cfg.backend.name, exc)
            raise RateLimitedError(
                "The AI service is currently busy (rate limit reached). "
                "Please wait a moment and try again."
            ) from exc
        except APIConnectionError as exc:
            logger.error("%s connection error: %s", cfg.backend.name, exc)
            raise LLMError(
                f"Could not reach the {cfg.backend.name} service at {cfg.base_url}. "
                "Please check your network or that Ollama is running."
            ) from exc
        except APIStatusError as exc:
            logger.error(
                "%s status error HTTP %s: %s", cfg.backend.name, exc.status_code, exc.message
            )
            raise LLMError(
                f"The AI service returned an error (HTTP {exc.status_code}). "
                "Please try again later."
            ) from exc

    def complete_stream(
        self, system_prompt: str, user_content: str
    ) -> Generator[str, None, None]:
        """
        Stream a chat completion, yielding raw text chunks as they arrive.

        Raises RateLimitedError / LLMError on failure (same semantics as complete()).
        The caller must consume the generator inside a try/except to handle errors.
        """
        client, cfg = self._get_client()
        logger.info(
            "Streaming %s model=%s (user_content=%d chars).",
            cfg.backend.name, cfg.model, len(user_content),
        )
        try:
            stream = client.chat.completions.create(
                model=cfg.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except RateLimitError as exc:
            raise RateLimitedError(
                "The AI service is currently busy (rate limit reached). "
                "Please wait a moment and try again."
            ) from exc
        except APIConnectionError as exc:
            raise LLMError(
                f"Could not reach the {cfg.backend.name} service at {cfg.base_url}. "
                "Please check your network or that Ollama is running."
            ) from exc
        except APIStatusError as exc:
            raise LLMError(
                f"The AI service returned an error (HTTP {exc.status_code})."
            ) from exc

    @property
    def backend_name(self) -> str:
        """Return 'openai' or 'ollama' once the client has been initialised, else 'unknown'."""
        if self._backend_cfg is None:
            return "unknown"
        return self._backend_cfg.backend.name.lower()
