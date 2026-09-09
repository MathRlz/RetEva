"""Shared OpenAI-compatible LLM client used by judge, answer_generation, and query_optimization."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from collections import OrderedDict
from typing import Any, Dict, List

from ..config.llm_backend import LLMConfig
from ..logging_config import get_logger
from .cost import COST

logger = get_logger(__name__)

_CACHE_MAXSIZE = 1024
_cache: "OrderedDict[str, str]" = OrderedDict()

#: Backoff before retry attempt N (capped at the last entry).
_RETRY_BACKOFF_S = (0.5, 2.0)


def _is_transient(exc: Exception) -> bool:
    """Retryable only: timeouts, connection errors, HTTP 429/5xx; auth/schema 4xx fail
    immediately (retrying can't help)."""
    import socket
    import urllib.error

    if isinstance(exc, urllib.error.HTTPError):  # subclass of URLError — check first
        return exc.code == 429 or exc.code >= 500
    return isinstance(exc, (urllib.error.URLError, socket.timeout, TimeoutError))


def _explain(exc: Exception, cfg: Any) -> str:
    """The failure plus what the SERVER said — a bare exception (e.g. "HTTP Error 500" or
    "timed out") hides the real cause; a local server under memory pressure names it in the
    body (often an Ollama VRAM/KV-cache issue fixed via `OLLAMA_CONTEXT_LENGTH`, not the
    client)."""
    import urllib.error

    detail = str(exc)
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", "replace").strip()
            if body:
                detail = f"{detail} — server said: {body[:300]}"
        except Exception:  # noqa: BLE001 - the body is best-effort context
            pass
    workers = int(getattr(cfg, "concurrency", 1) or 1)
    if workers > 1:
        detail += (
            f" [concurrency={workers}: each slot reserves its own context, so a local server "
            f"can exhaust VRAM and 5xx or stall. Check `ollama ps` (PROCESSOR should be "
            f"100% GPU) and cap it with OLLAMA_CONTEXT_LENGTH, or lower concurrency]"
        )
    return detail


def _cache_key(messages: List[Dict[str, str]], model: str, temperature: float) -> str:
    raw = json.dumps(
        {"messages": messages, "model": model, "temperature": temperature},
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


class LLMClient:
    """OpenAI-compatible chat completions client; ``config`` may be any object with
    the same attributes as :class:`LLMConfig`."""

    def __init__(self, config: LLMConfig, *, component: str = "llm") -> None:
        self._config = config
        self._component = component  # cost-accounting label (judge / answer_gen / …)

    def call(
        self,
        messages: List[Dict[str, str]],
        *,
        use_cache: bool = False,
    ) -> str:
        """POST messages to the configured endpoint, return assistant content; failures
        (HTTP, JSON, timeout) surface as ``RuntimeError``."""
        cfg = self._config
        model = cfg.model
        temperature = cfg.temperature

        if use_cache:
            key = _cache_key(messages, model, temperature)
            if key in _cache:
                return _cache[key]

        api_key = os.getenv(cfg.api_key_env, "")
        body = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
        }
        seed = getattr(cfg, "seed", None)
        if seed is not None:
            body["seed"] = int(seed)  # omitted when unset: unknown keys upset some servers
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            cfg.get_api_base(),
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        _t0 = time.perf_counter()
        retries = int(getattr(cfg, "max_retries", 2) or 0)
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                break
            except Exception as exc:
                if attempt < retries and _is_transient(exc):
                    delay = _RETRY_BACKOFF_S[min(attempt, len(_RETRY_BACKOFF_S) - 1)]
                    logger.warning(
                        "transient LLM failure (%s) — retry %d/%d in %.1fs: %s",
                        self._component, attempt + 1, retries, delay, exc,
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"LLM call failed ({cfg.get_api_base()}): {_explain(exc, cfg)}"
                ) from exc

        if "error" in body:
            raise RuntimeError(f"LLM API returned error: {body['error']}")
        try:
            content = body["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected LLM response structure: {body}") from exc

        # Usage fields default to 0 when the server omits them (call count + latency still
        # count); may raise BudgetExceededError to stop the run.
        usage = body.get("usage") or {}
        COST.record(
            self._component,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            latency_s=time.perf_counter() - _t0,
        )

        if use_cache:
            _cache[key] = content
            if len(_cache) > _CACHE_MAXSIZE:
                _cache.popitem(last=False)
        return content
