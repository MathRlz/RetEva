"""Shared OpenAI-compatible LLM client used by judge, answer_generation, and query_optimization."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from collections import OrderedDict
from typing import Dict, List

from ..config.llm_backend import LLMConfig
from .cost import COST

_CACHE_MAXSIZE = 1024
_cache: "OrderedDict[str, str]" = OrderedDict()


def _cache_key(messages: List[Dict[str, str]], model: str, temperature: float) -> str:
    raw = json.dumps(
        {"messages": messages, "model": model, "temperature": temperature},
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


class LLMClient:
    """OpenAI-compatible chat completions client.

    Args:
        config: LLMConfig (or any object with the same attributes).
    """

    def __init__(self, config: LLMConfig, *, component: str = "llm") -> None:
        self._config = config
        self._component = component  # cost-accounting label (judge / answer_gen / …)

    def call(
        self,
        messages: List[Dict[str, str]],
        *,
        use_cache: bool = False,
    ) -> str:
        """POST messages to the configured endpoint, return assistant content.

        Args:
            messages: OpenAI-format message list.
            use_cache: If True, return cached result for identical requests.

        Raises:
            RuntimeError: On HTTP or JSON errors.
        """
        cfg = self._config
        model = cfg.model
        temperature = cfg.temperature

        if use_cache:
            key = _cache_key(messages, model, temperature)
            if key in _cache:
                return _cache[key]

        api_key = os.getenv(cfg.api_key_env, "")
        payload = json.dumps(
            {
                "model": model,
                "temperature": temperature,
                "messages": messages,
            }
        ).encode("utf-8")
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
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"LLM call failed ({cfg.get_api_base()}): {exc}"
            ) from exc

        if "error" in body:
            raise RuntimeError(f"LLM API returned error: {body['error']}")
        try:
            content = body["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected LLM response structure: {body}") from exc

        # Record token usage + latency for the run's cost report (T8). The usage block is
        # OpenAI-compatible; missing fields default to 0 (a server that omits them still
        # contributes latency + a call count). May raise BudgetExceededError to stop the run.
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
