"""Available-LLM listing for the builder's model picker (the global graph LLM).

Two endpoint kinds:

* ``local`` — the curated local catalog (:class:`LLMModelCatalog`) merged with a *live* probe of
  a running Ollama server (``/api/tags``), so models you've actually ``ollama pull``-ed are
  flagged ``pulled`` and float to the top. Any pulled model missing from the catalog is added.
* ``api`` — a small static set of common OpenAI-compatible chat models (the catalog is
  local-only; API model ids are different).

Pure-ish: the only side effect is a 2s best-effort HTTP GET to Ollama, which degrades to an
empty pull-set when the server is down.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from .registry import LLMModelCatalog

# Common OpenAI-compatible chat models for the `api` endpoint (the curated catalog is local).
_API_MODELS: List[Dict[str, str]] = [
    {"id": "gpt-4o-mini", "label": "GPT-4o mini"},
    {"id": "gpt-4o", "label": "GPT-4o"},
    {"id": "gpt-4-turbo", "label": "GPT-4 Turbo"},
    {"id": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo"},
    {"id": "o1-mini", "label": "o1-mini"},
]


def _ollama_base(url: Optional[str]) -> str:
    """The Ollama root from a configured URL (strip an OpenAI-compat/`/api` suffix) or env."""
    base = url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    return base.split("/v1/")[0].split("/api/")[0].rstrip("/")


def _ollama_probe(base: str) -> Tuple[bool, List[str]]:
    """``(running, pulled_model_names)`` from Ollama ``/api/tags`` — best effort, 2s timeout."""
    try:
        import requests

        r = requests.get(f"{base}/api/tags", timeout=2)
        if r.status_code == 200:
            names = [m.get("name", "") for m in r.json().get("models", [])]
            return True, [n for n in names if n]
    except Exception:  # noqa: BLE001 — server down / no requests / bad json → not running
        pass
    return False, []


def list_llm_models(
    endpoint: str = "local", ollama_url: Optional[str] = None
) -> Dict[str, Any]:
    """Models for the LLM picker, for the chosen endpoint kind (``local`` / ``api``).

    Returns ``{endpoint, running, models:[{id, label, pulled?, domain?, size?, min_ram_gb?}]}``;
    ``running`` is the Ollama reachability for ``local`` (``None`` for ``api``)."""
    if endpoint == "api":
        return {"endpoint": "api", "running": None, "models": list(_API_MODELS)}

    base = _ollama_base(ollama_url)
    running, pulled_names = _ollama_probe(base)
    pulled = set(pulled_names)

    models: List[Dict[str, Any]] = []
    seen: set = set()
    for m in LLMModelCatalog.to_dict_list():
        mid = m.get("ollama_name") or m.get("name")
        if not mid or mid in seen:
            continue
        seen.add(mid)
        models.append(
            {
                "id": mid,
                "label": m["display_name"],
                "domain": m["domain"],
                "size": m["size"],
                "min_ram_gb": m.get("min_ram_gb"),
                "pulled": mid in pulled,
            }
        )
    # pulled models not in the curated catalog (custom pulls) — surface them too
    for name in sorted(pulled):
        if name not in seen:
            models.append({"id": name, "label": name, "pulled": True})

    # pulled first, then alphabetical — so what you can actually run is at the top
    models.sort(key=lambda x: (not x.get("pulled"), str(x["label"]).lower()))
    return {"endpoint": "local", "running": running, "models": models}
