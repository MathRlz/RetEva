import json
import os
from functools import lru_cache
from typing import Any, Dict, Optional

_BUILTIN_DIR = os.path.join(
    os.path.dirname(__file__), "..", "resources", "term_weights"
)


def load_term_weights(domain: str, path: Optional[str] = None) -> Dict[str, float]:
    """Load term weights keyed by ``domain`` (built-in, e.g. ``"medical"``) unless ``path``
    is given (overrides ``domain``). Returns lowercase term -> weight."""
    if path is not None:
        target = path
    else:
        target = os.path.join(_BUILTIN_DIR, f"{domain}.json")

    if not os.path.isfile(target):
        raise FileNotFoundError(f"Term weights file not found: {target}")

    with open(target, encoding="utf-8") as fh:
        data = json.load(fh)

    return {
        str(k).lower(): float(v) for k, v in data.items() if not str(k).startswith("_")
    }


def _load_json(name: str, path: Optional[str]) -> Any:
    """A bundled ``resources/<name>`` JSON, or the explicit override ``path``."""
    target = path or os.path.join(_BUILTIN_DIR, "..", name)
    if not os.path.isfile(target):
        raise FileNotFoundError(f"Resource file not found: {target}")
    with open(target, encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=None)
def load_drug_terms(path: Optional[str] = None) -> tuple:
    """Canonical drug/term names for the phonetic corrector's vocabulary and the ``ceer_rx``
    critical-term extension. ``path`` overrides the bundled ``resources/drug_terms.json``
    (format: ``{"terms": [...]}``)."""
    terms = _load_json("drug_terms.json", path)["terms"]
    return tuple(str(t).lower() for t in terms)


@lru_cache(maxsize=None)
def load_drug_doses(path: Optional[str] = None) -> dict:
    """Plausible-dose table for unit-slip repair: ``{drug: {min, max, unit}}``.
    ``path`` overrides the bundled ``resources/drug_doses.json``."""
    doses = _load_json("drug_doses.json", path)["doses"]
    return {str(drug).lower(): spec for drug, spec in doses.items()}
