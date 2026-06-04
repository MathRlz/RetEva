import json
import os
from typing import Dict, Optional

_BUILTIN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "term_weights")


def load_term_weights(domain: str, path: Optional[str] = None) -> Dict[str, float]:
    """Load term weights from a JSON file.

    Args:
        domain: Built-in domain name (e.g. ``"medical"``). Used when *path* is None.
        path: Explicit path to a JSON file mapping term → weight. Overrides *domain*.

    Returns:
        Dict mapping lowercase term to float weight.

    Raises:
        FileNotFoundError: When neither built-in domain file nor explicit path exists.
    """
    if path is not None:
        target = path
    else:
        target = os.path.join(_BUILTIN_DIR, f"{domain}.json")

    if not os.path.isfile(target):
        raise FileNotFoundError(f"Term weights file not found: {target}")

    with open(target, encoding="utf-8") as fh:
        data = json.load(fh)

    return {str(k).lower(): float(v) for k, v in data.items() if not str(k).startswith("_")}
