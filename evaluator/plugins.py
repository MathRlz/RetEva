"""Third-party plugin discovery via Python entry points (architecture-improvements §5).

A researcher's package can register a model / metric / node / handler / dataset *without
editing the core tree* by declaring entry points, e.g. in its ``pyproject.toml``::

    [project.entry-points."evaluator.models"]
    my_embedder = "my_pkg.models"      # importing the module runs its @register_* decorators

    [project.entry-points."evaluator.datasets"]
    my_dataset = "my_pkg.datasets"

``load_plugins(group)`` imports each entry point's module (which self-registers via the
existing decorators), idempotently per group. Best-effort: a broken plugin is logged, never
raised — a third-party bug must not take down the core. Discovery is wired into each
registry's lazy-population point, so plugins load exactly when that registry is first used.
"""

from __future__ import annotations

from typing import List

from .logging_config import get_logger

logger = get_logger(__name__)

# Entry-point groups, one per extensible kind.
GROUPS = (
    "evaluator.models",
    "evaluator.nodes",
    "evaluator.handlers",
    "evaluator.metrics",
    "evaluator.datasets",
)

_LOADED_GROUPS: set = set()


def load_plugins(group: str) -> List[str]:
    """Import every entry point in ``group`` (idempotent per group); return the names loaded.

    Each module's ``@register_*`` decorators run on import. A plugin that fails to import is
    logged at WARNING and skipped.
    """
    if group in _LOADED_GROUPS:
        return []
    _LOADED_GROUPS.add(group)

    try:
        from importlib.metadata import entry_points

        eps = entry_points()
        selected = (
            eps.select(group=group) if hasattr(eps, "select") else eps.get(group, [])
        )
    except Exception as exc:  # importlib.metadata quirks across versions — never fatal
        logger.debug("plugin discovery for group %s failed: %s", group, exc)
        return []

    loaded: List[str] = []
    for ep in selected:
        try:
            ep.load()  # imports the module → its decorators register the plugin
            loaded.append(ep.name)
            logger.info("loaded plugin '%s' (group %s)", ep.name, group)
        except Exception as exc:  # a broken third-party plugin must not break the core
            logger.warning("plugin '%s' (group %s) failed to load: %s", ep.name, group, exc)
    return loaded


def discover_all_plugins() -> List[str]:
    """Load every plugin group (used at pre-flight so the eval path sees all extensions)."""
    loaded: List[str] = []
    for group in GROUPS:
        loaded.extend(load_plugins(group))
    return loaded
