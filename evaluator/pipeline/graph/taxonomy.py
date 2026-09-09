"""Node taxonomy — the declared class of every pipeline node.

``category`` is the data-flow ROLE, exactly one per node — load-bearing for
lifecycle/device/offload and execution; ``domain`` is UI-only grouping, not MECE-critical.
"""

from __future__ import annotations

CATEGORIES = frozenset(
    {"source", "model", "transform", "metric", "sink"}
)

DOMAINS = frozenset(
    {
        "ingest",
        "query",
        "transcription",
        "embedding",
        "retrieval",
        "refine",
        "fusion",
        "generation",
        "robustness",
        "scoring",
        "reporting",
        "export",
    }
)


def validate_taxonomy(stage: str, category, domain) -> None:
    """Raise if a node declares a category/domain outside the allow-list (typo guard at
    registration — every node MUST classify into a known class).

    An operator may declare a *callable* category/domain (it varies by field, e.g.
    ``convert`` is model for asr / transform for tts); a callable is trusted here and its
    resolved value is validated where it's read (``node_category`` + the taxonomy test)."""
    if not callable(category) and category not in CATEGORIES:
        raise ValueError(
            f"node {stage!r}: unknown category {category!r}; "
            f"choose from {sorted(CATEGORIES)}"
        )
    if not callable(domain) and domain not in DOMAINS:
        raise ValueError(
            f"node {stage!r}: unknown domain {domain!r}; choose from {sorted(DOMAINS)}"
        )
