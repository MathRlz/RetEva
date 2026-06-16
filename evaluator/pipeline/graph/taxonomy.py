"""Node taxonomy — the declared class of every pipeline node.

Two orthogonal axes, declared once per node on its ``register_stage_node`` call:

- ``category`` — the data-flow ROLE, exactly one per node (a clean partition):
  ``source`` (emits dataset artifacts, no upstream) · ``model`` (identity is a managed
  encoder/ASR model — device + offload lifecycle) · ``transform`` (pure data→data) ·
  ``metric`` (compares expected GT × actual → scores) · ``sink`` (terminal side-effect, no
  artifact output). This axis is load-bearing: lifecycle/device/offload, validation, and
  execution reason about a node by its category.

- ``domain`` — the functional AREA, for UI grouping / DAG coloring / docs (not MECE-critical).

``model`` is reserved for the registry encoder/ASR nodes (the device/offload-managed set), so
``category == "model"`` ⟺ the set the device/offload maps encode. LLM-backed generation/judge
and a cross-encoder rerank *use* models but keep their data-flow category; their model-backing
is a secondary property (``model_field`` / runtime config), not the category.
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
