"""Operator vocabulary: the alias bijection (DAG_OPERATOR_ABSTRACTION.md).

The 34 named node types collapse into ~11 generic **operators** whose behavior is selected
by fields. This module holds the two halves of that bijection:

* ``ALIASES`` — *forward*: ``old_name -> (operator, fixed_fields)``. Used at one chokepoint
  (``wiring._normalize_spec_item``) to expand a legacy node-type string (or an
  operator-authored ``{op, fields}`` spec) into ``(operator, params)`` **without changing the
  node id** — so old configs/presets and the parity baselines keep their exact node ids.
* ``node_kind(operator, params)`` — *reverse*: maps an operator instance back to its concrete
  node kind (e.g. ``embed{axis:corpus}`` → ``corpus_embedding``), so every place that keys on a
  specific behavior (device map, offload, V[s] validation, branch ordering, handler dispatch, …)
  recovers it and reuses its kind-specific logic. The kind names are the pre-collapse node-type
  vocabulary, which is also the ``ALIASES`` keys — so the two halves stay in lockstep.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

# old node-type name -> (operator, fixed discriminator fields merged into params).
ALIASES: Dict[str, Tuple[str, dict]] = {
    "dataset_source": ("source", {}),
    "dataset_union": ("source", {"union": True}),
    "tts": ("convert", {"op": "tts"}),
    "asr": ("convert", {"op": "asr"}),
    "text_embedding": ("embed", {"axis": "query", "modality": "text"}),
    "audio_embedding": ("embed", {"axis": "query", "modality": "audio"}),
    "corpus_embedding": ("embed", {"axis": "corpus"}),
    "query_correction": ("transform", {"op": "correct", "axis": "query"}),
    "query_optimization": ("transform", {"op": "optimize", "axis": "query"}),
    "query_refine": ("transform", {"op": "refine", "axis": "query"}),
    "augmenter": ("transform", {"op": "perturb", "axis": "query"}),
    "augment_audio": ("transform", {"op": "perturb", "modality": "audio"}),
    "fusion": ("combine", {"level": "embedding"}),
    "result_fusion": ("combine", {"level": "result"}),
    "corpus_merge": ("combine", {"level": "set"}),
    "vector_db": ("index", {}),
    "retrieval": ("search", {}),
    "multi_query_retrieval": ("search", {"method": "multi_query"}),
    "rerank": ("refine", {"op": "rerank"}),
    "mmr": ("refine", {"op": "mmr"}),
    "threshold": ("refine", {"op": "threshold"}),
    "answer_gen": ("generate", {}),
    "transcription_metrics": ("measure", {"family": "transcription"}),
    "retrieval_metrics": ("measure", {"family": "retrieval"}),
    "answer_metrics": ("measure", {"family": "answer"}),
    "embedding_alignment_metrics": ("measure", {"family": "alignment"}),
    "metrics": ("measure", {"family": "report"}),
    "answer_judge": ("measure", {"family": "judge"}),
    "build_query_traces": ("measure", {"trace": True}),
    "finalize": ("sink", {"target": "finalize"}),
    "aggregate": ("sink", {"target": "aggregate"}),
    "dataset_sink": ("sink", {"target": "dataset"}),
    "leaderboard_sink": ("sink", {"target": "leaderboard"}),
    "tracking_sink": ("sink", {"target": "tracking"}),
}

# The operator vocabulary (the registered node-type names after the collapse).
OPERATORS = tuple(dict.fromkeys(op for op, _ in ALIASES.values()))

# operator -> the discriminator field keys that select its sub-behavior (the union of every
# alias's fixed fields). Changing one of these in the builder re-resolves the field-aware
# form (ports / model family / param switches), so the UI re-fetches /api/graph/node-form.
_DISCRIMINATORS: Dict[str, frozenset] = {}
for _op, _fixed in ALIASES.values():
    _DISCRIMINATORS.setdefault(_op, frozenset())
    if _fixed:
        _DISCRIMINATORS[_op] = _DISCRIMINATORS[_op] | frozenset(_fixed)


def operator_discriminators(operator: str) -> frozenset:
    """The discriminator field keys for an operator (empty for single-kind operators)."""
    return _DISCRIMINATORS.get(operator, frozenset())


def is_operator(name: str) -> bool:
    return name in OPERATORS


def expand_alias(name: str, params: Optional[dict]) -> Tuple[str, Optional[dict]]:
    """Forward: a legacy node-type name → (operator, merged params); explicit params win.

    Registration-aware so the collapse can land **one cluster at a time**: a legacy name
    expands only when its operator is actually registered; otherwise it stays legacy (still
    its own registered node). A name that is already an operator (or unknown) is unchanged.
    Pure — the input params dict is not mutated.
    """
    if name not in ALIASES:
        return name, params
    operator, fixed = ALIASES[name]
    from .registry import _NODE_REGISTRY  # lazy: avoid an import cycle

    if operator not in _NODE_REGISTRY:
        return name, params  # operator not collapsed yet — keep the legacy node
    return operator, {**fixed, **(params or {})}


def node_kind(operator: str, params: Optional[dict] = None) -> str:
    """The concrete node kind an operator instance resolves to — e.g.
    ``node_kind("embed", {"axis": "corpus"}) == "corpus_embedding"``.

    This is the reverse of the alias expansion: the discriminator fields uniquely identify the
    kind, so every site that keys on a specific behavior (handler dispatch, the device/offload
    maps, V[s] validation, branch ordering, the builder forms) calls this to recover it. The
    returned names are the pre-collapse node-type vocabulary (which is also the alias keys); a
    bare operator/unknown name is returned unchanged.
    """
    p = params or {}
    if operator == "source":
        return "dataset_union" if p.get("union") else "dataset_source"
    if operator == "convert":
        return "tts" if p.get("op") == "tts" else "asr"
    if operator == "embed":
        if p.get("axis") == "corpus":
            return "corpus_embedding"
        if p.get("modality") == "audio":
            return "audio_embedding"
        return "text_embedding"
    if operator == "transform":
        op = p.get("op")
        if op == "correct":
            return "query_correction"
        if op == "optimize":
            return "query_optimization"
        if op == "refine":
            return "query_refine"
        if op == "perturb":
            return "augment_audio" if p.get("modality") == "audio" else "augmenter"
        return "augmenter"
    if operator == "combine":
        return {
            "embedding": "fusion",
            "result": "result_fusion",
            "set": "corpus_merge",
        }.get(p.get("level"), "fusion")
    if operator == "index":
        return "vector_db"
    if operator == "search":
        return (
            "multi_query_retrieval"
            if p.get("method") in ("multi_query", "decompose")
            else "retrieval"
        )
    if operator == "refine":
        return {"rerank": "rerank", "mmr": "mmr", "threshold": "threshold"}.get(
            p.get("op"), "rerank"
        )
    if operator == "generate":
        return "answer_gen"
    if operator == "measure":
        if p.get("trace"):
            return "build_query_traces"
        return {
            "transcription": "transcription_metrics",
            "retrieval": "retrieval_metrics",
            "answer": "answer_metrics",
            "alignment": "embedding_alignment_metrics",
            "report": "metrics",
            "judge": "answer_judge",
        }.get(p.get("family"), "metrics")
    if operator == "sink":
        return {
            "finalize": "finalize",
            "aggregate": "aggregate",
            "dataset": "dataset_sink",
            "leaderboard": "leaderboard_sink",
            "tracking": "tracking_sink",
        }.get(p.get("target"), "finalize")
    # Already a legacy stage name (pre-collapse), or an unknown type — identity.
    return operator
