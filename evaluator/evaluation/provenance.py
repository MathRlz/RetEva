"""Run provenance + deterministic per-item seeds (architecture A10).

An eval framework's output must be reproducible/comparable, so the ``report`` carries a
``provenance`` block: config hash, resolved model identities, the run seed, library
versions, git commit, and per-stage timing. Augmentation is seeded *per item* —
``item_seed(seed, query_id, node_id, variant)`` — so a given item's variant is reproducible
regardless of batch order or parallelism.

See ``evaluator-architecture.md`` §5/§8.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from typing import Any, Dict, List, Optional

from ..logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_SEED = 42


def set_global_determinism(seed: Optional[int] = None) -> Dict[str, Any]:
    """Seed every RNG + request deterministic kernels at run start (S5).

    Seeds ``random``/``numpy``/``torch`` (CPU+CUDA), sets cuDNN deterministic, and asks
    torch for deterministic algorithms. The last is opt-out via ``EVALUATOR_NONDETERMINISM=1``
    (some GPU kernels have no deterministic impl and would otherwise raise). Returns the flags
    actually applied, for the provenance block — so a run never *claims* reproducibility it
    couldn't enforce. Best-effort: missing torch / unsupported kernels never fail the run.
    """
    s = DEFAULT_SEED if seed is None else int(seed)
    import random as _random

    _random.seed(s)
    flags: Dict[str, Any] = {
        "seed": s,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }
    try:
        import numpy as np

        np.random.seed(s % (2**32))
    except Exception as exc:
        logger.debug("numpy seeding skipped: %s", exc)
    try:
        import torch

        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            flags["cudnn_deterministic"] = True
        except Exception as exc:
            logger.debug("cuDNN determinism flags not applied: %s", exc)
        opt_out = os.environ.get("EVALUATOR_NONDETERMINISM") == "1"
        if not opt_out:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
                flags["deterministic_algorithms"] = True
            except Exception as exc:  # no deterministic kernel for some op
                flags["deterministic_algorithms"] = False
                flags["deterministic_note"] = str(exc)
        else:
            flags["deterministic_algorithms"] = False
            flags["deterministic_note"] = "opt-out via EVALUATOR_NONDETERMINISM=1"
    except Exception as exc:
        logger.debug("torch seeding skipped: %s", exc)
        flags["torch"] = "unavailable"
    return flags


def item_seed(seed: int, query_id: str, node_id: str, variant: int = 0) -> int:
    """Deterministic 32-bit seed for one (item, node, variant) — order/parallelism-independent."""
    key = f"{seed}\x1f{query_id}\x1f{node_id}\x1f{variant}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception as exc:
        logger.debug("git commit probe failed: %s", exc)
        return None


def _library_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for mod in ("torch", "transformers", "numpy", "sentence_transformers"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception as exc:
            logger.debug("version probe for %s failed: %s", mod, exc)
    import platform

    versions["python"] = platform.python_version()
    return versions


def _determinism_state() -> Dict[str, Any]:
    """Current-process reproducibility state for the provenance block (S5): the hash seed
    and the torch determinism flags actually in effect (records the truth, GPU caveats and
    all)."""
    state: Dict[str, Any] = {"pythonhashseed": os.environ.get("PYTHONHASHSEED")}
    try:
        import torch

        state["cudnn_deterministic"] = bool(
            getattr(torch.backends.cudnn, "deterministic", False)
        )
        try:
            state["deterministic_algorithms"] = bool(
                torch.are_deterministic_algorithms_enabled()
            )
        except Exception as exc:
            logger.debug("deterministic-algorithms probe failed: %s", exc)
    except Exception as exc:
        logger.debug("torch determinism probe skipped: %s", exc)
    return state


def config_hash(config: Any) -> str:
    """Stable short hash over the identity-bearing config (models, data, retrieval)."""
    model = getattr(config, "model", None)
    vdb = getattr(config, "vector_db", None)
    data = getattr(config, "data", None)
    payload = {
        "pipeline_mode": str(getattr(model, "pipeline_mode", "")),
        "asr": getattr(model, "asr_model_type", None),
        "asr_name": getattr(model, "asr_model_name", None),
        "text_emb": getattr(model, "text_emb_model_type", None),
        "text_emb_name": getattr(model, "text_emb_model_name", None),
        "audio_emb": getattr(model, "audio_emb_model_type", None),
        "retrieval_mode": getattr(vdb, "retrieval_mode", None),
        "k": getattr(vdb, "k", None),
        "dataset": getattr(data, "dataset_name", None),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def resolved_models(config: Any) -> Dict[str, str]:
    """The model identities that produced this run (type:name per family)."""
    model = getattr(config, "model", None)
    if model is None:
        return {}
    out: Dict[str, str] = {}
    for family, type_attr, name_attr in (
        ("asr", "asr_model_type", "asr_model_name"),
        ("text_embedding", "text_emb_model_type", "text_emb_model_name"),
        ("audio_embedding", "audio_emb_model_type", "audio_emb_model_name"),
    ):
        mtype = getattr(model, type_attr, None)
        if mtype:
            name = getattr(model, name_attr, None)
            out[family] = f"{mtype}:{name}" if name else str(mtype)
    return out


def dataset_content_fingerprint(dataset: Any) -> Optional[Dict[str, Any]]:
    """A content fingerprint of the evaluated dataset (R1 reproducibility).

    Records the corpus + question counts and a stable SHA-256 over the corpus doc ids + text,
    so two runs can tell whether they evaluated *the same data* — turning "the numbers moved"
    into "the corpus changed" (or not). Deterministic for a fixed dataset; ``None`` when the
    dataset exposes neither a corpus nor a length. The corpus is re-read once (already loaded
    for embedding); questions are counted, not hashed (hashing them would reload audio).
    """
    if dataset is None:
        return None
    info: Dict[str, Any] = {}
    try:
        if hasattr(dataset, "get_corpus"):
            corpus = list(dataset.get_corpus())
            digest = hashlib.sha256()
            for doc in corpus:
                if isinstance(doc, dict):
                    doc_id = str(doc.get("doc_id", ""))
                    text = str(doc.get("text", ""))
                else:
                    doc_id, text = "", str(doc)
                digest.update(doc_id.encode("utf-8"))
                digest.update(b"\x1f")
                digest.update(text.encode("utf-8"))
                digest.update(b"\x1e")
            info["corpus_docs"] = len(corpus)
            info["corpus_sha256"] = digest.hexdigest()[:16]
    except Exception as exc:
        logger.debug("corpus fingerprint skipped: %s", exc)
    try:
        info["questions"] = len(dataset)
    except (TypeError, AttributeError) as exc:
        logger.debug("question-count fingerprint skipped: %s", exc)
    return info or None


def build_provenance(
    config: Any,
    *,
    seed: Optional[int] = None,
    dataset: Optional[Dict[str, Any]] = None,
    failure_analysis: Optional[Dict[str, Any]] = None,
    timing: Optional[Dict[str, float]] = None,
    dropped: Optional[Dict[str, int]] = None,
    dropped_by_branch: Optional[Dict[str, List[str]]] = None,
    dropped_by_node: Optional[Dict[str, List[str]]] = None,
    cache_stats: Optional[Dict[str, Any]] = None,
    cost: Optional[Dict[str, Any]] = None,
    offload: Optional[Dict[str, Any]] = None,
    models: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the ``report.provenance`` block for a run.

    ``dropped_by_branch`` makes the silently-smaller paired denominator (S1) visible at
    report level: per branch, the query ids absent from that branch (failed/dropped) so a
    reader can see *which* items each branch lost relative to the full run (S2).
    ``dropped_by_node`` records per-node per-item failures isolated by drop-and-log (T1) —
    which node dropped which query ids, so a shrinking sample is attributable to a stage.
    ``cache_stats`` records per-stage cache hit/miss counts (T3) so a reader knows which
    artifacts were recomputed vs reused — a stale-cache or version-bump effect is auditable.
    """
    prov: Dict[str, Any] = {
        "config_hash": config_hash(config) if config is not None else None,
        # Structured per-pipeline identity when the caller supplies it (F30 — type/size/dim/
        # pooling/embedding_space/model_path/adapter + retrieval knobs, so a result is
        # reproducible/groupable); else the lightweight "type:name" map from the config.
        "models": models if models is not None else resolved_models(config),
        "seed": seed,
        "versions": _library_versions(),
        "git_commit": _git_commit(),
        "determinism": _determinism_state(),
    }
    if dataset:
        # Content fingerprint (R1) — corpus/question counts + corpus hash so a reader can tell
        # whether two runs evaluated the same data. Deterministic for a fixed dataset.
        prov["dataset"] = dict(dataset)
    if failure_analysis:
        # Per-item failure attribution (R7) — only present when items were actually dropped.
        prov["failure_analysis"] = dict(failure_analysis)
    if timing:
        prov["timing"] = dict(timing)
    if dropped:
        prov["dropped"] = dict(dropped)
    if dropped_by_branch:
        prov["dropped_by_branch"] = {
            b: list(ids) for b, ids in dropped_by_branch.items()
        }
    if dropped_by_node:
        prov["dropped_by_node"] = {n: list(ids) for n, ids in dropped_by_node.items()}
    if cache_stats:
        prov["cache"] = dict(cache_stats)
    if cost:
        prov["cost"] = dict(cost)
    if offload:
        # Soft-CPU offload events (2c) — only present when a model was actually parked warm,
        # so the default (full-free) policy leaves the block absent. Deterministic per run.
        prov["offload"] = dict(offload)
    return prov
