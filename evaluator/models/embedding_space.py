"""Embedding-space typing (architecture A2).

Dense retrieval dots a query vector with a corpus vector; the result is only meaningful when
both come from the **same embedding space**. Two embedders may share a space (joint /
cross-modal models such as SONAR text+speech) or not. Each embedder declares an
``embedding_space`` via registry metadata; when absent the space defaults to a unique id per
``(model_type, model_name)`` so distinct models never accidentally compare equal.

:func:`validate_embedding_spaces` rejects, at build time, a dense/hybrid retrieval whose
query-side and corpus-side embedders live in *different* spaces — e.g. ``audio_emb_retrieval``
with a ``wavlm`` audio query against a ``labse`` text corpus (silently garbage scores). The
classic *valid* cross-modal case (``sonar_speech`` query vs ``sonar`` text corpus) passes
because both declare ``embedding_space='sonar'``.

See ``evaluator-architecture.md`` §4.
"""

from __future__ import annotations

from typing import Any, Optional

from ..errors import ConfigurationError
from .registry import audio_embedding_registry, text_embedding_registry

# Retrieval modes that compare vectors via inner product (so spaces must match).
_VECTOR_MODES = {"dense", "hybrid"}


def resolve_embedding_space(
    registry: Any, model_type: str, model_name: Optional[str] = None
) -> str:
    """The embedding-space id for a model: declared ``embedding_space`` metadata if present,
    else a unique fallback ``"<type>:<name>"`` (distinct models never collide)."""
    declared = registry.get_metadata(model_type).get("embedding_space")
    if declared:
        return str(declared)
    name = model_name or registry.get_default_name(model_type) or ""
    return f"{model_type}:{name}"


def validate_embedding_spaces(config: Any) -> None:
    """Raise ``ConfigurationError`` if a dense/hybrid retrieval would dot vectors from
    different embedding spaces (query-side vs corpus-side). No-op for non-vector retrieval,
    asr_text (query embedder == corpus embedder), or fusion (combined space, not checked).

    Today the corpus is text-embedded for cross-modal retrieval, so the check that matters is
    ``audio_emb_retrieval``: the audio query embedder must share a space with the text corpus
    embedder.
    """
    model = getattr(config, "model", None)
    if model is None:
        return
    mode = str(getattr(model, "pipeline_mode", ""))
    vdb = getattr(config, "vector_db", None)
    retrieval_mode = str(getattr(vdb, "retrieval_mode", "dense")) if vdb else "dense"
    if retrieval_mode not in _VECTOR_MODES:
        return

    text_type = getattr(model, "text_emb_model_type", None)
    if mode == "audio_emb_retrieval":
        audio_type = getattr(model, "audio_emb_model_type", None)
        if not audio_type or not text_type:
            return  # not enough info to compare
        query_space = resolve_embedding_space(
            audio_embedding_registry,
            audio_type,
            getattr(model, "audio_emb_model_name", None),
        )
        corpus_space = resolve_embedding_space(
            text_embedding_registry,
            text_type,
            getattr(model, "text_emb_model_name", None),
        )
        if query_space != corpus_space:
            raise ConfigurationError(
                f"Embedding-space mismatch for dense '{mode}': audio query embedder "
                f"'{audio_type}' is in space '{query_space}' but the text corpus embedder "
                f"'{text_type}' is in space '{corpus_space}'. Dense retrieval dots these "
                f"vectors, so the scores would be meaningless. Use embedders that share a "
                f"space (e.g. sonar_speech + sonar), declare a shared `embedding_space` in "
                f"the model registry metadata, or switch to sparse retrieval."
            )
    # asr_text_retrieval: query and corpus both use text_emb → same space, always OK.
    # audio_text_retrieval (fusion): query is a fused vector — space combination is not
    # validated here (revisit with n-ary fusion).
