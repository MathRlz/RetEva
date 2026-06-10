"""Embedding stage handlers: audio / text embedding, fusion, corpus index.

Moved verbatim from the former ``evaluation/phased.py`` (Phase 1, X6). Each handler
registers itself via ``@register_stage_handler`` at import time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from ..stage_registry import register_stage_handler
from ...logging_config import get_logger, TimingContext
from ..helpers import _build_relevant_from_item, collate_fn
from ..item_isolation import isolate_batch
from ..executor.state import RunState
from ..executor.node_pipeline import _node_pipeline
from ...models.retrieval.embedding_fusion import fuse_embeddings, validate_fusion_config
from .source import _node_dataset

logger = get_logger(__name__)


def _log_embedding_stats(embeddings, label: str) -> None:
    """Log shape / mean / std / norm of an embedding matrix (debug aid)."""
    if len(embeddings) == 0:
        return
    emb_array = np.array(embeddings)
    logger.info(f"{label} shape: {emb_array.shape}")
    logger.info(
        f"{label} stats - mean: {emb_array.mean():.4f}, std: {emb_array.std():.4f}"
    )
    logger.info(f"{label} norms - mean: {np.linalg.norm(emb_array, axis=1).mean():.4f}")


def _embed_all_audio(dataset, audio_embedding_pipeline, batch_size, num_workers):
    """Run the audio-embedding pass over the whole dataset.

    Returns parallel lists (embeddings, ground_truth, relevance, query_ids).
    Shared by audio_text_retrieval and audio_emb_retrieval modes.
    """
    embeddings: list = []
    ground_truth: list = []
    relevance: list = []
    query_ids: list = []
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    with TimingContext("Audio Embedding Phase", logger):
        for batch in tqdm(dataloader, desc="Audio embedding"):
            audio_list = [item["audio_array"] for item in batch]
            sampling_rates = [item["sampling_rate"] for item in batch]
            transcriptions = [item["transcription"] for item in batch]
            relevance_batch = [_build_relevant_from_item(item) for item in batch]
            query_ids_batch = [
                str(item.get("question_id", len(query_ids) + idx))
                for idx, item in enumerate(batch)
            ]
            batch_embeddings = audio_embedding_pipeline.process_batch(
                audio_list, sampling_rates
            )
            embeddings.extend(batch_embeddings)
            ground_truth.extend(transcriptions)
            relevance.extend(relevance_batch)
            query_ids.extend(query_ids_batch)
    return embeddings, ground_truth, relevance, query_ids


def _publish_query_vectors(s: RunState, embeddings: Any) -> None:
    """Publish ``query_vectors`` as a keyed ``ItemSet`` when query ids align 1:1 (W2);
    otherwise the plain array. ``get_artifact('query_vectors')`` returns the values either way
    (retrieval reads them as a matrix unchanged), so this is behaviour-preserving; keyed
    consumers (per-branch embedding-space checks, metric nodes) can read it via ``get_items``.
    """
    vals = list(embeddings)
    ids = [str(i) for i in (s.all_query_ids or [])]
    if len(ids) == len(vals) and len(set(ids)) == len(ids):
        from ..item_set import ItemSet

        s.put_items("query_vectors", ItemSet(ids, vals))
    else:
        s.put_artifact("query_vectors", embeddings)


@register_stage_handler("audio_embedding", time_key="asr_s")
def _stage_audio_embedding(s: RunState) -> None:
    """Embed all audio; becomes the retrieval input unless fusion overrides it."""
    s.cb("phase_1_audio", 0, s.total, "Phase 1: Audio embedding")
    params = getattr(s.current_node, "params", None)
    with _node_pipeline(s, "audio_embedding", params):
        emb, gt, rel, qids = _embed_all_audio(
            s.dataset, s.audio_embedding_pipeline, s.batch_size, s.num_workers
        )
    s.all_embeddings = emb
    s.audio_embeddings = np.array(emb) if len(emb) > 0 else emb
    s.all_ground_truth, s.all_relevance, s.all_query_ids = gt, rel, qids
    _publish_query_vectors(s, emb)  # keyed by query id when aligned (W2)
    logger.info(
        f"Audio Embedding Phase complete: {len(s.all_embeddings)} audio embeddings"
    )
    s.cb("phase_1_audio", s.total, s.total, "Audio embedding complete")
    _log_embedding_stats(s.all_embeddings, "Audio embedding")


@register_stage_handler("text_embedding", time_key="embedding_s")
def _stage_text_embedding(s: RunState) -> None:
    """Text embedding. For ASR modes embeds the (optimized) hypotheses into the
    retrieval input; for fused audio_text it embeds reference texts for fusion."""
    if s.mode == "asr_text_retrieval":
        if s.query_opt_bypassed:
            return
        s.cb("phase_2_embedding", 0, s.total, "Phase 2: Text embedding")
        # Read the query text from its bound producer (asr / query_optimization) (R4d).
        query_text = s.get_artifact("query_text", default=s.all_hypotheses)
        params = getattr(s.current_node, "params", None)
        node_id = getattr(s.current_node, "id", "text_embedding")
        ids = [str(i) for i in (s.all_query_ids or range(len(query_text)))]
        sentinel = object()
        with TimingContext("Embedding Phase", logger), _node_pipeline(
            s, "text_embedding", params
        ):
            # Per-item isolation (T1): a single un-embeddable query drops out instead of
            # aborting the run. A placeholder keeps positional alignment; the keyed report
            # excludes the dropped id so the placeholder never reaches a metric.
            embs = isolate_batch(
                ids,
                list(query_text),
                batch_fn=lambda xs: s.text_embedding_pipeline.process_batch(
                    xs, show_progress=True, desc="Embedding query transcriptions (ASR)"
                ),
                item_fn=lambda t: s.text_embedding_pipeline.process_batch([t])[0],
                node_id=node_id,
                placeholder=sentinel,
                sink=s.drop_sink,
            )
        valid = next((e for e in embs if e is not sentinel), None)
        if valid is not None:
            zero = np.zeros_like(np.asarray(valid))
            embs = [zero if e is sentinel else e for e in embs]
        s.all_embeddings = np.array(embs)
        _publish_query_vectors(
            s, s.all_embeddings
        )  # keyed by query id when aligned (W2)
        logger.info(
            f"Text Embedding Phase complete: {len(s.all_embeddings)} embeddings"
        )
        s.cb("phase_2_embedding", s.total, s.total, "Text embedding complete")
        if s.cache_manager and s.experiment_id:
            s.cache_manager.set_checkpoint(
                f"{s.experiment_id}_phased",
                {
                    "phase": "embedding",
                    "hypotheses": s.all_hypotheses,
                    "ground_truth": s.all_ground_truth,
                },
            )
    elif s.mode == "audio_text_retrieval":
        # Fusion path: embed reference texts (skipped gracefully if no pipeline).
        if s.text_embedding_pipeline is None:
            return
        s.cb("phase_1_5_fusion", 0, s.total, "Phase 1.5: Text embedding & fusion")
        with TimingContext("Text Embedding for Fusion", logger):
            te = s.text_embedding_pipeline.process_batch(
                s.all_ground_truth,
                show_progress=True,
                desc="Embedding reference texts (fusion)",
            )
            s.text_embeddings_for_fusion = np.array(te)
        logger.info(f"Text embedding shape: {s.text_embeddings_for_fusion.shape}")
        logger.info(
            f"Text embedding stats - mean: {s.text_embeddings_for_fusion.mean():.4f}, "
            f"std: {s.text_embeddings_for_fusion.std():.4f}"
        )


@register_stage_handler("fusion", time_key="embedding_s")
def _stage_fusion(s: RunState) -> None:
    """Fuse audio + reference-text embeddings into the retrieval input."""
    if s.text_embedding_pipeline is None or s.text_embeddings_for_fusion is None:
        logger.warning(
            "Embedding fusion is enabled but text_embedding_pipeline is not available. "
            "Skipping fusion and using audio embeddings only."
        )
        return
    audio_arr = s.audio_embeddings
    text_arr = s.text_embeddings_for_fusion
    s.audio_emb_for_alignment = audio_arr
    s.text_emb_for_alignment = text_arr
    validate_fusion_config(
        s.embedding_fusion_config,
        audio_dim=audio_arr.shape[1],
        text_dim=text_arr.shape[1],
    )
    with TimingContext("Embedding Fusion", logger):
        fused_embeddings, fusion_meta = fuse_embeddings(
            audio_arr, text_arr, s.embedding_fusion_config
        )
    if fusion_meta:
        logger.debug("Fusion metadata: %s", fusion_meta)
    logger.info(f"Fused embedding shape: {fused_embeddings.shape}")
    logger.info(
        f"Fused embedding stats - mean: {fused_embeddings.mean():.4f}, std: {fused_embeddings.std():.4f}"
    )
    logger.info(
        f"Fused embedding norms - mean: {np.linalg.norm(fused_embeddings, axis=1).mean():.4f}"
    )
    logger.info(
        f"Using fused embeddings for retrieval (method: {s.embedding_fusion_config.fusion_method})"
    )
    s.all_embeddings = fused_embeddings
    s.put_artifact("query_vectors", fused_embeddings)


@register_stage_handler("corpus_index", time_key="embedding_s")
def _stage_corpus_index(s: RunState) -> None:
    """Embed the corpus (text or audio) + build the vector index, inside the DAG.

    Reuses the shared ``build_corpus_index`` implementation (behavior-preserving).
    No-op when ``config`` is absent (direct callers pre-build the index) or the index
    already exists (e.g. the oracle baseline re-run reusing the same pipeline).
    """
    if s.retrieval_pipeline is None:
        return
    from ...services.evaluation_service import build_corpus_index

    params = getattr(s.current_node, "params", None) or {}
    # Per-branch corpus index (R3): when this node names an embedder, build a SEPARATE
    # indexed pipeline in that embedder's space (so it can't clobber another branch's index
    # on the shared pipeline) and publish it as this node's vector_index. The branch must set
    # corpus_index.model == its text_embedding.model so query and corpus share a space.
    if (params.get("model") or params.get("name")) and s.config is not None:
        from ...pipeline.factory import _create_retrieval_pipeline

        rp = _create_retrieval_pipeline(s.config, s.cache_manager, None)
        with _node_pipeline(s, "text_embedding", params):
            build_corpus_index(
                s.config,
                _node_dataset(s),  # corpus from this node's bound source (B2)
                rp,
                text_emb_pipeline=s.text_embedding_pipeline,
                audio_emb_pipeline=s.audio_embedding_pipeline,
                cache_manager=s.cache_manager,
                load_info=s.load_info,
            )
        s.put_artifact("vector_index", rp)
        return
    # Shared path: build on the run's pipeline (CSE collapses identical corpus_index nodes).
    already_indexed = getattr(s.retrieval_pipeline, "_index_payloads", None) is not None
    if s.config is not None and not already_indexed:
        build_corpus_index(
            s.config,
            _node_dataset(s),  # corpus from this node's bound source (B2)
            s.retrieval_pipeline,
            text_emb_pipeline=s.text_embedding_pipeline,
            audio_emb_pipeline=s.audio_embedding_pipeline,
            cache_manager=s.cache_manager,
            load_info=s.load_info,
        )
    s.put_artifact("vector_index", s.retrieval_pipeline)
