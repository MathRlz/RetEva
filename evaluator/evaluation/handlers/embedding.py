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
        for batch in tqdm(dataloader, desc="Audio embedding", disable=None):
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


def _publish_query_vectors(
    s: RunState, embeddings: Any, query_ids: list, name: str = "query_vectors"
) -> None:
    """Publish the query embeddings under ``name`` as a keyed ``ItemSet`` when ``query_ids``
    align 1:1 (W2); otherwise the plain array. Retrieval reads the chosen stream via
    ``s.input('query_vectors')`` (one_of the per-stream names), so the per-stream name here
    is what makes fusion-bail-to-audio fall out of the wiring. Keyed consumers read it via
    ``get_items``."""
    vals = list(embeddings)
    ids = [str(i) for i in (query_ids or [])]
    if len(ids) == len(vals) and len(set(ids)) == len(ids):
        from ..item_set import ItemSet

        s.put_items(name, ItemSet(ids, vals))
    else:
        s.put_artifact(name, embeddings)


@register_stage_handler("embed", time_key="embedding_s")
def _stage_embed(s: RunState) -> None:
    """The ``embed`` operator: dispatch by axis/modality to the text / audio / corpus
    embedder body (collapsed from three nodes; the bodies are unchanged and still publish
    their stream by literal name)."""
    from ._dispatch import dispatch_operator

    dispatch_operator("embed", {
        "text_embedding": _stage_text_embedding,
        "audio_embedding": _stage_audio_embedding,
        "corpus_embedding": _stage_corpus_embedding,
    }, s)


def _stage_audio_embedding(s: RunState) -> None:
    """Embed all audio; becomes the retrieval input unless fusion overrides it."""
    s.cb("phase_1_audio", 0, s.total, "Phase 1: Audio embedding")
    params = s.node_params
    # Bus-first audio (P4): republished refs wrap the dataset in a decoding view.
    from ..audio_refs import resolve_audio_dataset

    audio_dataset = resolve_audio_dataset(s, s.dataset)
    with _node_pipeline(s, "audio_embedding", params):
        emb, gt, rel, qids = _embed_all_audio(
            audio_dataset, s.audio_embedding_pipeline, s.batch_size, s.num_workers
        )
    # Pure transform (Phase 3): publishes only the audio query vectors. GT
    # (reference_transcription / relevant_docs) comes from dataset_source.
    _publish_query_vectors(s, emb, qids, name="audio_query_vectors")
    logger.info(f"Audio Embedding Phase complete: {len(emb)} audio embeddings")
    s.cb("phase_1_audio", s.total, s.total, "Audio embedding complete")
    _log_embedding_stats(emb, "Audio embedding")


def _stage_text_embedding(s: RunState) -> None:
    """Mode-agnostic text embedding: embed the current query text → text_query_vectors.

    The query text is read via ``s.input('query_text')`` (the asr/correction/optimization
    chain in ASR modes, the dataset_source question text in audio_text fusion). No
    fusion-specific branch and no reference_transcription read — the fusion node fuses the
    audio + text vector streams downstream."""
    if s.text_embedding_pipeline is None:
        return
    s.cb("phase_2_embedding", 0, s.total, "Phase 2: Text embedding")
    query_text = s.input("query_text")
    params = s.node_params
    node_id = getattr(s.current_node, "id", "text_embedding")
    # Per-item identity rides the keyed query_text ItemSet (M1d-2).
    keyed = s.input_items("query_text")
    ids = (
        list(keyed.ids) if keyed is not None else [str(i) for i in range(len(query_text))]
    )
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
                xs, show_progress=True, desc="Embedding query text"
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
    embeddings = np.array(embs)
    # Distinct stream (no query_vectors mutation): retrieval/fusion read it by name.
    _publish_query_vectors(s, embeddings, ids, name="text_query_vectors")
    logger.info(f"Text Embedding Phase complete: {len(embeddings)} embeddings")
    s.cb("phase_2_embedding", s.total, s.total, "Text embedding complete")
    if s.cache_manager and s.experiment_id:
        s.cache_manager.set_checkpoint(
            f"{s.experiment_id}_phased",
            {
                "phase": "embedding",
                "hypotheses": list(query_text),
                "ground_truth": list(
                    s.get_artifact("reference_transcription", default=[])
                ),
            },
        )


@register_stage_handler("combine", time_key="embedding_s")
def _stage_combine(s: RunState) -> None:
    """The ``combine`` operator: dispatch by level to embedding fusion / result fusion /
    corpus set-merge (bodies unchanged; result_fusion lives in the retrieval handlers)."""
    from .retrieval import _stage_result_fusion
    from ._dispatch import dispatch_operator

    return dispatch_operator("combine", {
        "fusion": _stage_fusion,
        "result_fusion": _stage_result_fusion,
        "corpus_merge": _stage_corpus_merge,
    }, s)


def _stage_fusion(s: RunState) -> None:
    """Fuse the audio + text query-vector streams into a distinct fused stream.

    Inputs are the two same-space streams from the bus: ``audio_query_vectors`` (from
    audio_embedding) + ``text_query_vectors`` (from text_embedding). Skipping (no text
    pipeline / no text vectors) publishes nothing, so retrieval's one_of falls back to
    ``audio_query_vectors``."""
    text_arr = s.get_artifact("text_query_vectors", default=None)
    if s.text_embedding_pipeline is None or text_arr is None:
        logger.warning(
            "Embedding fusion is enabled but text_embedding_pipeline is not available. "
            "Skipping fusion and using audio embeddings only."
        )
        return
    audio_arr = np.asarray(s.get_artifact("audio_query_vectors"))
    text_arr = np.asarray(text_arr)
    # (audio↔text alignment is its own embedding_alignment_metrics node now.)
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
    s.put_artifact("fused_query_vectors", fused_embeddings)


def _stage_embedding_alignment_metrics(s: RunState) -> None:
    """Diagnostic comparison node: the audio↔text cosine alignment of the two query-vector
    streams (a fusion-quality signal the report reads). Was inline in `fusion`."""
    text_arr = s.get_artifact("text_query_vectors", default=None)
    audio_arr = s.get_artifact("audio_query_vectors", default=None)
    if text_arr is None or audio_arr is None:
        return
    try:
        from ...metrics import embedding_alignment

        alignment = embedding_alignment(np.asarray(audio_arr), np.asarray(text_arr))
        if alignment is not None:
            s.put_artifact("embedding_alignment", alignment)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Embedding alignment computation failed: %s", exc)


def _stage_corpus_embedding(s: RunState) -> None:
    """Embed the corpus (text or audio) → ``corpus_vectors`` (the §4 split, embed half).

    Dispatch mirrors the former combined node: text path when a text embedder
    exists (incl. per-node ``model``/``name``/``device`` branch override + the
    store-agnostic embedding cache), else the audio-corpus path (TTS-synthesize
    + audio embed, uncached — parity with the old ``corpus_index``). No-op when
    ``config`` is absent (direct callers pre-build the index, R4a).
    """
    if s.config is None:
        return
    if s.text_embedding_pipeline is not None:
        from ...models.embedding_space import resolve_embedding_space
        from ...models.registry import text_embedding_registry
        from ...services.corpus_index import embed_corpus

        params = s.node_params
        model_identity = None
        if params.get("model") or params.get("name"):
            model_identity = {
                "model_name": params.get("name"),
                "model_type": params.get("model"),
                "device": params.get("device"),
            }
        # Bus-first (§4.1 T1): the corpus artifact from this node's bound producer —
        # a corpus-axis transform's perturbed docs win over the raw dataset corpus.
        bus_corpus = s.get_artifact("corpus", default=None)
        with _node_pipeline(s, "text_embedding", params):
            cv = embed_corpus(
                s.config,
                _node_dataset(s),  # dataset fallback / cache identity (B2)
                s.text_embedding_pipeline,
                cache_manager=s.cache_manager,
                model_identity=model_identity,
                load_info=s.load_info,
                corpus=list(bus_corpus) if bus_corpus is not None else None,
            )
        model_type = params.get("model") or s.config.model.text_emb_model_type
        if model_type:
            override = params.get("embedding_space") or getattr(
                s.config.model, "text_emb_embedding_space", None
            )
            cv.space = (
                str(override)
                if override
                else resolve_embedding_space(
                    text_embedding_registry,
                    str(model_type),
                    params.get("name") or s.config.model.text_emb_model_name,
                )
            )
        s.put_artifact("corpus_vectors", cv)
        return
    if s.audio_embedding_pipeline is not None:
        from ...services.corpus_index import embed_corpus_audio

        cv = embed_corpus_audio(s.config, _node_dataset(s), s.audio_embedding_pipeline)
        if cv is not None:
            s.put_artifact("corpus_vectors", cv)


def _stage_corpus_merge(s: RunState) -> None:
    """Concatenate the embedded corpora of EVERY bound producer into one set.

    Fusion's corpus-side sibling: combining corpora is an explicit node on the
    diagram, never an implicit overload of the bus's newest-wins read. Producer
    order is preserved (stable doc identity); a single publisher passes through
    unchanged; no publisher → no-op (direct callers, R4a). Space/dim mismatches
    fail loud in ``merge_corpus_vectors`` (§4 V[s]).
    """
    from ...services.corpus_index import merge_corpus_vectors

    published = [
        s.ctx.get(pid, "corpus_vectors")
        for pid in s._producers("corpus_vectors")
        if s.ctx.has(pid, "corpus_vectors")
    ]
    if not published:
        return
    merged = merge_corpus_vectors(published)
    if len(published) > 1:
        logger.info(
            "corpus_merge: %d corpora -> %d vectors",
            len(published),
            len(merged.payloads),
        )
    s.put_artifact("corpus_vectors", merged)
