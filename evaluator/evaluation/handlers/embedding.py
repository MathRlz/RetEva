"""Embedding stage handlers: audio / text embedding, fusion, corpus index.

Each handler registers itself via ``@register_stage_handler`` at import time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from ..stage_registry import register_stage_handler
from ...logging_config import get_logger, TimingContext
from ..helpers import _build_relevant_from_item, collate_fn
from ..item_isolation import isolate_batch
from ..executor.state import RunState
from ..executor.node_pipeline import _node_pipeline
from ...models.retrieval.embedding_fusion import fuse_embeddings, validate_fusion_config
from .source import _node_dataset
from ._common import publish_keyed_or_plain
from ...utils.progress import progress_iter
from ._dispatch import dispatch_operator
from ..audio_refs import resolve_audio_dataset
from .retrieval import _stage_result_fusion
from ...metrics import embedding_alignment
from ...models.embedding_space import resolve_embedding_space
from ...models.registry import text_embedding_registry
from ...pipeline.corpus_index import embed_corpus, embed_corpus_audio, merge_corpus_vectors
from ...config.graph_config import resolved_model_config

logger = get_logger(__name__)


def _log_embedding_stats(embeddings, label: str) -> None:
    """Log shape / mean / std / norm of an embedding matrix (debug aid)."""
    if len(embeddings) == 0:
        return
    emb_array = np.array(embeddings)
    logger.info("%s shape: %s", label, emb_array.shape)
    logger.info(
        "%s stats - mean: %.4f, std: %.4f", label, emb_array.mean(), emb_array.std()
    )
    logger.info(
        "%s norms - mean: %.4f", label, np.linalg.norm(emb_array, axis=1).mean()
    )


def _embed_all_audio(dataset, audio_embedding_pipeline, batch_size, num_workers):
    """Run the audio-embedding pass over the whole dataset -> parallel lists (embeddings,
    ground_truth, relevance, query_ids); shared by audio_text_retrieval and
    audio_emb_retrieval modes."""
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
    with TimingContext("Audio embedding step", logger):
        for batch in progress_iter(dataloader, "Audio embedding", unit="batch", min_items=1):
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
    align 1:1; otherwise the plain array. Retrieval reads the chosen stream via
    ``s.input('query_vectors')`` (one_of the per-stream names), so the per-stream name here
    is what makes fusion-bail-to-audio fall out of the wiring."""
    publish_keyed_or_plain(s, name, embeddings, query_ids)


@register_stage_handler("embed", time_key="embedding_s")
def _stage_embed(s: RunState) -> None:
    """The ``embed`` operator: dispatches by axis/modality to the typed handler."""

    dispatch_operator("embed", {
        "text_embedding": _stage_text_embedding,
        "audio_embedding": _stage_audio_embedding,
        "corpus_embedding": _stage_corpus_embedding,
    }, s)


def _stage_audio_embedding(s: RunState) -> None:
    """Embed all audio; becomes the retrieval input unless fusion overrides it."""
    s.cb("step_1_audio", 0, s.total, "Step 1: Audio embedding")
    params = s.node_params
    # Bus-first audio: republished refs wrap the dataset in a decoding view.

    audio_dataset = resolve_audio_dataset(s, s.dataset)
    with _node_pipeline(s, "audio_embedding", params):
        emb, gt, rel, qids = _embed_all_audio(
            audio_dataset, s.audio_embedding_pipeline, s.batch_size, s.num_workers
        )
    # Pure transform: publishes only the audio query vectors. GT
    # (reference_transcription / relevant_docs) comes from dataset_source.
    _publish_query_vectors(s, emb, qids, name="audio_query_vectors")
    logger.info("Audio embedding step complete: %d audio embeddings", len(emb))
    s.cb("step_1_audio", s.total, s.total, "Audio embedding complete")
    _log_embedding_stats(emb, "Audio embedding")


def _stage_text_embedding(s: RunState) -> None:
    """Mode-agnostic text embedding: embed the current query text → text_query_vectors. The
    query text is read via ``s.input('query_text')`` (the asr/correction/optimization chain
    in ASR modes, the dataset_source question text in audio_text fusion); no
    fusion-specific branch — the fusion node fuses the streams downstream."""
    if s.text_embedding_pipeline is None:
        return
    s.cb("step_2_embedding", 0, s.total, "Step 2: Text embedding")
    query_text = s.input("query_text")
    if not query_text:
        # No text query in this run (e.g. an audio_text fusion over a pure-audio dataset with
        # no text-query column) → publish nothing, so a downstream fusion node falls back to
        # audio instead of crashing on `len(None)`.
        logger.debug("text_embedding: no query_text bound — skipping (no text_query_vectors)")
        return
    params = s.node_params
    node_id = getattr(s.current_node, "id", "text_embedding")
    # Per-item identity rides the keyed query_text ItemSet.
    keyed = s.input_items("query_text")
    ids = (
        list(keyed.ids) if keyed is not None else [str(i) for i in range(len(query_text))]
    )
    sentinel = object()
    with TimingContext("Text embedding step", logger), _node_pipeline(
        s, "text_embedding", params
    ):
        # Per-item isolation: a single un-embeddable query drops out instead of
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
    logger.info("Text embedding step complete: %d embeddings", len(embeddings))
    s.cb("step_2_embedding", s.total, s.total, "Text embedding complete")
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
    """The ``combine`` operator: dispatches by level to the typed handler."""

    return dispatch_operator("combine", {
        "fusion": _stage_fusion,
        "result_fusion": _stage_result_fusion,
        "corpus_merge": _stage_corpus_merge,
    }, s)


def _stage_fusion(s: RunState) -> None:
    """Fuse the audio + text query-vector streams into a distinct fused stream. Inputs are
    the two same-space streams from the bus: ``audio_query_vectors`` + ``text_query_vectors``.
    Skipping (no text pipeline / no text vectors) publishes nothing, so retrieval's one_of
    falls back to ``audio_query_vectors``."""
    text_arr = s.get_artifact("text_query_vectors", default=None)
    if s.text_embedding_pipeline is None or text_arr is None:
        logger.warning(
            "Embedding fusion is enabled but text_embedding_pipeline is not available. "
            "Skipping fusion and using audio embeddings only."
        )
        return
    audio_arr = np.asarray(s.get_artifact("audio_query_vectors"))
    text_arr = np.asarray(text_arr)
    # The embedding-fusion config is carried on the fusion node, resolved before execution
    # (global ⊕ node params, presence ⇒ enabled).
    fcfg = s.resolved_config(default=s.embedding_fusion_config)
    if fcfg is not None and not fcfg.enabled:
        return  # a per-branch {enabled: false} fusion node — retrieval falls back to audio-only
    # audio↔text alignment is its own embedding_alignment_metrics node.
    validate_fusion_config(
        fcfg,
        audio_dim=audio_arr.shape[1],
        text_dim=text_arr.shape[1],
    )
    with TimingContext("Embedding Fusion", logger):
        fused_embeddings, fusion_meta = fuse_embeddings(
            audio_arr, text_arr, fcfg
        )
    # Concatenate+PCA fusion fits a dim-reducer on *this* call's query vectors. It is
    # deterministic within a run (random_state=42) but fit per-run; keep it on the bus rather
    # than discarding it, so it is auditable and reusable if corpus-side fusion is ever added
    # (which MUST reuse this reducer — fitting a second one would project the two sides into
    # different spaces). Cross-run comparability requires pinning it.
    if fusion_meta is not None:
        s.put_artifact("fusion_reducer", fusion_meta)
        logger.debug("Fusion reducer fit per-run (kept on the bus for reuse/audit).")
    logger.info("Fused embedding shape: %s", fused_embeddings.shape)
    logger.info(
        "Fused embedding stats - mean: %.4f, std: %.4f",
        fused_embeddings.mean(), fused_embeddings.std(),
    )
    logger.info(
        "Fused embedding norms - mean: %.4f",
        np.linalg.norm(fused_embeddings, axis=1).mean(),
    )
    logger.info(
        "Using fused embeddings for retrieval (method: %s)",
        fcfg.fusion_method,
    )
    s.put_artifact("fused_query_vectors", fused_embeddings)


def _stage_embedding_alignment_metrics(s: RunState) -> None:
    """Diagnostic comparison node: the audio↔text cosine alignment of the two query-vector
    streams (a fusion-quality signal the report reads)."""
    text_arr = s.get_artifact("text_query_vectors", default=None)
    audio_arr = s.get_artifact("audio_query_vectors", default=None)
    if text_arr is None or audio_arr is None:
        return
    try:
        alignment = embedding_alignment(np.asarray(audio_arr), np.asarray(text_arr))
        if alignment is not None:
            s.put_artifact("embedding_alignment", alignment)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Embedding alignment computation failed: %s", exc)


def _stage_corpus_embedding(s: RunState) -> None:
    """Embed the corpus (text or audio) → ``corpus_vectors``. Text path when a text embedder
    exists (incl. per-node overrides + the store-agnostic embedding cache), else the
    audio-corpus path (TTS-synthesize + audio embed, uncached — parity with the old
    ``corpus_index``). No-op when ``config`` is absent (direct callers pre-build the index)."""
    if s.config is None:
        return
    if s.text_embedding_pipeline is not None:
        params = s.node_params
        model_identity = None
        if params.get("model") or params.get("name"):
            model_identity = {
                "model_name": params.get("name"),
                "model_type": params.get("model"),
                "device": params.get("device"),
            }
        # Bus-first: the corpus artifact from this node's bound producer —
        # a corpus-axis transform's perturbed docs win over the raw dataset corpus.
        bus_corpus = s.get_artifact("corpus", default=None)
        with _node_pipeline(s, "text_embedding", params):
            cv = embed_corpus(
                s.config,
                _node_dataset(s),  # dataset fallback / cache identity
                s.text_embedding_pipeline,
                cache_manager=s.cache_manager,
                model_identity=model_identity,
                load_info=s.load_info,
                corpus=list(bus_corpus) if bus_corpus is not None else None,
            )
        # Same overlay _node_pipeline just built the pipeline from (stage="text_embedding", this
        # node's own params) — reuse it instead of hand-rolling `params.get(x) or
        # config.model.y`.
        from types import SimpleNamespace

        mcfg = resolved_model_config(s.config, SimpleNamespace(stage="text_embedding", params=params))
        model_type = mcfg.text_emb_model_type
        if model_type:
            cv.space = (
                str(mcfg.text_emb_embedding_space)
                if mcfg.text_emb_embedding_space
                else resolve_embedding_space(
                    text_embedding_registry,
                    str(model_type),
                    mcfg.text_emb_model_name,
                )
            )
        s.put_artifact("corpus_vectors", cv)
        return
    if s.audio_embedding_pipeline is not None:
        cv = embed_corpus_audio(s.config, _node_dataset(s), s.audio_embedding_pipeline)
        if cv is not None:
            s.put_artifact("corpus_vectors", cv)


def _stage_corpus_merge(s: RunState) -> None:
    """Concatenate the embedded corpora of EVERY bound producer into one set. Producer order
    is preserved (stable doc identity); a single publisher passes through unchanged; no
    publisher → no-op. Space/dim mismatches fail loud in ``merge_corpus_vectors``."""

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
