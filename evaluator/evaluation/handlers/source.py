"""Source stage handlers: dataset_source (graph root) + tts synthesis.

Moved verbatim from the former ``evaluation/phased.py`` (Phase 1, X5). Each handler
registers itself via ``@register_stage_handler`` at import time.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ..stage_registry import register_stage_handler
from ...logging_config import get_logger
from ..executor.state import RunState

logger = get_logger(__name__)


@register_stage_handler("source", self_timed=True)
def _stage_source(s: RunState) -> None:
    """The ``source`` operator: dispatch to dataset_source or dataset_union (union:true).
    Bodies unchanged."""
    from ._dispatch import dispatch_operator

    return dispatch_operator("source", {
        "dataset_source": _stage_dataset_source,
        "dataset_union": _stage_dataset_union,
    }, s)


def _stage_dataset_source(s: RunState) -> None:
    """Graph root: the dataset's source artifacts enter the DAG here. Loading + TTS
    happen in prepare_dataset before the graph (lifecycle reasons), so this validates
    the dataset is present + non-empty and logs its size; downstream nodes read it.

    Multi-dataset (B1): when the graph has several dataset_source nodes, each selects its source
    via ``params.dataset`` → ``s.dataset_sources[id]``; single-source falls back to ``s.dataset``.
    """
    dataset = _node_dataset(s)
    if dataset is None:
        raise ValueError("dataset_source: no dataset on the execution context")
    try:
        n = len(dataset)
    except TypeError:
        n = -1
    logger.info("dataset_source: %s samples", n if n >= 0 else "?")
    _publish_source_itemsets(s, dataset)
    _publish_corpus_itemset(s, dataset)
    _publish_vector_columns(s, dataset)
    from ..audio_refs import publish_audio_refs

    publish_audio_refs(s, dataset)


def _node_dataset(s: RunState) -> Any:
    """The dataset a node operates on (B1/B2): its ``params.dataset`` → ``s.dataset_sources[id]``,
    else the single ``s.dataset``. Lets ``corpus_index`` / ``dataset_source`` bind to a *specific*
    source in a multi-dataset graph; single-source graphs always get ``s.dataset``."""
    params = s.node_params
    sid = params.get("dataset")
    if sid is not None:
        ds = s.dataset_sources.get(str(sid))
        if ds is not None:
            return ds
    return s.dataset


def _question_short_answer(question: Any) -> Optional[str]:
    """A reference short answer for a question, if the dataset carries one."""
    direct = getattr(question, "short_answer", None) or getattr(
        question, "answer", None
    )
    if direct:
        return str(direct)
    meta = getattr(question, "metadata", None) or {}
    for key in ("short_answer", "answer", "reference_answer"):
        if meta.get(key):
            return str(meta[key])
    return None


def _source_rows(dataset: Any) -> Optional[List[dict]]:
    """Per-query GT rows ``{id, question_text, transcription, relevance, short_answer}``.

    Sourced cheaply from ``dataset.questions`` (BenchmarkQuestion metadata — no audio
    decode). Falls back to the positional sample dicts for legacy datasets with no
    ``questions`` attr; every real audio dataset exposes ``questions``, so that branch
    never force-decodes production audio. Returns None when neither path is available.
    """
    from ..helpers import _build_relevant_from_item

    questions = getattr(dataset, "questions", None)
    rows: List[dict] = []
    if questions:
        for i, q in enumerate(questions):
            qtext = getattr(q, "question_text", "") or ""
            transcription = getattr(q, "transcription", None) or qtext
            rows.append(
                {
                    "id": str(getattr(q, "question_id", i)),
                    "question_text": qtext,
                    "transcription": transcription,
                    "relevance": _build_relevant_from_item(
                        {
                            "relevance_grades": getattr(q, "relevance_grades", None),
                            "groundtruth_doc_ids": getattr(q, "groundtruth_doc_ids", None),
                            "transcription": transcription,
                        }
                    ),
                    "short_answer": _question_short_answer(q),
                }
            )
        return rows
    # Legacy positional path: sample dicts carry the GT fields directly.
    if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
        return None
    for i in range(len(dataset)):
        item = dataset[i]
        transcription = str(item.get("transcription", "") or "")
        rows.append(
            {
                "id": str(item.get("question_id", i)),
                "question_text": str(item.get("question_text", transcription) or ""),
                "transcription": transcription,
                "relevance": _build_relevant_from_item(item),
                "short_answer": item.get("short_answer") or item.get("answer"),
            }
        )
    return rows


def _publish_source_itemsets(s: RunState, dataset: Any) -> None:
    """Publish per-item source artifacts as keyed ``ItemSet``s (architecture W1/A1/A3).

    dataset_source is the sole ground-truth producer (Phase 3): asr/audio_embedding
    published reference_transcription/relevant_docs before becoming pure transforms, so
    the values here are byte-identical (transcription == question_text today)."""
    rows = _source_rows(dataset)
    if not rows:
        return
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        dupes = len(ids) - len(set(ids))
        logger.warning(
            "dataset_source: %d duplicate question_ids — ground truth NOT published, so "
            "WER/CER + retrieval metrics will be EMPTY. Give the dataset unique sample_ids.",
            dupes,
        )
        return

    from ..item_set import ItemSet

    s.put_artifact(
        "reference_text", ItemSet(ids, [r["question_text"] for r in rows])
    )
    s.put_artifact(
        "reference_transcription", ItemSet(ids, [r["transcription"] for r in rows])
    )
    s.put_artifact("relevant_docs", ItemSet(ids, [r["relevance"] for r in rows]))

    answers = [r["short_answer"] for r in rows]
    if any(a is not None for a in answers):
        s.put_artifact(
            "short_answers", ItemSet(ids, [str(a) if a is not None else "" for a in answers])
        )
    logger.debug("dataset_source: published %d keyed source items", len(ids))


def _publish_corpus_itemset(s: RunState, dataset: Any) -> None:
    """Publish the corpus as a doc_id-keyed ``ItemSet`` (§4.1 T1).

    The corpus rides the bus like every per-item artifact — the prerequisite for
    corpus-axis transforms (an ``axis: docs`` augmenter republishes a perturbed
    ``corpus`` and downstream ``corpus_embedding`` picks the newest producer).
    Positional consumers are untouched: ``get_artifact`` unwraps to the docs list.
    Duplicate/missing doc_ids fall back to positional ids (best effort, logged).
    """
    corpus = (
        dataset.get_corpus() if hasattr(dataset, "get_corpus") else None
    ) or []
    if not corpus:
        return
    from ..item_set import ItemSet

    ids = [
        str(doc.get("doc_id") or i) if isinstance(doc, dict) else str(i)
        for i, doc in enumerate(corpus)
    ]
    if len(set(ids)) != len(ids):
        logger.warning("dataset_source: duplicate doc_ids — positional corpus ids used")
        ids = [str(i) for i in range(len(corpus))]
    s.put_artifact("corpus", ItemSet(ids, list(corpus)))
    logger.debug("dataset_source: published corpus ItemSet (%d docs)", len(corpus))


def _publish_vector_columns(s: RunState, dataset: Any) -> None:
    """Publish precomputed-vector columns (§4.1 T4) when the schema declares them.

    A dataset whose ``fields`` map a column to ``query_vectors``/``corpus_vectors``
    feeds retrieval directly — third-party embeddings without an embed brick.
    Query vectors: each question's ``embedding`` (attr or metadata); corpus
    vectors: each doc's ``embedding`` key → a :class:`CorpusVectors` with the
    descriptor-declared ``embedding_space`` (rides ``params``).
    """
    params = s.node_params
    declared = set((params.get("fields") or {}).values())
    if not declared & {"query_vectors", "corpus_vectors"}:
        return
    import numpy as np

    from ..item_set import ItemSet

    space = params.get("embedding_space")

    if "query_vectors" in declared:
        questions = getattr(dataset, "questions", None) or []
        embeddings = []
        ids = []
        for i, q in enumerate(questions):
            emb = getattr(q, "embedding", None)
            if emb is None:
                emb = (getattr(q, "metadata", None) or {}).get("embedding")
            if emb is None:
                break
            ids.append(str(getattr(q, "question_id", i)))
            embeddings.append(np.asarray(emb, dtype=np.float32))
        if embeddings and len(embeddings) == len(questions):
            s.put_items("query_vectors", ItemSet(ids, embeddings))
            logger.info(
                "dataset_source: published %d precomputed query vectors", len(ids)
            )
        elif questions:
            logger.warning(
                "dataset_source: query_vectors column declared but not every "
                "question carries an embedding — skipped"
            )

    if "corpus_vectors" in declared:
        corpus = (
            dataset.get_corpus() if hasattr(dataset, "get_corpus") else None
        ) or []
        vecs = [
            doc.get("embedding") for doc in corpus if isinstance(doc, dict)
        ]
        if corpus and all(v is not None for v in vecs) and len(vecs) == len(corpus):
            from ...services.corpus_index import CorpusVectors

            cv = CorpusVectors(
                vectors=np.asarray([np.asarray(v, dtype=np.float32) for v in vecs]),
                payloads=list(corpus),
                space=space,
            )
            s.put_artifact("corpus_vectors", cv)
            logger.info(
                "dataset_source: published %d precomputed corpus vectors", len(corpus)
            )
        elif corpus:
            logger.warning(
                "dataset_source: corpus_vectors column declared but not every "
                "doc carries an 'embedding' — skipped"
            )


def _stage_dataset_union(s: RunState) -> None:
    """Union the question-axis ItemSets of every bound producer (§4.1 T3).

    For each question-axis artifact (query_audio refs, query_text, relevant_docs,
    short_answers, reference_text): concatenate all published producers' ItemSets in
    producer order; a duplicate query_id across sources is a hard
    ``ConfigurationError`` (paired metrics would silently mis-join otherwise).
    """
    from ...errors import ConfigurationError
    from ..item_set import ItemSet

    union_arts = (
        "query_audio",
        "query_text",
        "relevant_docs",
        "short_answers",
        "reference_text",
    )
    published_any = False
    for name in union_arts:
        sets = [
            s.ctx.get(pid, name)
            for pid in s._producers(name)
            if s.ctx.has(pid, name)
        ]
        sets = [x for x in sets if isinstance(x, ItemSet) and x.ids]
        if not sets:
            continue
        ids: list = []
        values: list = []
        for item_set in sets:
            ids.extend(str(i) for i in item_set.ids)
            values.extend(item_set.values)
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ConfigurationError(
                f"dataset_union: duplicate query_id(s) across sources for "
                f"'{name}': {sorted(dupes)[:5]} — paired metrics would mis-join. "
                f"Namespace the datasets' ids."
            )
        s.put_artifact(name, ItemSet(ids, values))
        published_any = True
    if published_any:
        logger.info("dataset_union: published unioned question-axis artifacts")
    else:
        logger.warning("dataset_union: nothing to union (no bound ItemSets)")
