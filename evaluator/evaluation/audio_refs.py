"""Audio refs on the bus (§4.1 P4).

The ``query_audio`` artifact carries **refs** (file paths keyed by query_id), never
decoded waveforms — memory stays bounded and audio-axis nodes (augment_audio) can
republish perturbed refs. ASR / audio-embedding consume refs **bus-first**:
when the newest bound refs differ from the dataset's own audio paths, the dataset is
wrapped in a :class:`RefAudioDatasetView` that decodes from the refs while keeping the
base item metadata (transcription, relevance, …) joined by question_id; when they match
(default graphs), the dataset object passes through untouched — byte-identical parity.
"""

from __future__ import annotations

from typing import Any, Optional

from ..logging_config import get_logger
from .item_set import ItemSet, lineage_parent as _lineage_parent

logger = get_logger(__name__)


class RefAudioDatasetView:
    """Dataset adapter: base items with audio decoded from bus refs.

    Order and cardinality follow the refs (supports fan-out: a variant id maps to
    its lineage parent's metadata). Decoding is lazy per item (paths, not arrays).
    """

    def __init__(self, base: Any, refs: ItemSet, extra_bases: Optional[list] = None):
        self._base = base
        self._ids = [str(i) for i in refs.ids]
        self._paths = {str(i): str(v) for i, v in zip(refs.ids, refs.values)}
        # id → (owning base, index). First base wins; extra bases (multi-source /
        # union graphs) fill the rest so a unioned ref set resolves metadata too.
        # Question-based datasets expose ``questions`` (question_id); sample-based ones
        # (AudioSamplesQueryDataset — admed_voice) expose ``samples`` (sample_id).
        self._owner: dict = {}
        for b in [base, *(extra_bases or [])]:
            if b is None:
                continue
            rows = getattr(b, "questions", None) or getattr(b, "samples", None) or []
            for i, q in enumerate(rows):
                qid = str(
                    getattr(q, "question_id", None)
                    or getattr(q, "sample_id", None)
                    or i
                )
                self._owner.setdefault(qid, (b, i))

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, idx: int) -> dict:
        from ..datasets.core import load_audio_file

        qid = self._ids[idx]
        owner = self._owner.get(qid) or self._owner.get(_lineage_parent(qid))
        item = self._base_metadata(owner) if owner is not None else {}
        waveform, sr = load_audio_file(self._paths[qid])
        item["audio_array"] = waveform.squeeze().numpy()
        item["sampling_rate"] = int(sr)
        item["question_id"] = qid
        return item

    @staticmethod
    def _base_metadata(owner: tuple) -> dict:
        """Metadata for a base question WITHOUT going through ``base.__getitem__``.

        The base's item accessor decodes the base waveform (discarded here — the ref
        audio replaces it) and refuses questions without an ``audio_path``. That check is
        the base's business, not the view's: on a journal-resumed run the tts levels are
        skipped, the reloaded questions carry no audio_path, and the old
        ``dict(base[idx])`` call crashed every audio consumer with a misleading
        "has no audio_path" — even though the refs held all the audio needed.
        """
        base, i = owner
        rows = getattr(base, "questions", None) or getattr(base, "samples", None)
        q = rows[i]
        # question-based rows carry question_text; sample-based rows (admed_voice) carry
        # the transcription as the text.
        text = getattr(q, "question_text", None) or getattr(q, "transcription", None)
        meta = getattr(q, "metadata", None) or {}
        return {
            "transcription": text,
            "question_text": text,
            "question_id": (
                getattr(q, "question_id", None) or getattr(q, "sample_id", None) or i
            ),
            "groundtruth_doc_ids": (
                getattr(q, "groundtruth_doc_ids", None)
                or meta.get("groundtruth_doc_ids")
            ),
            "relevance_grades": (
                getattr(q, "relevance_grades", None) or meta.get("relevance_grades")
            ),
            "language": getattr(q, "language", None),
            "metadata": meta,
        }

    # Some consumers introspect questions (e.g. relevance derivation) — expose base's.
    @property
    def questions(self):  # pragma: no cover - passthrough
        return getattr(self._base, "questions", None)

    def get_corpus(self):
        return self._base.get_corpus() if hasattr(self._base, "get_corpus") else []


def audio_refs_from_questions(questions: Any) -> "Optional[tuple]":
    """``(ids, paths)`` when every question carries an ``audio_path``; else None."""
    if not questions:
        return None
    ids, paths = [], []
    for i, q in enumerate(questions):
        path = getattr(q, "audio_path", None)
        if not path:
            return None
        ids.append(str(getattr(q, "question_id", i)))
        paths.append(str(path))
    if len(set(ids)) != len(ids):
        return None
    return ids, paths


def publish_audio_refs(s: Any, dataset: Any) -> None:
    """Publish ``query_audio`` as an ItemSet of audio REFS — file paths, never decoded
    waveforms (§4.1 P4; memory rule). The bus ref is what audio-axis nodes
    (augment_audio) republish; ASR/audio-embedding consume refs bus-first and fall
    back to the dataset object when the refs match it (parity-preserving)."""
    refs = audio_refs_from_questions(getattr(dataset, "questions", None))
    if refs is None:
        return
    from .item_set import ItemSet

    ids, paths = refs
    s.put_artifact("query_audio", ItemSet(ids, paths))
    logger.debug("dataset_source: published %d audio refs", len(ids))


def resolve_audio_dataset(s: Any, dataset: Any) -> Any:
    """The dataset an audio consumer should iterate: ref-view when the bus refs
    diverged from the dataset's own audio paths, else the dataset itself (parity)."""
    refs = None
    try:
        refs = s.keyed_items("query_audio")
    except Exception as exc:
        logger.debug("audio_refs: keyed_items('query_audio') failed: %s", exc)
        refs = None
    if not isinstance(refs, ItemSet) or not refs.ids:
        # Falling back to the dataset object is correct for graphs whose refs mirror it,
        # but when this node is explicitly BOUND to a query_audio producer that published
        # nothing, the fallback usually ends in a misleading "no audio_path" crash — say
        # so here, at the decision point.
        try:
            bound = [pid for art, pid in getattr(s.current_node, "bindings", ()) if art == "query_audio"]
        except Exception:  # noqa: BLE001 - diagnostics only
            bound = []
        if bound:
            logger.warning(
                "audio_refs: node '%s' is bound to query_audio from %s but no refs were "
                "published — falling back to the dataset's own audio paths",
                getattr(s.current_node, "id", "?"), bound,
            )
        return dataset
    # Refs must actually be path strings (a positional wrap of decoded arrays is not
    # a ref publish — leave those to the legacy path).
    if not all(isinstance(v, str) for v in refs.values):
        return dataset
    own = audio_refs_from_questions(getattr(dataset, "questions", None))
    if own is not None and list(refs.ids) == own[0] and list(refs.values) == own[1]:
        return dataset  # bus mirrors the dataset → no wrap, byte-identical path
    extra = list(getattr(s, "dataset_sources", {}).values() or [])
    return RefAudioDatasetView(dataset, refs, extra_bases=extra)
