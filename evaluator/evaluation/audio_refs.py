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
from .item_set import ItemSet

logger = get_logger(__name__)


def _lineage_parent(qid: str) -> str:
    """``q42·aug0`` → ``q42`` (fan-out variants resolve metadata via their parent)."""
    return qid.split("·", 1)[0]


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
        self._owner: dict = {}
        for b in [base, *(extra_bases or [])]:
            if b is None:
                continue
            for i, q in enumerate(getattr(b, "questions", None) or []):
                qid = str(getattr(q, "question_id", i))
                self._owner.setdefault(qid, (b, i))

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, idx: int) -> dict:
        from ..datasets.core import load_audio_file

        qid = self._ids[idx]
        owner = self._owner.get(qid) or self._owner.get(_lineage_parent(qid))
        item = dict(owner[0][owner[1]]) if owner is not None else {}
        waveform, sr = load_audio_file(self._paths[qid])
        item["audio_array"] = waveform.squeeze().numpy()
        item["sampling_rate"] = int(sr)
        item["question_id"] = qid
        return item

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
    except Exception:
        refs = None
    if not isinstance(refs, ItemSet) or not refs.ids:
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
