"""Full-text context selection must never drive the embedder from two threads at once.

The answer loop is concurrent, and `select_full_text_context` embeds article chunks inside it —
a torch forward pass plus a cache/sqlite lookup, neither of which is thread-safe. Four workers
through one sentence-transformers model wedged a 1209-question sweep for twelve hours, with no
timeout able to break it (the client's own ceiling is timeout_s x retries).
"""

import threading

import numpy as np

from evaluator.evaluation.answer_gen import select_full_text_context
from evaluator.llm_client.parallel import map_completions


class _JealousEmbedder:
    """Fails loudly if a second thread enters while one is inside — what torch does silently."""

    def __init__(self):
        self._inside = 0
        self._guard = threading.Lock()
        self.calls = 0

    def process_batch(self, texts, **kw):
        with self._guard:
            self._inside += 1
            self.calls += 1
            if self._inside > 1:
                raise AssertionError("embedder entered concurrently")
        try:
            # long enough that unserialized workers would overlap here
            threading.Event().wait(0.02)
            return np.array([[float(len(t))] for t in texts], dtype="float32")
        finally:
            with self._guard:
                self._inside -= 1


class _Cfg:
    context_source = "full_text"
    context_chunk_chars = 60
    context_chunks = 2
    context_max_chars = 4000


_ARTICLE = "\n\n".join(
    f"Section {i}. Aspirin and outcome number {i} in this cohort." for i in range(12)
)


def test_concurrent_workers_do_not_embed_simultaneously():
    embedder = _JealousEmbedder()
    corpus = {f"d{i}": {"doc_id": f"d{i}", "metadata": {"full_text": _ARTICLE}} for i in range(8)}

    def _one(i):
        return select_full_text_context(
            f"question {i}?", {"doc_id": f"d{i}"}, corpus, _Cfg(), embedder
        )

    out = map_completions(range(8), _one, workers=4, desc="ctx")
    assert len(out) == 8
    assert all(o for o in out)              # every doc produced context
    assert embedder.calls == 16             # 2 calls per doc (chunks + query), none lost


def test_still_works_without_an_embedder():
    corpus = {"d1": {"doc_id": "d1", "metadata": {"full_text": _ARTICLE}}}
    picked = select_full_text_context("q?", {"doc_id": "d1"}, corpus, _Cfg(), None)
    assert picked                            # falls back to the first chunks, no crash
