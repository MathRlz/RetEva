"""Sparse (lexical) retrieval: minimal in-memory BM25 index.

Extracted from ``pipeline/retrieval_pipeline.py`` so strategy logic lives with
the other retrieval strategies; the pipeline only orchestrates.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from ...constants import MIN_NORM_THRESHOLD
from .scoring import payload_text, tokenize


class SparseBM25Index:
    """Minimal BM25 index for lexical retrieval over payload texts."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.payloads: List[Any] = []
        self.doc_lens: List[int] = []
        self.avgdl: float = 0.0
        self.doc_term_freqs: List[Counter] = []
        self.doc_freq: Dict[str, int] = defaultdict(int)
        self.doc_count: int = 0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return tokenize(text)

    def build(self, payloads: List[Any]) -> None:
        self.payloads = payloads
        self.doc_term_freqs = []
        self.doc_lens = []
        self.doc_freq = defaultdict(int)

        for payload in payloads:
            tokens = tokenize(payload_text(payload))
            tf = Counter(tokens)
            self.doc_term_freqs.append(tf)
            self.doc_lens.append(len(tokens))
            for token in tf.keys():
                self.doc_freq[token] += 1

        self.doc_count = len(payloads)
        self.avgdl = (
            (sum(self.doc_lens) / self.doc_count) if self.doc_count > 0 else 0.0
        )

    def search(self, query_text: str, k: int = 10) -> List[Tuple[Any, float]]:
        if self.doc_count == 0:
            return []

        q_tokens = tokenize(query_text)
        if not q_tokens:
            return []

        scores = np.zeros(self.doc_count, dtype=np.float32)
        n = self.doc_count

        for token in q_tokens:
            df = self.doc_freq.get(token, 0)
            if df == 0:
                continue

            idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))

            for i, tf in enumerate(self.doc_term_freqs):
                f = tf.get(token, 0)
                if f == 0:
                    continue

                dl = self.doc_lens[i]
                denom = f + self.k1 * (
                    1.0 - self.b + self.b * (dl / max(self.avgdl, MIN_NORM_THRESHOLD))
                )
                scores[i] += (
                    idf * (f * (self.k1 + 1.0)) / max(denom, MIN_NORM_THRESHOLD)
                )

        top_idx = np.argsort(-scores)[:k]
        return [
            (self.payloads[i], float(scores[i])) for i in top_idx if scores[i] > 0.0
        ]
