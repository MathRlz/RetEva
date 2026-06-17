"""Streaming reduction primitives for windowed evaluation (architecture-improvements §2).

The scale unlock is to evaluate in windows (embed → retrieve → score a slice, fold its
sufficient statistics into a running accumulator, release the slice) instead of holding the
whole result set in RAM. Most metrics (MRR, Recall@k, NDCG, WER, …) are *mean-reducible*: a
window contributes a (sum, count), and the overall mean is ``Σsum / Σcount`` regardless of how
the items were batched. :class:`RunningMean` is that fold; :class:`StreamingMetrics` keeps one
per metric name.

This is the reduction core a streaming executor driver builds on; it is exact (not an
approximation) and order-independent.
"""

from __future__ import annotations

from typing import Dict, Iterable


class RunningMean:
    """An exact running mean folded from windows of per-item values or (mean, n) pairs."""

    __slots__ = ("_sum", "_n")

    def __init__(self) -> None:
        self._sum = 0.0
        self._n = 0

    def update(self, values: Iterable[float]) -> "RunningMean":
        """Fold a window of per-item scores."""
        for v in values:
            self._sum += float(v)
            self._n += 1
        return self

    def update_mean(self, mean: float, n: int) -> "RunningMean":
        """Fold a window summarized as its (mean, count) — the sufficient statistic."""
        n = int(n)
        self._sum += float(mean) * n
        self._n += n
        return self

    @property
    def mean(self) -> float:
        return self._sum / self._n if self._n else 0.0

    @property
    def n(self) -> int:
        return self._n


class StreamingMetrics:
    """A bag of named :class:`RunningMean`s — fold per-window metric dicts, read the overall."""

    def __init__(self) -> None:
        self._acc: Dict[str, RunningMean] = {}

    def update_window(self, per_item: Dict[str, Iterable[float]]) -> "StreamingMetrics":
        """Fold one window's ``{metric: [per-item scores]}``."""
        for name, values in per_item.items():
            self._acc.setdefault(name, RunningMean()).update(values)
        return self

    def update_window_means(self, means: Dict[str, "tuple[float, int]"]) -> "StreamingMetrics":
        """Fold one window's ``{metric: (mean, n)}`` sufficient statistics."""
        for name, (mean, n) in means.items():
            self._acc.setdefault(name, RunningMean()).update_mean(mean, n)
        return self

    def result(self) -> Dict[str, Dict[str, float]]:
        """Overall ``{metric: {"mean": …, "n": …}}`` — the same shape a report branch uses."""
        return {name: {"mean": acc.mean, "n": acc.n} for name, acc in self._acc.items()}
