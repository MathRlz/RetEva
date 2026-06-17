"""Post-hoc embedding transforms for the mean-pooling audio variants (whitening / ABTT).

Both consume **precomputed statistics** carried as buffers in the trained checkpoint (see
``attention_pool.py``): whitening needs a mean ``m`` + whitening matrix ``W``; ABTT needs a mean
``mu`` + the top principal component(s) ``pc1``. The stats are produced by the training pipeline;
here we only *apply* the transform.

NOTE — these formulas must match the transform used when the stats were fit (otherwise the loaded
buffers are invalid). They implement the standard definitions:
* whitening: ``(x - m) @ W``
* ABTT (All-But-The-Top): subtract the mean, then remove the projection onto the top PC(s).
"""

from __future__ import annotations

import torch


def whiten_batch(emb: torch.Tensor, m: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Whiten a batch of embeddings: center by ``m`` then apply the whitening matrix ``W``.

    Args:
        emb: ``(B, H)`` embeddings.
        m:   ``(H,)`` mean used during fitting.
        W:   ``(H, H)`` whitening matrix (e.g. PCA / ZCA).
    Returns ``(B, H)``.
    """
    return (emb - m) @ W


def abtt_batch(emb: torch.Tensor, mu: torch.Tensor, pc1: torch.Tensor) -> torch.Tensor:
    """All-But-The-Top: subtract the mean ``mu``, remove the top principal component(s), then
    **row-wise L2 normalize**.

    The final normalization is load-bearing and must match the training-time ABTT
    (``apm_new/apm/postprocessing.py:abtt_batch`` ends with
    ``X = X / (X.norm(dim=1, keepdim=True) + 1e-9)``): the projection head was trained on
    unit-norm post-ABTT inputs, so omitting it feeds the projection a mismatched input
    distribution and collapses retrieval. (Whitening is asymmetric here — training
    ``whiten_batch`` is ``(E - m) @ W`` with no normalize, so :func:`whiten_batch` must NOT
    normalize.)

    Args:
        emb: ``(B, H)`` embeddings.
        mu:  the mean used during fitting — ``(H,)`` or ``(1, H)`` (a trailing singleton from
             the training pipeline is squeezed).
        pc1: top principal component(s) — accepts ``(H,)`` / ``(1, H)`` / ``(H, 1)`` for a single
             component or ``(k, H)`` for the top-``k`` (each a unit vector). Reshaped to ``(k, H)``.
    Returns ``(B, H)`` (each row L2-normalized).
    """
    h = emb.shape[-1]
    v = emb - mu.reshape(-1)  # (H,) broadcasts over (B, H); tolerates a (1, H) mean
    comps = pc1.reshape(-1, h)  # (k, H) from (H,), (1, H), (H, 1) → single component, or (k, H)
    # remove the projection of each row of `v` onto every principal component
    coeffs = v @ comps.transpose(0, 1)  # (B, k)
    out = v - coeffs @ comps  # (B, H)
    # L2 normalize each row — matches the training-time ABTT the projection head was fit on.
    return out / (out.norm(dim=-1, keepdim=True) + 1e-9)
