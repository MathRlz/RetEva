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
    """All-But-The-Top: subtract the mean ``mu`` then remove the top principal component(s).

    Args:
        emb: ``(B, H)`` embeddings.
        mu:  ``(H,)`` mean used during fitting.
        pc1: top principal component(s) — ``(H,)`` for a single component or ``(k, H)`` for the
             top-``k`` (each row a unit vector).
    Returns ``(B, H)``.
    """
    v = emb - mu
    comps = pc1.unsqueeze(0) if pc1.dim() == 1 else pc1  # (k, H)
    # remove the projection of each row of `v` onto every principal component
    coeffs = v @ comps.transpose(0, 1)  # (B, k)
    return v - coeffs @ comps  # (B, H)
