"""Post-hoc embedding transforms for the mean-pooling audio variants (whitening / ABTT).

Formulas here must match the transform used when the checkpoint's stats were fit
(see ``attention_pool.py``), or the loaded buffers are invalid.
"""

from __future__ import annotations

import torch


def whiten_batch(emb: torch.Tensor, m: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Whiten a batch of embeddings: center by ``m`` then apply the whitening matrix ``W``."""
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
    """
    h = emb.shape[-1]
    v = emb - mu.reshape(-1)  # (H,) broadcasts over (B, H); tolerates a (1, H) mean
    comps = pc1.reshape(-1, h)  # (k, H) from (H,), (1, H), (H, 1) → single component, or (k, H)
    # remove the projection of each row of `v` onto every principal component
    coeffs = v @ comps.transpose(0, 1)  # (B, k)
    out = v - coeffs @ comps  # (B, H)
    # L2 normalize each row — matches the training-time ABTT the projection head was fit on.
    return out / (out.norm(dim=-1, keepdim=True) + 1e-9)
