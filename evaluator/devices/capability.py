"""Usable-GPU detection.

Single source of truth for *which* physical CUDA device indices are actually
safe to run compute on. ``torch.cuda.device_count()`` happily counts devices the
torch build cannot drive — most notably integrated GPUs (e.g. an AMD RAPHAEL
iGPU, ``gfx1036``) on a box whose ROCm wheel was compiled only for the discrete
card (``gfx1100``). Placing a model on such a device SIGSEGVs the process.

We filter by comparing each device's arch against ``torch.cuda.get_arch_list()``
(the archs the torch build was compiled for). This generalizes to NVIDIA too: a
device whose ``sm_XX`` is absent from the build's arch list is excluded.
"""

import os
from typing import List, Optional

from ..logging_config import get_logger

logger = get_logger(__name__)

_ENV_OVERRIDE = "EVALUATOR_VISIBLE_GPUS"

# Module-level cache. ``None`` means "not computed yet".
_cache: Optional[List[int]] = None


def reset_cache() -> None:
    """Clear the cached usable-index list (test hook / hardware change)."""
    global _cache
    _cache = None


def _normalize_arch(arch: str) -> str:
    """Lowercase and strip ROCm feature flags: ``gfx1100:sramecc+:xnack-`` -> ``gfx1100``."""
    return arch.split(":")[0].strip().lower()


def _device_arch(props, idx: int) -> Optional[str]:
    """Return the normalized arch for a device, or None if undeterminable."""
    gcn = getattr(props, "gcnArchName", None)
    if gcn:  # ROCm
        return _normalize_arch(gcn)
    try:  # NVIDIA: build sm_XX from compute capability
        import torch

        major, minor = torch.cuda.get_device_capability(idx)
        return f"sm_{major}{minor}"
    except (RuntimeError, ValueError, OSError, AttributeError):
        return None


def _env_override(device_count: int) -> Optional[List[int]]:
    """Parse ``EVALUATOR_VISIBLE_GPUS`` into a bounds-checked index list, if set."""
    raw = os.environ.get(_ENV_OVERRIDE)
    if not raw:
        return None
    indices: List[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            idx = int(tok)
        except ValueError:
            logger.warning("%s: ignoring non-integer token %r", _ENV_OVERRIDE, tok)
            continue
        if 0 <= idx < device_count:
            indices.append(idx)
        else:
            logger.warning("%s: index %d out of range (device_count=%d)", _ENV_OVERRIDE, idx, device_count)
    return indices


def usable_gpu_indices() -> List[int]:
    """Return physical CUDA device indices safe for compute (cached).

    Empty when CUDA is unavailable. Conservative: a device is excluded only on a
    *positive* arch mismatch, so a good card is never dropped on ambiguity (e.g.
    an empty/undeterminable arch list keeps every device).
    """
    global _cache
    if _cache is not None:
        return _cache

    try:
        import torch
    except ImportError:
        _cache = []
        return _cache

    if not torch.cuda.is_available():
        _cache = []
        return _cache

    device_count = torch.cuda.device_count()

    override = _env_override(device_count)
    if override is not None:
        logger.info("%s set -> using GPU indices %s", _ENV_OVERRIDE, override)
        _cache = override
        return _cache

    try:
        supported = {_normalize_arch(a) for a in torch.cuda.get_arch_list()}
    except (RuntimeError, ValueError, OSError, AttributeError, TypeError):
        supported = set()

    usable: List[int] = []
    excluded: List[str] = []
    for idx in range(device_count):
        try:
            props = torch.cuda.get_device_properties(idx)
        except (RuntimeError, ValueError, OSError):
            continue
        arch = _device_arch(props, idx)
        # Keep when we cannot determine support, or arch is in the build list.
        if not supported or arch is None or arch in supported:
            usable.append(idx)
        else:
            excluded.append(f"cuda:{idx} ({props.name!r}, {arch})")

    if excluded:
        logger.info(
            "GPU filter: usable=%s; excluded unsupported arch (not in %s): %s",
            usable,
            sorted(supported),
            ", ".join(excluded),
        )

    _cache = usable
    return _cache


def usable_gpu_count() -> int:
    """Number of compute-usable CUDA devices."""
    return len(usable_gpu_indices())


def is_usable_index(idx: int) -> bool:
    """True if physical CUDA index ``idx`` is compute-usable."""
    return idx in usable_gpu_indices()
