"""Stage-scoped model load/offload for the DAG executor.

The evaluation DAG (``pipeline/stage_graph.py`` + ``evaluation/phased._execute_stage_graph``)
runs stage handlers in topological order. A stage may declare a :class:`ModelSpec`; the
executor uses :class:`ServiceModelManager` to load that model onto the device before the
stage runs and to release it after the last stage that needs it — so a TTS model is gone
before the embedder loads (no co-resident native runtimes), and only the active stage's
model occupies the device.

The model handle is a lazy service (``start``/``stop``/``move_to_device``/``get``, e.g.
``services.model_services.FactoryModelService``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence


@dataclass
class ModelSpec:
    """A model the executor loads before a stage and offloads after its last use.

    Args:
        key: Lifecycle id. Stages sharing a model share a key (loaded once, offloaded
            after the last stage that uses it).
        service: Lazy model service (``start``/``stop``/``move_to_device``/``get``). May be
            ``None`` for a stage that declares a key but loads the model itself.
        device: Device to place the model on while the stage runs.
    """

    key: str
    service: Any = None
    device: str = "cpu"


class ServiceModelManager:
    """Loads/offloads :class:`ModelSpec` services, honoring an offload policy.

    ``offload_policy="on_finish"`` (default) fully releases a model after its last use —
    freeing device + host memory and any native runtime it holds, which is what keeps an
    embedder from inheriting a TTS model's poisoned native state. ``"never"`` keeps models
    resident for reuse across runs.
    """

    def __init__(self, offload_policy: str = "on_finish") -> None:
        self.offload_policy = offload_policy

    def acquire(self, spec: ModelSpec) -> Any:
        """Load ``spec`` onto its device and return the model handle (idempotent)."""
        svc = spec.service
        if svc is None:
            return None
        mover = getattr(svc, "move_to_device", None)
        if callable(mover):
            mover(spec.device)
        starter = getattr(svc, "start", None)
        if callable(starter):
            starter()
        getter = getattr(svc, "get", None)
        return getter() if callable(getter) else svc

    def release(self, spec: ModelSpec) -> None:
        """Offload ``spec`` unless the policy says to keep it resident."""
        if self.offload_policy == "never" or spec.service is None:
            return
        stopper = getattr(spec.service, "stop", None)
        if callable(stopper):
            stopper()


def last_use_positions(specs_in_order: Sequence[Optional[ModelSpec]]) -> Dict[int, ModelSpec]:
    """Map each stage position to the spec (if any) that is last used right there.

    ``specs_in_order[i]`` is the :class:`ModelSpec` for the i-th stage (or ``None``). The
    returned ``{position: spec}`` tells the executor which model to release after running
    the stage at that position.
    """
    last_index: Dict[str, int] = {}
    spec_by_key: Dict[str, ModelSpec] = {}
    for i, spec in enumerate(specs_in_order):
        if spec is not None:
            last_index[spec.key] = i
            spec_by_key[spec.key] = spec
    return {i: spec_by_key[key] for key, i in last_index.items()}


def acquire_for(manager: Optional[ServiceModelManager], spec: Optional[ModelSpec]) -> Any:
    """Acquire ``spec`` via ``manager`` if both are present; else return ``None``."""
    if manager is None or spec is None:
        return None
    return manager.acquire(spec)
