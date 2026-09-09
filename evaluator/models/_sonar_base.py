"""Shared base for SONAR embedding models.

Both wrap a device-bound pipeline that must be rebuilt to move devices; this
mixin holds the shared device-coercion and rebuild logic.
"""
from abc import ABC, abstractmethod

import torch

SONAR_INSTALL_HINT = (
    "SONAR not installed. Install with: pip install 'evaluator[sonar]'\n"
    "(pulls sonar-space + fairseq2)."
)


class SonarPipelineMixin(ABC):
    """Device handling for SONAR pipeline-backed models."""

    @staticmethod
    def _coerce_device(device) -> torch.device:
        return device if isinstance(device, torch.device) else torch.device(device)

    @abstractmethod
    def _build_pipeline(self, device: torch.device):
        """Construct the SONAR pipeline bound to ``device``. Subclass implements."""
        raise NotImplementedError

    def to(self, device: torch.device):
        # SONAR pipelines bind device at construction; rebuild on move.
        self.device = self._coerce_device(device)
        self._pipeline = self._build_pipeline(self.device)
        return self
