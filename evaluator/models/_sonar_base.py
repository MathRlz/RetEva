"""Shared base for SONAR embedding models.

SONAR speech (``a2e/sonar.py``) and text (``t2e/sonar.py``) encoders both wrap a
device-bound inference pipeline that must be rebuilt to move devices. This mixin
holds the common install hint, device coercion, and ``to()`` rebuild logic; each
subclass supplies ``_build_pipeline(device)``.
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
