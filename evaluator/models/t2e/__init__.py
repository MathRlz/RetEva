"""T2E (Text-to-Embedding) models."""

from .jina import JinaV4Model
from .clip import ClipModel
from .labse import LabseModel
from .nemotron import NemotronModel
from .bgem3 import BgeM3Model
from .sonar import SonarTextModel
# The CLAP-style wrapper is one class registered in BOTH families (shared contrastive
# space); it lives under a2e/ but must import here too, or the text registry — which only
# imports this package — never sees its `clap_text` registration.
from ..a2e.clap_style import MultimodalClapStyleModel  # noqa: F401
# Same reason: EamAlignmentModel is one class registered in both families (a genuinely
# joint audio-text space) and lives under a2e/.
from ..a2e.eam_alignment import EamAlignmentModel  # noqa: F401

__all__ = [
    "JinaV4Model",
    "ClipModel",
    "LabseModel",
    "NemotronModel",
    "BgeM3Model",
    "SonarTextModel",
    "MultimodalClapStyleModel",
    "EamAlignmentModel",
]
