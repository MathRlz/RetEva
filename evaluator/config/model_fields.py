"""Single source of truth for the model-family config field layout.

Two user-input layers map onto ``config.model``: the CLI (``cli/parser.py``,
attribute overrides on a built config) and the WebUI form (``webapi/form_config.py``,
a dict overlaid on a preset). Both need the same per-family field names
(``asr_model_type`` / ``asr_model_name`` / ``asr_size`` / …) and the same
*model-coherence* rule — when the chosen model **type** changes, the inherited
``name``/``adapter`` (and maybe ``size``) belong to a different model and must be
dropped, or the run gets a whisper type with a seamless-m4t name and crashes. Declaring
the families here keeps that knowledge (and the coherence fix) in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, MutableMapping, Tuple


@dataclass(frozen=True)
class ModelFamilyFields:
    """The ``config.model`` field names for one model family (ASR / text / audio emb)."""

    prefix: str  # "asr" / "text_emb" / "audio_emb"
    registry_family: str  # registry key: "asr" / "text_embedding" / "audio_embedding"

    @property
    def type_field(self) -> str:
        return f"{self.prefix}_model_type"

    @property
    def name_field(self) -> str:
        return f"{self.prefix}_model_name"

    @property
    def adapter_field(self) -> str:
        return f"{self.prefix}_adapter_path"

    @property
    def size_field(self) -> str:
        return f"{self.prefix}_size"

    @property
    def device_field(self) -> str:
        return f"{self.prefix}_device"

    @property
    def params_field(self) -> str:
        return f"{self.prefix}_params"

    @property
    def model_field_path(self) -> str:
        """The node registry's ``model_field`` for this family (``model.<type_field>``)."""
        return f"model.{self.type_field}"


MODEL_FAMILY_FIELDS: Tuple[ModelFamilyFields, ...] = (
    ModelFamilyFields("asr", "asr"),
    ModelFamilyFields("text_emb", "text_embedding"),
    ModelFamilyFields("audio_emb", "audio_embedding"),
)

# model_field path ("model.asr_model_type") → registry family ("asr"). Used by the
# builder palette to pick the right model-choice registry for a node.
MODEL_FIELD_FAMILY: Dict[str, str] = {
    fam.model_field_path: fam.registry_family for fam in MODEL_FAMILY_FIELDS
}


def strip_stale_family_fields(
    model: MutableMapping[str, object], fam: ModelFamilyFields, *, drop_size: bool
) -> None:
    """Drop a family's inherited ``name``/``adapter`` (+ ``size`` when ``drop_size``).

    Call when the model **type** for ``fam`` has changed relative to the inherited
    config so the leftover identity of the *previous* model can't survive (the
    whisper+m4t incoherence). In-place; absent keys are ignored.
    """
    model.pop(fam.name_field, None)
    model.pop(fam.adapter_field, None)
    if drop_size:
        model.pop(fam.size_field, None)
