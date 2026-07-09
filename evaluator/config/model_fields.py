"""Single source of truth for the model-family → config-field mapping.

The builder palette maps a node's registry ``model_field`` (``model.<prefix>_model_type``)
to the model-choice registry family, so it can show the right model picker for a node.
"""

from __future__ import annotations

from typing import Dict

# model_field path ("model.asr_model_type") → registry family ("asr").
MODEL_FIELD_FAMILY: Dict[str, str] = {
    "model.asr_model_type": "asr",
    "model.text_emb_model_type": "text_embedding",
    "model.audio_emb_model_type": "audio_embedding",
}
