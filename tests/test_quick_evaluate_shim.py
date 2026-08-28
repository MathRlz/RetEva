"""W7 pin: `quick_evaluate` rides the legacy `model.pipeline_mode` load shim.

`api._build_quick_eval_config_dict` still emits `model: {pipeline_mode: ...}`;
`config/loading.build_from_dict` must keep rewriting that key to
`graph_override["template"]` (and never reject it). If either half changes,
quick_evaluate breaks — this is the only surface still using the shim.
"""

from pathlib import Path

from evaluator.api import _build_quick_eval_config_dict
from evaluator.config import EvaluationConfig


def _quick_dict(tmp_path):
    return _build_quick_eval_config_dict(
        audio_path=Path(tmp_path), asr_type="whisper", emb_type="labse",
        model="whisper", embedding="labse", model_size=None, embedding_size=None,
        k=5, batch_size=1, trace_limit=0, corpus_path=None, kwargs={},
    )


def test_quick_eval_dict_still_emits_pipeline_mode(tmp_path):
    d = _quick_dict(tmp_path)
    assert d["model"]["pipeline_mode"] == "asr_text_retrieval"


def test_loader_shim_rewrites_pipeline_mode_to_template(tmp_path):
    config = EvaluationConfig.from_dict(_quick_dict(tmp_path))
    # the key is consumed, not rejected…
    assert not hasattr(config.model, "pipeline_mode") or not getattr(
        config.model, "pipeline_mode", None
    )
    # …and lands as the graph template hint that drives the build
    assert (config.graph_override or {}).get("template") == "asr_text_retrieval"
