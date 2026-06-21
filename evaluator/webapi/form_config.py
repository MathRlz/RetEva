"""Form → validated run-config conversion for the WebAPI.

The config half of the form builder: take a submitted HTML form (flat string dict),
overlay it on the selected preset's full config (config-first — nothing the preset sets
is dropped), and produce a validated :class:`EvaluationConfig`. The UI/option-rendering
half lives in ``form_builder.py`` (which re-exports these names for back-compat).
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from fastapi.responses import HTMLResponse

from evaluator import ConfigurationError, EvaluationConfig
from evaluator.config.model_fields import MODEL_FAMILY_FIELDS, strip_stale_family_fields
from evaluator.datasets import validate_dataset_runtime_config
from evaluator.pipeline.factory import check_backend_dependencies


def load_config(
    payload_config: Dict[str, Any], *, auto_devices: bool
) -> EvaluationConfig:
    config = EvaluationConfig.from_dict(payload_config)
    if auto_devices:
        config = config.with_auto_devices()
    return config


#: Dataset path fields a form can set; confined under EVALUATOR_DATA_ROOT when it is set (L1).
_DATASET_PATH_FIELDS = (
    "questions_path", "corpus_path", "prepared_dataset_dir",
    "audio_dir", "data_path", "transcripts_file",
)


def _enforce_data_root(config: EvaluationConfig) -> None:
    """Confine form-supplied dataset paths under ``EVALUATOR_DATA_ROOT`` when it is set (L1).

    Opt-in: with the env unset (the default single-user / local deployment) there is no
    restriction, so absolute local paths keep working. When set — for a shared/multi-tenant
    server — a crafted ``../../etc/...`` that escapes the root is rejected up front.
    """
    root = os.environ.get("EVALUATOR_DATA_ROOT")
    if not root:
        return
    root_resolved = Path(root).resolve()
    data = getattr(config, "data", None)
    for field in _DATASET_PATH_FIELDS:
        value = getattr(data, field, None)
        if not value:
            continue
        target = Path(value).resolve()
        if target != root_resolved and root_resolved not in target.parents:
            raise ConfigurationError(
                f"data.{field} ({value}) is outside the allowed data root {root}"
            )


def prepare_run_config(
    payload_config: Dict[str, Any], *, auto_devices: bool
) -> EvaluationConfig:
    config = load_config(payload_config, auto_devices=auto_devices)
    _enforce_data_root(config)
    check_backend_dependencies(config.vector_db)
    template = config.graph_template
    validate_dataset_runtime_config(
        config,
        retrieval_required=(template != "asr_only"),
    )
    return config


#: The graph-block keys a builder canvas spec carries (mode/nodes/edges/branches). Anything
#: else on the spec (e.g. a global ``llm``) is a top-level config block, not a graph key.
_GRAPH_SPEC_KEYS = ("mode", "nodes", "edges", "branches")

#: Builder nodes that are gated by a config-block ``enabled`` flag (not by graph presence): the
#: LLM-feature nodes whose handler no-ops when their feature is off. Drawing one in the canvas is
#: the user's intent to run it, so node presence flips the flag on — the LLM endpoint still rides
#: the global ``llm`` block. (The metric/transform/structural nodes need no such flag.)
_FEATURE_BY_NODE_KIND = {
    "answer_judge": "judge",
    "answer_gen": "answer_generation",
}


def _enable_feature_blocks(config_dict: Dict[str, Any], nodes: list) -> None:
    """Set ``<feature>.enabled = True`` for each LLM-feature node present in the canvas, so a
    judge / answer-gen node the user drew actually runs (it would otherwise silently no-op). A
    block the spec already carries is respected — only the missing ``enabled`` is filled."""
    from evaluator.pipeline.graph.operators import node_kind

    present = {node_kind(n.get("type"), n.get("params") or {}) for n in nodes}
    for kind, feature in _FEATURE_BY_NODE_KIND.items():
        if kind in present:
            config_dict.setdefault(feature, {}).setdefault("enabled", True)


def graph_spec_to_config_dict(
    spec: Dict[str, Any], *, experiment_name: str = "builder_run"
) -> Dict[str, Any]:
    """Translate a builder canvas spec into a legacy ``EvaluationConfig`` dict.

    The canvas exports ``{mode, nodes:[{id,type,params}], edges:[{from,to}], branches?, llm?}``
    — a node-centric ``graph`` block plus an optional global ``llm``. We wrap it and run it
    through :func:`graph_config.to_legacy_dict` (the same translator the YAML loader uses), so
    a graph built in the UI takes the identical path as a hand-written config. The dataset
    rides a ``dataset_source`` node whose ``dataset`` param names a registered dataset
    (``to_legacy_dict`` synthesizes the ``data.datasets`` entry); a graph with no resolvable
    dataset passes translation but is rejected downstream by :func:`prepare_run_config`.
    """
    from evaluator.config.graph_config import to_legacy_dict

    # to_legacy_dict mutates the node dicts it's handed (it strips folded params in place);
    # deep-copy so the caller's spec is never altered (safe to call twice on the same object).
    graph = deepcopy({k: spec[k] for k in _GRAPH_SPEC_KEYS if k in spec})
    config_dict: Dict[str, Any] = {"graph": graph, "experiment_name": experiment_name}
    if spec.get("llm"):
        config_dict["llm"] = deepcopy(spec["llm"])
    # A drawn LLM-feature node (judge / answer-gen) flips its config flag on so it actually runs.
    _enable_feature_blocks(config_dict, graph.get("nodes") or [])
    return to_legacy_dict(config_dict)


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively overlay ``overlay`` onto ``base`` (overlay wins; dicts merged)."""
    out = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce_param(value: str) -> Any:
    """Best-effort type coercion of a form string to bool/int/float/str."""
    low = value.lower()
    if low in ("on", "true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _collect_model_params(form: Dict[str, str], prefix: str) -> Dict[str, Any]:
    """Gather ``param__<prefix>__*`` fields (+ free-text extra args) into a params dict.

    Registry fields land as ``name -> coerced value``; the ``extra`` textarea takes
    ``key=value`` lines for arguments the registry doesn't expose. Blank fields are
    skipped so the model's own defaults apply.
    """
    field_prefix = f"param__{prefix}__"
    extra_field = f"{field_prefix}extra"
    params: Dict[str, Any] = {}
    for key, raw in form.items():
        if not key.startswith(field_prefix) or key == extra_field:
            continue
        value = (raw or "").strip()
        if value:
            params[key[len(field_prefix):]] = _coerce_param(value)
    for line in (form.get(extra_field) or "").splitlines():
        line = line.strip()
        if line and "=" in line:
            k, _, v = line.partition("=")
            params[k.strip()] = _coerce_param(v.strip())
    return params


def _form_to_config(form: Dict[str, str]) -> Dict[str, Any]:
    """Build a FULL EvaluationConfig dict from the form.

    Config-first: the selected preset's complete config is the base; the form's exposed
    fields are overlaid on top. Everything the preset sets but the form doesn't expose
    (strict_validation, num_workers, asr_params, judge, hybrid retrieval, TTS seed /
    output_dir, …) is preserved instead of dropped.
    """
    def s(key: str) -> str:
        return (form.get(key) or "").strip()

    def num(key: str, default):
        v = s(key)
        try:
            return int(v) if v else default
        except ValueError:
            return default

    # The form's template selector → a graph template reference (no pipeline_mode field anymore).
    model: Dict[str, Any] = {}
    for fam in MODEL_FAMILY_FIELDS:
        if not s(fam.type_field):
            continue
        model[fam.type_field] = s(fam.type_field)
        if s(fam.size_field):
            model[fam.size_field] = s(fam.size_field)
        params = _collect_model_params(form, fam.prefix)
        if params:
            model[fam.params_field] = params

    data: Dict[str, Any] = {}
    for k in ("dataset_name", "questions_path", "corpus_path"):
        if s(k):
            data[k] = s(k)
    data["batch_size"] = num("batch_size", 8)
    # Quick-test caps the run to a cheap sample, overriding the trace_limit field.
    data["trace_limit"] = 20 if s("quick_test") == "on" else num("trace_limit", 0)

    audio_synthesis: Dict[str, Any] = {
        "enabled": s("audio_synthesis_enabled") == "on",
        "provider": s("tts_provider") or "mms",
    }
    if s("tts_language"):
        audio_synthesis["language"] = s("tts_language")
    if s("tts_voice"):
        audio_synthesis["voice"] = s("tts_voice")

    overlay: Dict[str, Any] = {
        "experiment_name": s("experiment_name") or "webui_experiment",
        "output_dir": s("output_dir") or "evaluation_results/webui",
        "model": model,
        "graph_override": {"template": s("pipeline_mode") or "asr_text_retrieval"},
        "data": data,
        "vector_db": {
            "type": s("vector_db_type") or "inmemory",
            "k": num("k", 10),
            "retrieval_mode": s("retrieval_mode") or "dense",
        },
        "audio_synthesis": audio_synthesis,
    }

    # Base = the selected preset's full config (config-first); form overlays on top.
    base: Dict[str, Any] = {}
    name = s("name")
    if name:
        from evaluator.config.model_presets import get_preset
        try:
            base = get_preset(name, auto_devices=False)
        except Exception:
            base = {}
    merged = _deep_merge(base, overlay)

    # Merge coherence (m4t/whisper bug): when the form selects a DIFFERENT model type
    # than the preset for a family, the preset's name/size/adapter/params belong to the
    # OTHER model and must not survive the deep merge — they produced runs like
    # `whisper` + `facebook/seamless-m4t-v2-large` + m4t-only `tgt_lang`, crashing
    # WhisperModel.__init__. The factory's _check_extra_params is the backstop; this
    # keeps the merged config coherent at the source (shared rule: model_fields.py).
    base_model = base.get("model") or {}
    for fam in MODEL_FAMILY_FIELDS:
        form_type = s(fam.type_field)
        if not form_type:
            continue
        if base_model.get(fam.type_field) and base_model[fam.type_field] != form_type:
            strip_stale_family_fields(
                merged["model"], fam, drop_size=not s(fam.size_field)
            )
            merged["model"][fam.params_field] = _collect_model_params(form, fam.prefix)
    return merged


# Keyword (lower-cased, found anywhere in the error message) → the form field name it points
# at, so a config error highlights the offending input instead of just printing a raw string.
# Ordered: first match wins (more-specific phrases before broad ones). Every target MUST be a
# real ``name="…"`` in _config_form.html — a drift-guard test pins this, so a hint can't point at
# a field that doesn't exist (e.g. reranker is configured via the model section, not a top-level
# field, so there's nothing to highlight — no mapping for it).
_ERROR_FIELD_HINTS = (
    ("retrieval_mode", "retrieval_mode"),
    ("retrieval mode", "retrieval_mode"),
    ("fusion", "retrieval_mode"),
    ("vector_db", "vector_db_type"),
    ("vector db", "vector_db_type"),
    ("batch", "batch_size"),
    ("trace", "trace_limit"),
    ("dataset", "dataset_name"),
)


def _field_for_error(message: str):
    """The form field a config-error message points at (keyword match), or None."""
    low = message.lower()
    return next((field for kw, field in _ERROR_FIELD_HINTS if kw in low), None)


def _friendly_error_html(exc: Exception) -> str:
    """A friendlier validation-error fragment: the message + (when the offending field is
    identifiable) a hint naming it with its glossary help, plus ``data-field`` so the form can
    highlight the input. Falls back to just the message when no field is recognised."""
    import html as _html

    from evaluator.webapi.field_help import FIELD_HELP

    message = _html.escape(str(exc))
    field = _field_for_error(str(exc))
    hint, attr = "", ""
    if field:
        attr = f' data-field="{_html.escape(field)}"'
        help_text = _html.escape(FIELD_HELP.get(field, ""))
        hint = (
            f'<p class="muted" style="font-size:12px">Check the <strong>{field}</strong> '
            f'field. {help_text}</p>'
        )
    return f'<div class="error-box"{attr}><p class="error">⚠ {message}</p>{hint}</div>'


def _prepared_config_or_error(form):
    """Build a validated run config from form data.

    Returns ``(config, None)`` on success or ``(None, HTMLResponse)`` with a friendly error
    fragment when the form is invalid — shared by the validate/graph/run routes.
    """
    try:
        return prepare_run_config(_form_to_config(dict(form)), auto_devices=True), None
    except (ConfigurationError, ImportError, ValueError) as exc:
        return None, HTMLResponse(_friendly_error_html(exc))
