"""Request-independent helpers for the htmx UI.

Form-field mapping and registry-driven model-section building, kept out of
``ui.py`` so the router file stays focused on route wiring.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from fastapi.responses import HTMLResponse

from evaluator import ConfigurationError
from evaluator.webapi.config_helpers import create_config_options, prepare_run_config
from evaluator.webapi.utils import with_provider

# Pipeline field -> (family, type form-field, size form-field).
_MODEL_FIELDS = {
    "model.asr_model_type": ("asr", "asr_model_type", "asr_size"),
    "model.text_emb_model_type": ("text_embedding", "text_emb_model_type", "text_emb_size"),
    "model.audio_emb_model_type": ("audio_embedding", "audio_emb_model_type", "audio_emb_size"),
}


def _required_fields(mode: str) -> List[str]:
    from evaluator.pipeline.stage_graph import resolve_pipeline_mode_spec
    try:
        return list(resolve_pipeline_mode_spec(mode).required_model_fields)
    except ValueError:
        return []


def _model_section(provider_factory, mode: str, current: Dict[str, str]) -> List[Dict[str, Any]]:
    """Build the registry-driven model-select descriptors for a pipeline mode."""
    models = with_provider(provider_factory, lambda p: p.list_available_models())
    import evaluator.models  # noqa: F401
    from evaluator.models.registry import FAMILY_REGISTRIES

    sections: List[Dict[str, Any]] = []
    for field in _required_fields(mode):
        spec = _MODEL_FIELDS.get(field)
        if not spec:
            continue
        family, type_field, size_field = spec
        reg = FAMILY_REGISTRIES.get(family)
        selected_type = current.get(type_field, "")
        sizes = list(reg.get_sizes(selected_type).keys()) if (reg and selected_type) else []
        prefix = type_field[: -len("_model_type")]  # asr / text_emb / audio_emb
        sections.append({
            "family": family,
            "type_field": type_field,
            "size_field": size_field,
            "options": [{"type": e["type"], "name": e["name"]} for e in models.get(family, [])],
            "selected_type": selected_type,
            "sizes": sizes,
            "selected_size": current.get(size_field, ""),
            "params": _model_param_fields(reg, selected_type, prefix, current),
            "extra_field": f"param__{prefix}__extra",
        })
    return sections


def _model_param_fields(reg, model_type: str, prefix: str, current: Dict[str, str]) -> List[Dict[str, Any]]:
    """Registry-declared optional args (Params dataclass) as renderable form fields.

    ``size`` is excluded (it has its own select). Each field carries the registry default
    so the UI shows it; the user can override or leave blank to keep the default.
    """
    if not (reg and model_type):
        return []
    import dataclasses
    fields: List[Dict[str, Any]] = []
    for name, info in reg.get_params_schema(model_type).items():
        if name == "size":
            continue
        default = info.get("default")
        if default is dataclasses.MISSING:
            default = ""
        field = f"param__{prefix}__{name}"
        fields.append({
            "name": name,
            "field": field,
            "value": current.get(field, ""),
            "default": "" if default is None else default,
            "choices": info.get("choices") or [],
            "is_bool": isinstance(default, bool),
        })
    return fields


def _preset_form_context(
    provider_factory, name: str = "", form: Dict[str, str] | None = None
) -> Dict[str, Any]:
    """Build the config-form template context, prefilled from a preset.

    Shared by ``/ui/config`` (default preset, Run-ready on load) and ``/ui/preset``
    (htmx swap when the user picks a preset). Returns the kwargs both templates expect:
    ``options``, ``mode``, ``preset`` (flat form values), ``model_sections``.

    When ``form`` (the current form values) is given, the preset fills gaps but does
    **not** clobber fields the user already entered that the preset doesn't define
    (e.g. dataset/paths — presets carry no dataset). Preset values win where present.
    """
    from evaluator.config.model_presets import get_preset

    cfg: Dict[str, Any] = {}
    if name:
        try:
            cfg = get_preset(name, auto_devices=False)
        except Exception:
            cfg = {}
    m = cfg.get("model", {})
    data = cfg.get("data", {})
    vdb = cfg.get("vector_db", {})
    asx = cfg.get("audio_synthesis", {})
    mode = m.get("pipeline_mode", "asr_text_retrieval")

    def fv(key: str) -> str:
        """Current user-entered form value for ``key`` (empty if none)."""
        return ((form or {}).get(key) or "").strip()

    def pick(preset_value, key, default):
        """Preset value if it set one, else the user's form value, else default."""
        if preset_value not in (None, ""):
            return preset_value
        return fv(key) or default

    # Models: preset wins; fall back to the user's current selection per field.
    current = {
        "asr_model_type": m.get("asr_model_type") or fv("asr_model_type"),
        "asr_size": m.get("asr_size") or fv("asr_size"),
        "text_emb_model_type": m.get("text_emb_model_type") or fv("text_emb_model_type"),
        "text_emb_size": m.get("text_emb_size") or fv("text_emb_size"),
        "audio_emb_model_type": m.get("audio_emb_model_type") or fv("audio_emb_model_type"),
        "audio_emb_size": m.get("audio_emb_size") or fv("audio_emb_size"),
    }
    return {
        "options": create_config_options(provider_factory),
        "mode": mode,
        "preset": {
            "experiment_name": pick(cfg.get("experiment_name"), "experiment_name", "webui_experiment"),
            "output_dir": pick(cfg.get("output_dir"), "output_dir", "evaluation_results/webui"),
            # Presets never carry a dataset — always keep what the user picked.
            "dataset_name": pick(data.get("dataset_name"), "dataset_name", ""),
            "questions_path": pick(data.get("questions_path"), "questions_path", ""),
            "corpus_path": pick(data.get("corpus_path"), "corpus_path", ""),
            "batch_size": pick(data.get("batch_size"), "batch_size", 8),
            "trace_limit": pick(data.get("trace_limit"), "trace_limit", 0),
            "vector_db_type": pick(vdb.get("type"), "vector_db_type", "inmemory"),
            "k": pick(vdb.get("k"), "k", 10),
            "retrieval_mode": pick(vdb.get("retrieval_mode"), "retrieval_mode", "dense"),
            # TTS audio synthesis — prefill so the preset's setting isn't dropped.
            "audio_synthesis_enabled": "on" if asx.get("enabled") else fv("audio_synthesis_enabled"),
            "tts_provider": pick(asx.get("provider"), "tts_provider", "mms"),
            "tts_language": pick(asx.get("language"), "tts_language", "en"),
            "tts_voice": pick(asx.get("voice"), "tts_voice", ""),
        },
        "model_sections": _model_section(provider_factory, mode, current),
    }


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

    model: Dict[str, Any] = {"pipeline_mode": s("pipeline_mode") or "asr_text_retrieval"}
    for _field, (_family, type_field, size_field) in _MODEL_FIELDS.items():
        if not s(type_field):
            continue
        model[type_field] = s(type_field)
        if s(size_field):
            model[size_field] = s(size_field)
        prefix = type_field[: -len("_model_type")]  # asr / text_emb / audio_emb
        params = _collect_model_params(form, prefix)
        if params:
            model[f"{prefix}_params"] = params

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
    return _deep_merge(base, overlay)


def _prepared_config_or_error(form):
    """Build a validated run config from form data.

    Returns ``(config, None)`` on success or ``(None, HTMLResponse)`` with an error
    fragment when the form is invalid — shared by the validate/graph/run routes.
    """
    try:
        return prepare_run_config(_form_to_config(dict(form)), auto_devices=True), None
    except (ConfigurationError, ImportError, ValueError) as exc:
        return None, HTMLResponse(f'<p class="error">{exc}</p>')
