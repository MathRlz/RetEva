# Evaluator: Architecture Analysis & Improvement Plan

## What Works Well

### Registry-driven model discovery
`ModelRegistry` with decorator-based registration (`@register_asr_model`, etc.) is clean. Models are discoverable, validated at config time, and new families plug in without touching evaluation loops.

### Config-first evaluation
`EvaluationConfig` dataclass tree with YAML/preset/programmatic construction, `with_auto_devices()`, and `validate()` gives reproducibility. The preset system (`from_preset`) enables quick-start workflows.

### DAG-lite stage graph
`build_stage_graph()` makes pipeline mode planning explicit and deterministic. WebUI can preview the DAG before running. `PipelineModeSpec` declares required model fields per mode.

### Cache manifest v2
SQLite metadata index + disk artifacts with fingerprinted keys (dataset + model + retrieval strategy) prevents unsafe cache reuse. Smart cache delta tracking in evaluation results.

### Leaderboard + matrix evaluation
`ExperimentStore` (SQLite) with `run_evaluation_matrix` and comparison bundles (baseline deltas, ranking) makes experiment comparison first-class.

### Service lifecycle layer
`ModelServiceProvider` with `startup_mode` (lazy/eager) and `offload_policy` (on_finish/never) + local LLM server management. Clean shutdown semantics.

---

## What Needs Improvement

### 1. CRITICAL: Flat model config — no family-specific parameters

**Problem:** `ModelConfig` uses flat fields (`asr_model_type`, `asr_model_name`, `asr_device`) shared across all families. User picks a family but can't set family-specific params. All model families share the same flat config shape.

Currently each model takes a raw HF `model_name` string (e.g. `"openai/whisper-small"`). User must know HuggingFace identifiers. There's no size picker, no family-specific knobs.

**Impact:** Core UX requirement — "user chooses family, then gets family-specific settings" — is impossible.

**Target UX — super easy:**
```python
# User picks family + size. That's it for basic use.
results = evaluate_from_preset(
    "whisper_labse",
    asr_size="medium",           # ← just size
    embedding_model="labse",
    data_path="questions.json",
)

# Power users can still override model_name directly
results = evaluate_from_preset(
    "whisper_labse",
    asr_model_name="openai/whisper-large-v3",  # ← explicit override
)
```

**Fix — registry declares sizes + params per family:**

Each model registration declares a `Params` dataclass with `size` as primary knob. Registry resolves `size → model_name` automatically. Extra family-specific fields only where they matter.

```python
# --- ASR families ---

@register_asr_model("whisper")
class WhisperModel(HuggingFaceASRModel):
    @dataclass
    class Params:
        size: str = "small"  # tiny|base|small|medium|large-v2|large-v3
        SIZES = {
            "tiny": "openai/whisper-tiny",
            "base": "openai/whisper-base",
            "small": "openai/whisper-small",
            "medium": "openai/whisper-medium",
            "large-v2": "openai/whisper-large-v2",
            "large-v3": "openai/whisper-large-v3",
        }

@register_asr_model("faster_whisper")
class FasterWhisperModel(ASRModel):
    @dataclass
    class Params:
        size: str = "large-v3"  # tiny|base|small|medium|large-v1|large-v2|large-v3|distil-large-v3
        compute_type: str = "float16"  # int8|float16|float32
        SIZES = {
            "tiny": "tiny", "base": "base", "small": "small",
            "medium": "medium", "large-v1": "large-v1",
            "large-v2": "large-v2", "large-v3": "large-v3",
            "distil-large-v3": "distil-large-v3",
        }

@register_asr_model("wav2vec2")
class Wav2Vec2Model(HuggingFaceASRModel):
    @dataclass
    class Params:
        size: str = "large-polish"  # base|large|large-polish
        SIZES = {
            "base": "facebook/wav2vec2-base-960h",
            "large": "facebook/wav2vec2-large-960h",
            "large-polish": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
        }


# --- Text embedding families ---

@register_text_embedding_model("labse")
class LabseModel(TextEmbeddingModel):
    @dataclass
    class Params:
        size: str = "default"  # single size
        SIZES = {"default": "sentence-transformers/LaBSE"}

@register_text_embedding_model("jina_v4")
class JinaV4Model(TextEmbeddingModel):
    @dataclass
    class Params:
        size: str = "default"
        SIZES = {"default": "jinaai/jina-embeddings-v4"}

@register_text_embedding_model("bge_m3")
class BgeM3Model(TextEmbeddingModel):
    @dataclass
    class Params:
        size: str = "default"
        SIZES = {"default": "BAAI/bge-m3"}

@register_text_embedding_model("nemotron")
class NemotronModel(TextEmbeddingModel):
    @dataclass
    class Params:
        size: str = "8b"
        SIZES = {"8b": "nvidia/llama-embed-nemotron-8b"}

@register_text_embedding_model("clip")
class ClipModel(TextEmbeddingModel):
    @dataclass
    class Params:
        size: str = "base"  # base|large
        SIZES = {
            "base": "openai/clip-vit-base-patch32",
            "large": "openai/clip-vit-large-patch14",
        }


# --- Audio embedding families ---

@register_audio_embedding_model("attention_pool")
class AttentionPoolAudioModel(AudioEmbeddingModel):
    @dataclass
    class Params:
        size: str = "default"
        emb_dim: int = 2048
        dropout: float = 0.1
        SIZES = {"default": "facebook/wav2vec2-base-960h"}

@register_audio_embedding_model("clap_style")
class MultimodalClapStyleModel(AudioEmbeddingModel):
    @dataclass
    class Params:
        model_path: str = ""  # required, path to pretrained weights
        # no size — user provides model_path directly
```

**ModelConfig changes:**
```python
@dataclass
class ModelConfig:
    pipeline_mode: str = "asr_text_retrieval"

    asr_model_type: Optional[str] = "whisper"
    asr_size: Optional[str] = None          # NEW — resolves via Params.SIZES
    asr_model_name: Optional[str] = None    # explicit override (skips size resolution)
    asr_params: Dict[str, Any] = field(default_factory=dict)  # NEW — extra family params
    asr_device: str = "cuda:0"

    text_emb_model_type: Optional[str] = "labse"
    text_emb_size: Optional[str] = None
    text_emb_model_name: Optional[str] = None
    text_emb_params: Dict[str, Any] = field(default_factory=dict)
    text_emb_device: str = "cuda:1"

    audio_emb_model_type: Optional[str] = None
    audio_emb_size: Optional[str] = None
    audio_emb_model_name: Optional[str] = None
    audio_emb_params: Dict[str, Any] = field(default_factory=dict)
    audio_emb_device: str = "cuda:0"
```

**Resolution chain:** `asr_model_name` (explicit) > `asr_size` via `Params.SIZES` > registry `default_name`.

**WebUI flow:**
1. User picks family (dropdown: whisper / faster_whisper / wav2vec2)
2. Backend returns `Params` schema → frontend shows size dropdown + any extra fields
3. User picks size. Done. No HF model names visible unless they open advanced panel.

**API endpoint for frontend:**
```
GET /api/models/{family}/params → { "size": {"choices": [...], "default": "small"}, ... }
```

### 2. CRITICAL: Dataset config doesn't drive pipeline/generation

**Problem:** `DataConfig` has flat fields mixing dataset identity (name, paths) with loader mechanics (source, column_mapping) and processing params (batch_size, trace_limit). No connection between dataset_type and:
- What pipeline modes are valid
- What data generation is needed (TTS, augmentation)
- What metrics to compute

`DatasetCapabilityProfile` exists but has only 2 hardcoded entries (`admed_voice`, `pubmed_qa`). The runtime spec system (`DatasetRuntimeSpec`) is a parallel concept that doesn't compose with profiles.

**Impact:** User can't say "I have a text QA dataset, generate audio, run through ASR pipeline" as a coherent workflow. Dataset type should drive the entire experiment shape.

**Fix:** Unify profiles + runtime specs into one registry. Dataset type should declare:
- Required input fields (audio_dir? questions_path? corpus_path?)
- Generation capabilities (can TTS? needs augmentation?)
- Compatible pipeline modes
- Metric set to compute
- WebUI form shape

### 3. HIGH: WebUI renders all config fields generically

**Problem:** `ConfigPage.tsx` walks the full nested config and renders every field with type-based widgets. No awareness of:
- Pipeline mode → which model sections are relevant
- Model family → family-specific parameters
- Dataset type → which data fields to show
- Conditional dependencies (reranker fields only if reranker_enabled)

Result: user sees 100+ fields regardless of context.

**Fix — stepped wizard, not a field dump:**

```
Step 1: Pipeline mode     → dropdown (asr_text_retrieval / audio_emb / ...)
Step 2: Dataset type      → dropdown, shows only relevant data fields
Step 3: Model families    → per-slot dropdown (ASR: whisper/wav2vec2, Embed: labse/jina/...)
Step 4: Model sizes       → per-family size dropdown + extra params if any
Step 5: Advanced (opt)    → retrieval settings, cache, tracking, augmentation
```

Each step fetches schema from backend. Frontend only renders what's relevant. No 100-field wall.

Backend endpoints needed:
- `GET /api/models/{family}/params` → size choices + extra fields
- `GET /api/pipeline/{mode}/required_models` → which model slots to show
- `GET /api/dataset/{type}/fields` → which data fields to render

### 4. HIGH: VectorDBConfig is a god-object with stub features

**Problem:** `VectorDBConfig` has 40+ fields including multi-vector retrieval, query expansion, pseudo-relevance feedback, adaptive fusion — most appear to be config stubs without backing implementation. Also mixes backend-specific settings (chromadb_path, qdrant_url, qdrant_api_key) with retrieval strategy settings (mmr_lambda, bm25_k1).

**Impact:** Confusing for users. WebUI exposes unimplemented features. Config validation doesn't warn about unsupported combinations.

**Fix:**
- Split into `VectorStoreConfig` (backend) + `RetrievalStrategyConfig` (search behavior) + per-backend config dataclasses
- Remove stub fields or gate them behind feature flags
- Validate that enabled features have backing implementations

### 5. HIGH: Pipeline factory has massive duplication

**Problem:** `pipeline/factory.py` `create_pipeline_from_config()` repeats model creation logic for each pipeline mode. The `service_provider is not None` check duplicates every model creation call. Audio embedding model creation is copy-pasted between `audio_emb_retrieval` and `audio_text_retrieval`.

**Fix:** Extract model creation into a helper that takes (model_category, config_fields, service_provider) and returns the model. Pipeline mode logic should only declare which models it needs, not how to create each one.

```python
def _create_model(category, config, service_provider, device_pool):
    """Unified model creation — service-backed or direct factory."""
    ...
```

### 6. MEDIUM: Two factory.py files with duplicated logic

**Problem:** `models/factory.py` and `pipeline/factory.py` both create models. `pipeline/factory.py` wraps `models/factory.py` functions with identical signatures. `models/factory.py` has hardcoded imports (`from .asr import whisper, wav2vec2, faster_whisper`) to force registration — defeating the purpose of a registry.

**Fix:** Registry should handle auto-discovery (entry points or package scan). Single factory entry point. Pipeline factory should only compose pipelines, not re-wrap model creation.

### 7. MEDIUM: Config serialization is manual and fragile

**Problem:** `to_runtime_dict()` and `to_experiment_dict()` manually enumerate every field. Adding a config field requires updating 3+ places: the dataclass, runtime_dict, experiment_dict, and from_dict parsing.

**Fix:** Use `dataclasses.asdict()` with a custom filter, or generate serialization from field metadata. Or switch sub-configs to Pydantic for automatic serialization.

### 8. MEDIUM: evaluate_phased has 16+ positional params

**Problem:** `evaluate_phased()` takes 16 individual parameters instead of structured objects. Callers in `evaluation_service.py` duplicate the same keyword argument block twice (parallel vs non-parallel path).

**Fix:** Accept `PipelineBundle` + `EvaluationConfig` (or a focused subset) instead of individual params. Deduplicate the parallel/sequential call in evaluation_service.

### 9. MEDIUM: api.py hardcodes model name mappings

**Problem:** `quick_evaluate()` has hardcoded dicts mapping `"whisper"` → `"openai/whisper-base"`, `"labse"` → `"sentence-transformers/LaBSE"`. Duplicates what registry knows. After #1, this becomes dead code.

**Fix:** `quick_evaluate` takes `model="whisper"`, `size="medium"`. Registry resolves name via `Params.SIZES`. Remove all hardcoded dicts.

```python
# Target API
results = quick_evaluate("test_audio/", model="whisper", size="medium", embedding="labse")
```

### 10. MEDIUM: Legacy dataset classes coexist with new loaders

**Problem:** `datasets/core.py` has `AdmedQueryDataset` with hardcoded CSV parsing and Polish-specific path logic, `PubMedQADataset` with its own JSON loading. `datasets/loaders/` has a new generic loader system. Both are re-exported from `datasets/__init__.py`. Runtime loading path (`load_runtime_dataset`) dispatches to either system based on config.

**Impact:** Two loading codepaths to maintain. Adding a new dataset requires knowing which system to use.

**Fix:** Migrate legacy datasets to the loader system. `AdmedQueryDataset` and `PubMedQADataset` become loader implementations registered in the runtime spec registry.

### 11. LOW: No progress streaming for WebUI jobs

**Problem:** `JobManager` uses `ThreadPoolExecutor` with fire-and-forget semantics. WebUI polls job status. No way to stream per-sample progress, current phase, or intermediate metrics.

**Fix:** Add progress callback to `evaluate_phased`. Expose via SSE or WebSocket endpoint. Store progress updates in `JobRecord`.

### 12. LOW: No schema introspection API for frontend

**Problem:** `/api/config/options` returns flat lists of choices (pipeline_modes, model types, etc.) but doesn't describe field dependencies or conditional visibility. Frontend hardcodes path → dropdown mappings in `selectOptionsForPath()`.

**Fix:** Add `/api/config/schema` endpoint that returns JSON Schema or equivalent per pipeline mode, including conditional field requirements and model-family-specific parameter schemas.

### 13. LOW: EvaluationConfig.from_preset override parsing is brittle

**Problem:** `from_preset()` parses underscore-separated keys (`model_asr_device` → `model.asr_device`) with special-case handling for `vector_db_`, `device_pool_`, `service_runtime_`, `audio_synthesis_`. Easy to break, hard to extend.

**Fix:** Accept dotted paths (`"model.asr_device"`) or use a dedicated override format. Or accept nested dicts directly.

---

## Recommended Priority Order

1. **Model family parameter schemas** (#1) — blocks the core UX goal
2. **Dataset-type-driven experiment shape** (#2) — blocks the dataset generation UX goal
3. **WebUI conditional rendering** (#3) — depends on #1 and #2
4. **VectorDBConfig split** (#4) — cleanup that unblocks config clarity
5. **Pipeline factory dedup** (#5, #6) — reduces maintenance burden
6. **Config serialization** (#7) — reduces change friction for all above
7. **Everything else** (#8-#13) — quality improvements
