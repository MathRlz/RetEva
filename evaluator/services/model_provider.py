"""Service provider for model lifecycle orchestration."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Optional, Tuple, Any, Callable, MutableMapping, List
import json
import time

from ..models import (
    create_asr_model as factory_create_asr_model,
    create_text_embedding_model as factory_create_text_embedding_model,
    create_audio_embedding_model as factory_create_audio_embedding_model,
    create_reranker as factory_create_reranker,
)
from ..models.llm import create_server as factory_create_llm_server
from .model_services import (
    FactoryModelService,
    LLMServerService,
)
from ..logging_config import get_logger


def _kw_default(obj: Any) -> Any:
    """JSON fallback for non-native kwarg values. Sets/frozensets become a sorted list so the key
    is deterministic across processes (``str(set)`` order is hash-seed dependent); everything else
    falls back to ``str``."""
    if isinstance(obj, (set, frozenset)):
        return sorted(map(str, obj))
    return str(obj)


def _kw_key(kwargs: Dict[str, Any]) -> Tuple[Any, ...]:
    """A stable, hashable key fragment for model-specific kwargs — empty (no key change) when no
    kwargs, so existing callers cache exactly as before; a non-empty set caches distinctly (e.g.
    ``pooling=mean_abtt`` vs ``attention`` are different models)."""
    return (json.dumps(kwargs, sort_keys=True, default=_kw_default),) if kwargs else ()


ASRKey = Tuple[str, Optional[str], Optional[str], str]
TextKey = Tuple[str, Optional[str], str]
AudioKey = Tuple[str, Optional[str], Optional[str], int, float, str]
RerankerKey = Tuple[str, Optional[str], Optional[str], int, int]
TTSKey = Tuple[str, str, str, int]
LLMServerKey = Tuple[str, str, int, str, int]
logger = get_logger(__name__)


class ModelServiceProvider:
    """Creates, reuses, and tears down model services for one runtime."""

    def __init__(
        self,
        *,
        soft_offload_max_warm: int = 0,
        soft_offload_ttl_s: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._asr_services: Dict[ASRKey, FactoryModelService] = {}
        self._text_services: Dict[TextKey, FactoryModelService] = {}
        self._audio_services: Dict[AudioKey, FactoryModelService] = {}
        self._reranker_services: Dict[RerankerKey, FactoryModelService[Any]] = {}
        self._tts_services: Dict[TTSKey, FactoryModelService] = {}
        self._llm_server_services: Dict[LLMServerKey, LLMServerService] = {}
        # Soft-CPU offload warm pool (Roadmap 2c): models released under the
        # ``on_finish_soft_cpu`` policy are parked on host RAM (warm) instead of freed, so a
        # later stage/run re-acquires them with a CPU→device move rather than a full reload.
        # Bounded LRU (capacity) + TTL keep host RAM in check; 0 = soft offload disabled.
        self._soft_offload_max_warm = int(soft_offload_max_warm or 0)
        self._soft_offload_ttl_s = soft_offload_ttl_s
        self._clock = clock
        # (id(bucket), key) -> {bucket, key, service, device, ts}; insertion order = LRU.
        self._warm: "OrderedDict[Tuple[int, Any], Dict[str, Any]]" = OrderedDict()
        self._offload_stats: Dict[str, int] = {
            "soft_offloads": 0, "evictions": 0, "warm_reuses": 0, "full_offloads": 0,
        }

    def configure_soft_offload(
        self, *, max_warm: Optional[int] = None, ttl_s: Optional[float] = None
    ) -> None:
        """Set the soft-CPU warm-pool capacity / TTL (called from the run when the
        ``on_finish_soft_cpu`` policy is active)."""
        if max_warm is not None:
            self._soft_offload_max_warm = int(max_warm or 0)
        if ttl_s is not None:
            self._soft_offload_ttl_s = ttl_s

    def offload_stats(self) -> Dict[str, int]:
        """Soft-offload event counters (warm parks / LRU+TTL evictions / warm reuses /
        full frees) for provenance + logging."""
        return dict(self._offload_stats)

    def _get_or_create(
        self,
        bucket: MutableMapping[Any, FactoryModelService[Any]],
        key: Any,
        factory,
        label: str,
    ):
        """Cached get-or-create over a service bucket: build the service on first
        request for ``key`` (wrapping ``factory``), reuse it after. A warm (soft-offloaded)
        service is reactivated — moved back to its device — on reuse."""
        service = bucket.get(key)
        if service is None:
            service = FactoryModelService(factory, label=label)
            bucket[key] = service
        else:
            self._reactivate_if_warm(bucket, key, service)
        return service.get()

    def _reactivate_if_warm(self, bucket, key, service) -> None:
        """If ``service`` is parked warm on CPU, move it back to its device and unpark it."""
        entry = self._warm.pop((id(bucket), key), None)
        if entry is None:
            return
        try:
            service.move_to_device(entry["device"])
        except Exception as exc:  # noqa: BLE001 - reuse must not break on a move failure
            logger.warning("warm reactivate move failed (%s); leaving on cpu", exc)
        self._offload_stats["warm_reuses"] += 1

    #: Ordered key fields per family — declared once so get/move/release can't drift in
    #: field order (B3/F16). ``device`` is uniform; move passes current/target through it.
    _KEY_FIELDS: Dict[str, Tuple[str, ...]] = {
        "asr": ("model_type", "model_name", "adapter_path", "device"),
        "text": ("model_type", "model_name", "device"),
        "audio": ("model_type", "model_name", "model_path", "emb_dim", "dropout", "device"),
        "reranker": ("model_type", "model_name", "device", "batch_size", "max_length"),
    }

    @classmethod
    def _key_for(cls, family: str, **fields: Any) -> Tuple[Any, ...]:
        """Build a service-bucket key for ``family`` from its declared field order (B3/F16)."""
        try:
            names = cls._KEY_FIELDS[family]
        except KeyError:
            raise ValueError(f"unknown model family {family!r}") from None
        return tuple(fields[name] for name in names)

    def get_asr_model(
        self,
        model_type: str,
        model_name: Optional[str],
        adapter_path: Optional[str],
        device: str,
        **kwargs: Any,
    ):
        # Model-specific params (size, quantization, the model's own Params fields) flow through
        # **kwargs straight to the factory, and are folded into the cache key (registry-driven —
        # the provider needn't know what they mean).
        key = self._key_for(
            "asr", model_type=model_type, model_name=model_name,
            adapter_path=adapter_path, device=device,
        ) + _kw_key(kwargs)
        return self._get_or_create(
            self._asr_services,
            key,
            lambda: factory_create_asr_model(
                model_type=model_type,
                model_name=model_name,
                adapter_path=adapter_path,
                device=device,
                **kwargs,
            ),
            f"asr:{model_type}@{device}",
        )

    def get_text_embedding_model(
        self,
        model_type: str,
        model_name: Optional[str],
        device: str,
        **kwargs: Any,
    ):
        key = self._key_for(
            "text", model_type=model_type, model_name=model_name, device=device,
        ) + _kw_key(kwargs)
        return self._get_or_create(
            self._text_services,
            key,
            lambda: factory_create_text_embedding_model(
                model_type=model_type,
                model_name=model_name,
                device=device,
                **kwargs,
            ),
            f"text_embedding:{model_type}@{device}",
        )

    def get_audio_embedding_model(
        self,
        model_type: str,
        model_name: Optional[str],
        model_path: Optional[str],
        emb_dim: int,
        dropout: float,
        device: str,
        **kwargs: Any,
    ):
        key = self._key_for(
            "audio", model_type=model_type, model_name=model_name, model_path=model_path,
            emb_dim=emb_dim, dropout=dropout, device=device,
        ) + _kw_key(kwargs)
        return self._get_or_create(
            self._audio_services,
            key,
            lambda: factory_create_audio_embedding_model(
                model_type=model_type,
                model_name=model_name,
                model_path=model_path,
                emb_dim=emb_dim,
                dropout=dropout,
                device=device,
                **kwargs,
            ),
            f"audio_embedding:{model_type}@{device}",
        )

    def get_reranker(
        self,
        model_type: str = "cross_encoder",
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        key = self._key_for(
            "reranker", model_type=model_type, model_name=model_name, device=device,
            batch_size=batch_size, max_length=max_length,
        )
        return self._get_or_create(
            self._reranker_services,
            key,
            lambda: factory_create_reranker(
                model_type=model_type,
                model_name=model_name,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
            ),
            f"reranker:{model_type}@{device}",
        )

    def release_model_instance(self, model: object, *, soft_cpu: bool = False) -> bool:
        for bucket in (
            self._asr_services,
            self._text_services,
            self._audio_services,
            self._reranker_services,
            self._tts_services,
        ):
            for key, service in list(bucket.items()):
                if service._instance is model:  # service identity check only
                    if soft_cpu and self._soft_offload_max_warm > 0:
                        self._soft_offload(bucket, key, service)
                    else:
                        service.stop()
                        del bucket[key]
                        self._warm.pop((id(bucket), key), None)
                        self._offload_stats["full_offloads"] += 1
                        logger.info("provider.release_instance key=%s", key)
                    return True
        return False

    def _soft_offload(self, bucket, key, service) -> None:
        """Park ``service`` warm on CPU (keep the loaded instance) + record it for LRU/TTL
        eviction. Falls back to a full free if the CPU move fails."""
        label = getattr(service, "label", "") or ""
        device = label.rsplit("@", 1)[-1] if "@" in label else "cpu"
        try:
            service.move_to_device("cpu")
        except Exception as exc:  # noqa: BLE001 - a failed park degrades to a full free
            logger.warning("soft offload move failed (%s); freeing instead", exc)
            service.stop()
            bucket.pop(key, None)
            self._offload_stats["full_offloads"] += 1
            return
        wk = (id(bucket), key)
        self._warm.pop(wk, None)  # refresh LRU position
        self._warm[wk] = {
            "bucket": bucket, "key": key, "service": service,
            "device": device, "ts": self._clock(),
        }
        self._offload_stats["soft_offloads"] += 1
        logger.info("provider.soft_offload key=%s parked on cpu (was %s)", key, device)
        self._evict_warm()

    def _evict_warm(self) -> None:
        """Free warm entries past the TTL, then LRU-evict down to the capacity."""
        if self._soft_offload_ttl_s is not None:
            now = self._clock()
            for wk, e in list(self._warm.items()):
                if now - e["ts"] > self._soft_offload_ttl_s:
                    self._free_warm(wk)
        while len(self._warm) > self._soft_offload_max_warm:
            self._free_warm(next(iter(self._warm)))  # oldest = LRU

    def _free_warm(self, wk) -> None:
        entry = self._warm.pop(wk, None)
        if entry is None:
            return
        entry["service"].stop()
        entry["bucket"].pop(entry["key"], None)
        self._offload_stats["evictions"] += 1
        logger.info("provider.warm_evict key=%s freed", entry["key"])

    def list_available_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """List all available models with normalized metadata schema.

        Fully registry-driven: each ``ModelRegistry`` lazily imports its family
        module on first lookup, which runs every model's ``@register_*``
        decorator — so a model registered by decorator alone shows up here, no
        second edit in this file. Per-model UI hints (``capabilities``,
        ``requires_path``, aliases, extras) come from decorator metadata, not
        hardcoded maps.
        """
        from ..models.registry import FAMILY_REGISTRIES

        # Fallback capability per family when a model declares none.
        default_capabilities = {
            "asr": ["transcription"],
            "text_embedding": ["text_embedding"],
            "audio_embedding": ["audio_embedding"],
            "reranker": ["reranking"],
            "tts": ["speech_synthesis"],
        }

        normalized: Dict[str, List[Dict[str, Any]]] = {}
        for family, registry in FAMILY_REGISTRIES.items():
            entries: List[Dict[str, Any]] = []
            for model_type in sorted(registry.list_types()):
                meta = dict(registry.get_metadata(model_type) or {})
                entry: Dict[str, Any] = {
                    "type": model_type,
                    "name": registry.get_default_name(model_type) or model_type,
                    "capabilities": meta.pop("capabilities", None)
                    or default_capabilities.get(family, []),
                    "requires_path": bool(meta.pop("requires_path", False)),
                    "default_device_hint": "cuda",
                }
                aliases = meta.pop("aliases", None)
                if aliases:
                    entry["aliases"] = list(aliases)
                if meta:
                    entry["metadata"] = meta
                entries.append(entry)
            normalized[family] = entries

        return normalized

    def get_llm_server(self, config):
        """Get or create local LLM server service from LLMServerConfig."""
        key: LLMServerKey = (
            config.backend,
            config.host,
            int(config.port),
            config.model,
            int(config.gpu_layers),
        )
        service = self._llm_server_services.get(key)
        if service is None:
            service = LLMServerService(
                lambda: self._create_llm_server_backend(config),
                label=f"llm_server:{config.backend}@{config.host}:{config.port}",
                auto_start=bool(config.auto_start),
            )
            self._llm_server_services[key] = service
        return service.get()

    @staticmethod
    def _create_tts_backend(config):
        provider_name = (config.provider or "piper").lower()
        if provider_name == "piper":
            from ..models.tts.piper_tts import PiperTTS

            return PiperTTS(config)
        if provider_name in {"xtts", "xtts_v2", "xtts-v2"}:
            from ..models.tts.xtts_v2_tts import XTTSv2TTS

            return XTTSv2TTS(config)
        if provider_name in {"mms", "mms_tts", "mms-tts"}:
            from ..models.tts.mms_tts import MMSTTS

            return MMSTTS(config)
        raise ValueError(
            f"Unknown TTS provider: {provider_name}. "
            f"Supported providers: piper, xtts_v2, mms"
        )

    @staticmethod
    def _create_llm_server_backend(config):
        server = factory_create_llm_server(
            backend=config.backend,
            model=config.model,
            host=config.host,
            port=config.port,
            gpu_layers=config.gpu_layers,
        )
        if server is None:
            raise RuntimeError(
                f"Failed to create local LLM server backend '{config.backend}' "
                f"for model '{config.model}'"
            )
        return server

    def shutdown(self, offload: bool = True) -> None:
        logger.info("provider.shutdown offload=%s", offload)
        self._warm.clear()  # warm services live in the buckets below; stopped there
        for bucket in (
            self._asr_services,
            self._text_services,
            self._audio_services,
            self._reranker_services,
            self._tts_services,
        ):
            if offload:
                for service in bucket.values():
                    service.stop()
            bucket.clear()
        if offload:
            for service in self._llm_server_services.values():
                service.stop(owned_only=True)
        self._llm_server_services.clear()
