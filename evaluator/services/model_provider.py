"""Service provider for model lifecycle orchestration."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Any, MutableMapping, List
import logging

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

ASRKey = Tuple[str, Optional[str], Optional[str], str]
TextKey = Tuple[str, Optional[str], str]
AudioKey = Tuple[str, Optional[str], Optional[str], int, float, str]
RerankerKey = Tuple[str, Optional[str], Optional[str], int, int]
TTSKey = Tuple[str, str, str, int]
LLMServerKey = Tuple[str, str, int, str, int]
logger = logging.getLogger(__name__)


class ModelServiceProvider:
    """Creates, reuses, and tears down model services for one runtime."""

    def __init__(self) -> None:
        self._asr_services: Dict[ASRKey, FactoryModelService] = {}
        self._text_services: Dict[TextKey, FactoryModelService] = {}
        self._audio_services: Dict[AudioKey, FactoryModelService] = {}
        self._reranker_services: Dict[RerankerKey, FactoryModelService[Any]] = {}
        self._tts_services: Dict[TTSKey, FactoryModelService] = {}
        self._llm_server_services: Dict[LLMServerKey, LLMServerService] = {}

    def get_asr_model(
        self,
        model_type: str,
        model_name: Optional[str],
        adapter_path: Optional[str],
        device: str,
    ):
        key: ASRKey = (model_type, model_name, adapter_path, device)
        service = self._asr_services.get(key)
        if service is None:
            service = FactoryModelService(
                lambda: factory_create_asr_model(
                    model_type=model_type,
                    model_name=model_name,
                    adapter_path=adapter_path,
                    device=device,
                ),
                label=f"asr:{model_type}@{device}",
            )
            self._asr_services[key] = service
        return service.get()

    def get_text_embedding_model(
        self,
        model_type: str,
        model_name: Optional[str],
        device: str,
    ):
        key: TextKey = (model_type, model_name, device)
        service = self._text_services.get(key)
        if service is None:
            service = FactoryModelService(
                lambda: factory_create_text_embedding_model(
                    model_type=model_type,
                    model_name=model_name,
                    device=device,
                ),
                label=f"text_embedding:{model_type}@{device}",
            )
            self._text_services[key] = service
        return service.get()

    def get_audio_embedding_model(
        self,
        model_type: str,
        model_name: Optional[str],
        model_path: Optional[str],
        emb_dim: int,
        dropout: float,
        device: str,
    ):
        key: AudioKey = (model_type, model_name, model_path, emb_dim, dropout, device)
        service = self._audio_services.get(key)
        if service is None:
            service = FactoryModelService(
                lambda: factory_create_audio_embedding_model(
                    model_type=model_type,
                    model_name=model_name,
                    model_path=model_path,
                    emb_dim=emb_dim,
                    dropout=dropout,
                    device=device,
                ),
                label=f"audio_embedding:{model_type}@{device}",
            )
            self._audio_services[key] = service
        return service.get()

    def get_reranker(
        self,
        model_type: str = "cross_encoder",
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        key: RerankerKey = (model_type, model_name, device, batch_size, max_length)
        service = self._reranker_services.get(key)
        if service is None:
            service = FactoryModelService(
                lambda: factory_create_reranker(
                    model_type=model_type,
                    model_name=model_name,
                    device=device,
                    batch_size=batch_size,
                    max_length=max_length,
                ),
                label=f"reranker:{model_type}@{device}",
            )
            self._reranker_services[key] = service
        return service.get()

    def move_asr_model(
        self,
        model_type: str,
        model_name: Optional[str],
        adapter_path: Optional[str],
        current_device: str,
        target_device: str,
    ):
        old_key: ASRKey = (model_type, model_name, adapter_path, current_device)
        new_key: ASRKey = (model_type, model_name, adapter_path, target_device)
        return self._move_service(
            self._asr_services, old_key, new_key, target_device, "ASR"
        )

    def move_text_embedding_model(
        self,
        model_type: str,
        model_name: Optional[str],
        current_device: str,
        target_device: str,
    ):
        old_key: TextKey = (model_type, model_name, current_device)
        new_key: TextKey = (model_type, model_name, target_device)
        return self._move_service(
            self._text_services,
            old_key,
            new_key,
            target_device,
            "TextEmbedding",
        )

    def move_audio_embedding_model(
        self,
        model_type: str,
        model_name: Optional[str],
        model_path: Optional[str],
        emb_dim: int,
        dropout: float,
        current_device: str,
        target_device: str,
    ):
        old_key: AudioKey = (
            model_type,
            model_name,
            model_path,
            emb_dim,
            dropout,
            current_device,
        )
        new_key: AudioKey = (
            model_type,
            model_name,
            model_path,
            emb_dim,
            dropout,
            target_device,
        )
        return self._move_service(
            self._audio_services,
            old_key,
            new_key,
            target_device,
            "AudioEmbedding",
        )

    def move_reranker(
        self,
        model_type: str = "cross_encoder",
        model_name: Optional[str] = None,
        current_device: Optional[str] = None,
        target_device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        if target_device is None:
            raise ValueError("target_device must be provided to move reranker.")
        old_key: RerankerKey = (
            model_type,
            model_name,
            current_device,
            batch_size,
            max_length,
        )
        new_key: RerankerKey = (
            model_type,
            model_name,
            target_device,
            batch_size,
            max_length,
        )
        return self._move_service(
            self._reranker_services,
            old_key,
            new_key,
            target_device,
            "Reranker",
        )

    def release_asr_model(
        self,
        model_type: str,
        model_name: Optional[str],
        adapter_path: Optional[str],
        device: str,
    ) -> None:
        key: ASRKey = (model_type, model_name, adapter_path, device)
        self._release_service(self._asr_services, key)

    def release_text_embedding_model(
        self,
        model_type: str,
        model_name: Optional[str],
        device: str,
    ) -> None:
        key: TextKey = (model_type, model_name, device)
        self._release_service(self._text_services, key)

    def release_audio_embedding_model(
        self,
        model_type: str,
        model_name: Optional[str],
        model_path: Optional[str],
        emb_dim: int,
        dropout: float,
        device: str,
    ) -> None:
        key: AudioKey = (model_type, model_name, model_path, emb_dim, dropout, device)
        self._release_service(self._audio_services, key)

    def release_reranker(
        self,
        model_type: str = "cross_encoder",
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        key: RerankerKey = (model_type, model_name, device, batch_size, max_length)
        self._release_service(self._reranker_services, key)

    def release_model_instance(self, model: object) -> bool:
        for bucket in (
            self._asr_services,
            self._text_services,
            self._audio_services,
            self._reranker_services,
            self._tts_services,
        ):
            for key, service in list(bucket.items()):
                if service._instance is model:  # service identity check only
                    service.stop()
                    del bucket[key]
                    logger.info("provider.release_instance key=%s", key)
                    return True
        return False

    def list_available_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """List all available models with normalized metadata schema.

        Fully registry-driven: importing ``evaluator.models`` triggers every
        model's ``@register_*`` decorator (each family subpackage is imported in
        ``models/__init__.py``), so a model registered by decorator alone shows
        up here — no second edit in this file. Per-model UI hints
        (``capabilities``, ``requires_path``, aliases, extras) come from
        decorator metadata, not hardcoded maps.
        """
        import evaluator.models  # noqa: F401  (populates all registries)
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

    def get_tts_model(self, config):
        """Get or create TTS service instance from AudioSynthesisConfig."""
        key: TTSKey = (
            (config.provider or "piper").lower(),
            config.voice or "",
            config.language or "en",
            int(config.sample_rate),
        )
        service = self._tts_services.get(key)
        if service is None:
            service = FactoryModelService(
                lambda: self._create_tts_backend(config),
                label=f"tts:{key[0]}",
            )
            self._tts_services[key] = service
        return service.get()

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

    def release_llm_server(self, config, *, owned_only: bool = True) -> None:
        key: LLMServerKey = (
            config.backend,
            config.host,
            int(config.port),
            config.model,
            int(config.gpu_layers),
        )
        service = self._llm_server_services.pop(key, None)
        if service is not None:
            service.stop(owned_only=owned_only)
            logger.info("provider.release key=%s", key)

    def release_tts_model(self, config) -> None:
        key: TTSKey = (
            (config.provider or "piper").lower(),
            config.voice or "",
            config.language or "en",
            int(config.sample_rate),
        )
        self._release_service(self._tts_services, key)

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

    @staticmethod
    def _release_service(
        bucket: MutableMapping[Any, FactoryModelService[Any]],
        key: Any,
    ) -> None:
        service = bucket.pop(key, None)
        if service is not None:
            service.stop()
            logger.info("provider.release key=%s", key)

    @staticmethod
    def _move_service(
        bucket: MutableMapping[Any, FactoryModelService[Any]],
        old_key: Any,
        new_key: Any,
        target_device: str,
        label: str,
    ):
        service = bucket.pop(old_key, None)
        if service is None:
            raise KeyError(f"{label} service not found for key={old_key}")
        service.move_to_device(target_device)
        bucket[new_key] = service
        logger.info("provider.move label=%s from=%s to=%s", label, old_key, new_key)
        return service.get()

    def shutdown(self, offload: bool = True) -> None:
        logger.info("provider.shutdown offload=%s", offload)
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
