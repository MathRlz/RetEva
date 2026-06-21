"""Shared request/response schemas for evaluator WebAPI."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EvaluationJobRequest(BaseModel):
    """Payload for single evaluation job creation."""

    config: Dict[str, Any] = Field(default_factory=dict, description="EvaluationConfig dict (flat or nested)")
    auto_devices: bool = Field(True, description="Auto-configure device assignments based on hardware")


class GraphRunRequest(BaseModel):
    """Payload to run a graph built in the visual builder.

    ``spec`` is the canvas export: ``{mode, nodes:[{id,type,params}], edges:[{from,to}],
    branches?, llm?}``. It is translated to a config and submitted as a normal job.
    """

    spec: Dict[str, Any] = Field(default_factory=dict, description="Builder canvas spec")
    experiment_name: str = Field("builder_run", description="Name for the run")
    auto_devices: bool = Field(True, description="Auto-configure device assignments")


class MatrixJobRequest(BaseModel):
    """Payload for matrix evaluation job creation."""

    base_config: Dict[str, Any] = Field(default_factory=dict, description="Base EvaluationConfig dict")
    test_setups: List[Dict[str, Any]] = Field(default_factory=list, description="List of override dicts per setup")
    baseline_setup_id: Optional[str] = Field(
        None,
        description="setup_id to use as comparison baseline (defaults to first setup)",
    )
    auto_devices: bool = True


class ConfigCreateRequest(BaseModel):
    """Payload for config creator endpoint."""

    preset_name: Optional[str] = Field(None, description="Preset to start from (e.g. 'fast_dev')")
    config_patch: Dict[str, Any] = Field(default_factory=dict, description="Nested patch dict to merge over preset")
    auto_devices: bool = True


class LiveQueryRequest(BaseModel):
    """Payload for ad-hoc live retrieval query."""

    config: Dict[str, Any] = Field(default_factory=dict, description="EvaluationConfig dict for pipeline setup")
    query_text: str = Field("", description="Query text to search")
    k: int = Field(5, description="Number of results to return")
    auto_devices: bool = True


class TtsPreviewRequest(BaseModel):
    """Payload for a one-off TTS synthesis preview."""

    text: str = Field(..., description="Text to synthesize")
    provider: str = Field("mms", description="TTS provider (registry type or alias)")
    voice: Optional[str] = Field(None, description="Voice id / model name (provider-specific)")
    language: str = Field("en", description="Language code (provider-specific)")
    sample_rate: int = Field(16000, description="Output sample rate in Hz")


class HealthResponse(BaseModel):
    status: str = "ok"


class JobSubmitResponse(BaseModel):
    job_id: str


class ErrorResponse(BaseModel):
    detail: str
