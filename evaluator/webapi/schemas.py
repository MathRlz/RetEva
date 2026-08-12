"""Shared request/response schemas for evaluator WebAPI."""

from typing import Any, Dict

from pydantic import BaseModel, Field


class EvaluationJobRequest(BaseModel):
    """Payload for single evaluation job creation."""

    config: Dict[str, Any] = Field(
        default_factory=dict, description="EvaluationConfig dict (flat or nested)")
    auto_devices: bool = Field(
        True, description="Auto-configure device assignments based on hardware")


class GraphRunRequest(BaseModel):
    """Payload to run a graph built in the visual builder.

    ``spec`` is the canvas export: ``{mode, nodes:[{id,type,params}], edges:[{from,to}],
    branches?, llm?}``. It is translated to a config and submitted as a normal job.
    """

    spec: Dict[str, Any] = Field(default_factory=dict, description="Builder canvas spec")
    experiment_name: str = Field("builder_run", description="Name for the run")
    auto_devices: bool = Field(True, description="Auto-configure device assignments")


class HealthResponse(BaseModel):
    status: str = "ok"


class JobSubmitResponse(BaseModel):
    job_id: str


class ErrorResponse(BaseModel):
    detail: str
