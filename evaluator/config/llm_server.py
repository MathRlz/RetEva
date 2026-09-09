"""LLM server configuration."""
from dataclasses import dataclass


@dataclass
class LLMServerConfig:
    """Configuration for a local LLM server (Ollama, vLLM, llama.cpp) instead of a paid
    API, for query optimization and judging."""
    enabled: bool = False
    backend: str = "ollama"  # ollama | vllm | llamacpp
    host: str = "localhost"
    port: int = 11434
    model: str = "mistral:7b-instruct"
    auto_start: bool = True
    gpu_layers: int = -1  # -1 = all layers on GPU, 0 = CPU only
    timeout_s: int = 60

    def __post_init__(self):
        valid_backends = {"ollama", "vllm", "llamacpp"}
        if self.backend not in valid_backends:
            raise ValueError(
                f"backend must be one of {valid_backends}, got {self.backend}"
            )

        if self.port < 1 or self.port > 65535:
            raise ValueError(f"port must be in [1, 65535], got {self.port}")

    def get_api_url(self) -> str:
        """Get the OpenAI-compatible API endpoint URL."""
        return f"http://{self.host}:{self.port}/v1/chat/completions"
