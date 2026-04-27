"""LLM server configuration."""
from dataclasses import dataclass


@dataclass
class LLMServerConfig:
    """
    Configuration for local LLM server.
    
    Enables running local LLM models (Ollama, vLLM, llama.cpp) instead of
    relying on paid API services for query optimization and judging.
    
    Attributes:
        enabled: Whether to use local LLM server. Default: False.
        backend: Server backend to use. Default: "ollama".
            Options: "ollama", "vllm", "llamacpp".
        host: Server host address. Default: "localhost".
        port: Server port. Default: 11434 (Ollama default).
        model: Model name or path. Default: "mistral:7b-instruct".
        auto_start: Auto-start server if not running. Default: True.
        gpu_layers: Number of GPU layers (-1 for all, 0 for CPU only). Default: -1.
        timeout_s: Server startup timeout in seconds. Default: 60.
    
    Examples:
        >>> # Use Ollama with Mistral
        >>> config = LLMServerConfig(
        ...     enabled=True,
        ...     backend="ollama",
        ...     model="mistral:7b-instruct"
        ... )
        >>> 
        >>> # Use medical domain model
        >>> config = LLMServerConfig(
        ...     enabled=True,
        ...     backend="ollama",
        ...     model="biomistral:7b",
        ...     port=11434
        ... )
    """
    enabled: bool = False
    backend: str = "ollama"  # ollama | vllm | llamacpp
    host: str = "localhost"
    port: int = 11434
    model: str = "mistral:7b-instruct"
    auto_start: bool = True
    gpu_layers: int = -1
    timeout_s: int = 60
    
    def __post_init__(self):
        """Validate configuration."""
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
