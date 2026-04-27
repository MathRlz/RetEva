"""Base classes for LLM server implementations."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
import requests


logger = logging.getLogger(__name__)


class ServerStatus(Enum):
    """Status of the LLM server."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


@dataclass
class ServerHealth:
    """Health check result for LLM server."""
    is_healthy: bool
    status: ServerStatus
    message: str
    model_loaded: Optional[str] = None
    response_time_ms: Optional[float] = None


class BaseLLMServer(ABC):
    """Abstract base class for LLM server implementations."""
    
    def __init__(
        self,
        model: str,
        host: str = "localhost",
        port: int = 11434,
        gpu_layers: int = -1,
        **kwargs
    ):
        """
        Initialize LLM server.
        
        Args:
            model: Model name or path
            host: Server host address
            port: Server port
            gpu_layers: Number of GPU layers (-1 for all, 0 for CPU only)
            **kwargs: Additional backend-specific parameters
        """
        self.model = model
        self.host = host
        self.port = port
        self.gpu_layers = gpu_layers
        self.kwargs = kwargs
        self.status = ServerStatus.STOPPED
        self._process = None
        
    @abstractmethod
    def start(self) -> bool:
        """
        Start the LLM server.
        
        Returns:
            True if server started successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def stop(self) -> bool:
        """
        Stop the LLM server.
        
        Returns:
            True if server stopped successfully, False otherwise
        """
        pass
    
    def health_check(self, timeout: int = 5) -> ServerHealth:
        """
        Check if server is healthy and responding.
        
        Args:
            timeout: Request timeout in seconds
            
        Returns:
            ServerHealth object with check results
        """
        import time
        start_time = time.time()
        
        try:
            url = self.get_api_url()
            # Try a simple health check endpoint or chat completion
            response = requests.get(
                f"http://{self.host}:{self.port}/health",
                timeout=timeout
            )
            
            if response.status_code == 200:
                elapsed_ms = (time.time() - start_time) * 1000
                return ServerHealth(
                    is_healthy=True,
                    status=ServerStatus.RUNNING,
                    message="Server is healthy",
                    model_loaded=self.model,
                    response_time_ms=elapsed_ms
                )
        except requests.exceptions.RequestException as e:
            logger.debug(f"Health check failed: {e}")
        
        # Fallback: try chat completions endpoint
        try:
            response = requests.post(
                self.get_api_url(),
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 1
                },
                timeout=timeout
            )
            
            if response.status_code in (200, 400):  # 400 might be format issue but server is up
                elapsed_ms = (time.time() - start_time) * 1000
                return ServerHealth(
                    is_healthy=True,
                    status=ServerStatus.RUNNING,
                    message="Server responding to API calls",
                    model_loaded=self.model,
                    response_time_ms=elapsed_ms
                )
        except requests.exceptions.RequestException as e:
            logger.debug(f"API endpoint check failed: {e}")
        
        return ServerHealth(
            is_healthy=False,
            status=self.status,
            message="Server not responding",
            model_loaded=None,
            response_time_ms=None
        )
    
    def get_api_url(self) -> str:
        """
        Get the OpenAI-compatible API endpoint URL.
        
        Returns:
            Full URL to chat completions endpoint
        """
        return f"http://{self.host}:{self.port}/v1/chat/completions"
    
    def get_base_url(self) -> str:
        """
        Get the base URL for the server.
        
        Returns:
            Base URL without endpoint path
        """
        return f"http://{self.host}:{self.port}"
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current server status information.
        
        Returns:
            Dictionary with status information
        """
        health = self.health_check()
        return {
            "status": self.status.value,
            "model": self.model,
            "host": self.host,
            "port": self.port,
            "api_url": self.get_api_url(),
            "is_healthy": health.is_healthy,
            "response_time_ms": health.response_time_ms,
        }
    
    @abstractmethod
    def get_backend_name(self) -> str:
        """
        Get the name of the backend implementation.
        
        Returns:
            Backend name (e.g., "ollama", "vllm", "llamacpp")
        """
        pass
