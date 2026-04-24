"""Ollama backend implementation for local LLM serving."""

import logging
import subprocess
import time
import os
import requests
from typing import Optional

from .base import BaseLLMServer, ServerStatus

logger = logging.getLogger(__name__)


class OllamaServer(BaseLLMServer):
    """Ollama backend for local LLM serving."""
    
    def __init__(
        self,
        model: str,
        host: str = "localhost",
        port: int = 11434,
        gpu_layers: int = -1,
        **kwargs
    ):
        """
        Initialize Ollama server.
        
        Args:
            model: Ollama model name (e.g., "mistral:7b-instruct")
            host: Server host
            port: Server port
            gpu_layers: Number of GPU layers (-1 for all)
            **kwargs: Additional Ollama parameters
        """
        super().__init__(model, host, port, gpu_layers, **kwargs)
        self.ollama_host = os.getenv("OLLAMA_HOST", f"http://{host}:{port}")
        
    def start(self, progress_callback=None) -> bool:
        """
        Start Ollama server.
        
        Ollama typically runs as a system service. This method checks
        if it's running and pulls the model if needed.
        
        Args:
            progress_callback: Optional callback(status, message, progress) to report progress
        
        Returns:
            True if server is ready, False otherwise
        """
        logger.info(f"Checking Ollama server at {self.ollama_host}")
        self.status = ServerStatus.STARTING
        
        if progress_callback:
            progress_callback("checking", "Checking Ollama installation...", 5)
        
        # Check if Ollama is installed
        if not self._check_ollama_installed():
            logger.error("Ollama is not installed. Please install from https://ollama.ai")
            self.status = ServerStatus.ERROR
            if progress_callback:
                progress_callback("error", "Ollama not installed", 0)
            return False
        
        if progress_callback:
            progress_callback("checking", "Checking if Ollama service is running...", 15)
        
        # Check if server is running
        if not self._check_server_running():
            logger.info("Starting Ollama service...")
            if progress_callback:
                progress_callback("starting", "Starting Ollama service...", 25)
            if not self._start_ollama_service():
                self.status = ServerStatus.ERROR
                if progress_callback:
                    progress_callback("error", "Failed to start Ollama service", 0)
                return False
        
        # Wait for server to be ready
        if progress_callback:
            progress_callback("checking", "Waiting for Ollama service...", 35)
        
        if not self._wait_for_server(timeout=30):
            logger.error("Ollama server failed to start")
            self.status = ServerStatus.ERROR
            if progress_callback:
                progress_callback("error", "Ollama service timeout", 0)
            return False
        
        # Pull model if not present
        if progress_callback:
            progress_callback("checking", f"Checking if model {self.model} is available...", 45)
        
        if not self._check_model_present():
            logger.info(f"Model {self.model} not found, pulling...")
            if progress_callback:
                progress_callback("downloading", f"Model {self.model} not found. Starting download...", 50)
            if not self._pull_model(progress_callback):
                logger.error(f"Failed to pull model {self.model}")
                self.status = ServerStatus.ERROR
                if progress_callback:
                    progress_callback("error", f"Failed to download {self.model}", 0)
                return False
        else:
            if progress_callback:
                progress_callback("checking", f"Model {self.model} already available", 95)
        
        self.status = ServerStatus.RUNNING
        logger.info(f"Ollama server ready with model {self.model}")
        if progress_callback:
            progress_callback("complete", f"Server ready with {self.model}!", 100)
        return True
    
    def stop(self) -> bool:
        """
        Stop Ollama server by unloading the model.
        
        Note: Ollama runs as a system service. We unload the model to free memory
        but the service continues running for other users/models.
        
        Returns:
            True if model unloaded successfully
        """
        logger.info(f"Unloading model {self.model} from Ollama")
        try:
            # Try to stop the model to free memory
            # First check if there are running processes
            result = subprocess.run(
                ["ollama", "ps"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if self.model in result.stdout:
                logger.info(f"Stopping model {self.model}...")
                # Stop the specific model
                subprocess.run(
                    ["ollama", "stop", self.model],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
            
            self.status = ServerStatus.STOPPED
            logger.info(f"Model {self.model} unloaded (Ollama service still running)")
            return True
            
        except Exception as e:
            logger.warning(f"Could not unload model: {e}")
            # Still mark as stopped even if unload failed
            self.status = ServerStatus.STOPPED
            return True
    
    def get_backend_name(self) -> str:
        """Get backend name."""
        return "ollama"
    
    def _check_ollama_installed(self) -> bool:
        """Check if Ollama is installed."""
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _check_server_running(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(
                f"{self.ollama_host}/api/tags",
                timeout=2
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def _start_ollama_service(self) -> bool:
        """
        Try to start Ollama service.
        
        Returns:
            True if started or already running
        """
        try:
            # Try to start Ollama serve in background
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            time.sleep(2)
            return True
        except Exception as e:
            logger.warning(f"Could not start Ollama service: {e}")
            # Service might already be running
            return self._check_server_running()
    
    def _wait_for_server(self, timeout: int = 30) -> bool:
        """Wait for server to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._check_server_running():
                return True
            time.sleep(1)
        return False
    
    def _check_model_present(self) -> bool:
        """Check if model is already pulled."""
        try:
            response = requests.get(
                f"{self.ollama_host}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                # Check if our model is in the list
                return any(self.model in name for name in model_names)
        except Exception as e:
            logger.debug(f"Error checking models: {e}")
        return False
    
    def _pull_model(self, progress_callback=None) -> bool:
        """
        Pull model from Ollama registry.
        
        Args:
            progress_callback: Optional callback(status, message, progress) to report progress
        """
        try:
            logger.info(f"Pulling model {self.model} (this may take a while)...")
            
            if progress_callback:
                progress_callback("downloading", f"Downloading {self.model}...", 0)
            
            # Use Popen to stream output
            process = subprocess.Popen(
                ["ollama", "pull", self.model],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Track progress from output
            for line in process.stdout:
                line = line.strip()
                if line:
                    logger.debug(f"Pull output: {line}")
                    
                    # Parse progress from output
                    # Ollama outputs lines like: "pulling manifest", "pulling <hash>", etc.
                    if progress_callback:
                        if "pulling manifest" in line.lower():
                            progress_callback("downloading", "Downloading model manifest...", 10)
                        elif "verifying" in line.lower():
                            progress_callback("downloading", "Verifying download...", 90)
                        elif "success" in line.lower():
                            progress_callback("complete", "Download complete!", 100)
                        elif line.startswith("pulling"):
                            # Extracting percentage if available
                            progress_callback("downloading", line, 50)
            
            process.wait()
            
            if process.returncode == 0:
                logger.info(f"Model {self.model} pulled successfully")
                if progress_callback:
                    progress_callback("complete", f"Model {self.model} ready!", 100)
                return True
            else:
                logger.error(f"Failed to pull model")
                if progress_callback:
                    progress_callback("error", f"Failed to download {self.model}", 0)
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Model pull timed out after 10 minutes")
            return False
        except Exception as e:
            logger.error(f"Error pulling model: {e}")
            return False
    
    def get_api_url(self) -> str:
        """Get OpenAI-compatible API URL."""
        return f"{self.ollama_host}/v1/chat/completions"
