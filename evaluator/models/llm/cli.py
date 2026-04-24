"""Command-line interface for managing local LLM server."""

import sys
import argparse
import logging
from typing import Optional

from llm_server.factory import create_llm_server
from llm_server.model_registry import ModelRegistry
from llm_server.base import ServerStatus

logger = logging.getLogger(__name__)


def start_server(args):
    """Start the local LLM server."""
    print(f"Starting {args.backend} server with model {args.model}...")
    
    server = create_llm_server(
        backend=args.backend,
        model=args.model,
        host=args.host,
        port=args.port,
        gpu_layers=args.gpu_layers
    )
    
    if server is None:
        print(f"Error: Failed to create {args.backend} server")
        print(f"Make sure {args.backend} is installed")
        return 1
    
    if server.start():
        print(f"✓ Server started successfully")
        print(f"  API URL: {server.get_api_url()}")
        print(f"  Model: {server.model}")
        print(f"  Status: {server.status.value}")
        
        # Run health check
        health = server.health_check()
        if health.is_healthy:
            print(f"  Health: ✓ Healthy (response time: {health.response_time_ms:.0f}ms)")
        else:
            print(f"  Health: ⚠ {health.message}")
        
        return 0
    else:
        print(f"✗ Failed to start server")
        return 1


def stop_server(args):
    """Stop the local LLM server."""
    print(f"Stopping {args.backend} server...")
    
    server = create_llm_server(
        backend=args.backend,
        model="",  # Not needed for stopping
        host=args.host,
        port=args.port
    )
    
    if server is None:
        print(f"Error: Failed to create {args.backend} server instance")
        return 1
    
    if server.stop():
        print(f"✓ Server stopped")
        return 0
    else:
        print(f"Note: Server may already be stopped or running as system service")
        return 0


def status_server(args):
    """Check server status and health."""
    print(f"Checking {args.backend} server status...")
    
    server = create_llm_server(
        backend=args.backend,
        model=args.model or "unknown",
        host=args.host,
        port=args.port
    )
    
    if server is None:
        print(f"Error: Failed to create {args.backend} server instance")
        return 1
    
    # Get status
    status_info = server.get_status()
    
    print(f"\nServer Status:")
    print(f"  Backend: {server.get_backend_name()}")
    print(f"  Host: {status_info['host']}")
    print(f"  Port: {status_info['port']}")
    print(f"  Model: {status_info['model']}")
    print(f"  API URL: {status_info['api_url']}")
    print(f"  Status: {status_info['status']}")
    
    if status_info['is_healthy']:
        print(f"  Health: ✓ Healthy")
        if status_info['response_time_ms']:
            print(f"  Response Time: {status_info['response_time_ms']:.0f}ms")
        return 0
    else:
        print(f"  Health: ✗ Not responding")
        return 1


def list_models(args):
    """List available models."""
    print("Available Models:\n")
    
    models = ModelRegistry.get_all_models()
    
    # Filter by domain if specified
    if args.domain:
        from llm_server.model_registry import ModelDomain
        try:
            domain = ModelDomain(args.domain)
            models = [m for m in models if m.domain == domain]
        except ValueError:
            print(f"Invalid domain: {args.domain}")
            return 1
    
    # Filter by RAM if specified
    if args.max_ram:
        models = [m for m in models if m.min_ram_gb <= args.max_ram]
    
    for model in models:
        print(f"• {model.display_name}")
        print(f"  Name: {model.name}")
        if model.ollama_name:
            print(f"  Ollama: {model.ollama_name}")
        print(f"  Description: {model.description}")
        print(f"  Parameters: {model.parameters}")
        print(f"  Domain: {model.domain.value}")
        print(f"  Min RAM: {model.min_ram_gb}GB")
        print(f"  Recommended for: {', '.join(model.recommended_for)}")
        if model.quantization:
            print(f"  Quantization: {model.quantization}")
        print()
    
    return 0


def download_model(args):
    """Download a model."""
    print(f"Downloading model: {args.model}")
    
    # Get model info
    model_info = ModelRegistry.get_model(args.model)
    if model_info:
        print(f"Model: {model_info.display_name}")
        print(f"Size: {model_info.parameters}")
        print(f"Min RAM: {model_info.min_ram_gb}GB\n")
    
    # For Ollama, use the pull command via server
    if args.backend == "ollama":
        from llm_server.ollama import OllamaServer
        
        # Use Ollama name if available
        model_name = args.model
        if model_info and model_info.ollama_name:
            model_name = model_info.ollama_name
            print(f"Using Ollama model name: {model_name}")
        
        server = OllamaServer(model=model_name)
        
        # This will pull the model if not present
        if server._check_ollama_installed():
            print("Pulling model (this may take a while)...")
            if server._pull_model():
                print(f"✓ Model {model_name} downloaded successfully")
                return 0
            else:
                print(f"✗ Failed to download model {model_name}")
                return 1
        else:
            print("✗ Ollama not installed. Install from: https://ollama.ai")
            return 1
    else:
        print(f"Download not yet implemented for backend: {args.backend}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manage local LLM server for evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start LLM server")
    start_parser.add_argument(
        "--backend",
        default="ollama",
        choices=["ollama", "vllm", "llamacpp"],
        help="Server backend (default: ollama)"
    )
    start_parser.add_argument(
        "--model",
        default="mistral:7b-instruct",
        help="Model name (default: mistral:7b-instruct)"
    )
    start_parser.add_argument("--host", default="localhost", help="Server host")
    start_parser.add_argument("--port", type=int, default=11434, help="Server port")
    start_parser.add_argument(
        "--gpu-layers",
        type=int,
        default=-1,
        help="GPU layers (-1=all, 0=CPU only)"
    )
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop LLM server")
    stop_parser.add_argument(
        "--backend",
        default="ollama",
        choices=["ollama", "vllm", "llamacpp"],
        help="Server backend"
    )
    stop_parser.add_argument("--host", default="localhost", help="Server host")
    stop_parser.add_argument("--port", type=int, default=11434, help="Server port")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check server status")
    status_parser.add_argument(
        "--backend",
        default="ollama",
        choices=["ollama", "vllm", "llamacpp"],
        help="Server backend"
    )
    status_parser.add_argument("--model", help="Model name")
    status_parser.add_argument("--host", default="localhost", help="Server host")
    status_parser.add_argument("--port", type=int, default=11434, help="Server port")
    
    # List models command
    list_parser = subparsers.add_parser("list-models", help="List available models")
    list_parser.add_argument(
        "--domain",
        choices=["general", "medical", "code", "science"],
        help="Filter by domain"
    )
    list_parser.add_argument(
        "--max-ram",
        type=int,
        help="Filter by max RAM (GB)"
    )
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download a model")
    download_parser.add_argument("--model", required=True, help="Model name")
    download_parser.add_argument(
        "--backend",
        default="ollama",
        choices=["ollama", "vllm", "llamacpp"],
        help="Server backend"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Execute command
    if args.command == "start":
        return start_server(args)
    elif args.command == "stop":
        return stop_server(args)
    elif args.command == "status":
        return status_server(args)
    elif args.command == "list-models":
        return list_models(args)
    elif args.command == "download":
        return download_model(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
