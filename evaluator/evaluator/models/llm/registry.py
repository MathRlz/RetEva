"""Model registry with recommended models for different use cases."""

from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum


class ModelSize(Enum):
    """Model size categories."""
    TINY = "tiny"       # < 3B parameters
    SMALL = "small"     # 3-8B parameters
    MEDIUM = "medium"   # 8-15B parameters
    LARGE = "large"     # 15-30B parameters
    XLARGE = "xlarge"   # > 30B parameters


class ModelDomain(Enum):
    """Domain specialization of models."""
    GENERAL = "general"
    MEDICAL = "medical"
    CODE = "code"
    SCIENCE = "science"


@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    display_name: str
    description: str
    size: ModelSize
    domain: ModelDomain
    parameters: str
    context_length: int
    recommended_for: List[str]
    ollama_name: Optional[str] = None
    huggingface_id: Optional[str] = None
    min_ram_gb: int = 8
    supports_gpu: bool = True
    quantization: Optional[str] = None


class ModelRegistry:
    """Registry of recommended models for local LLM serving."""
    
    MODELS: List[ModelInfo] = [
        # ===== GENERAL PURPOSE MODELS =====
        
        # Tiny Models (< 3B) - Fast, Low Memory
        ModelInfo(
            name="phi3-mini",
            display_name="Phi-3 Mini (3.8B)",
            description="Lightweight and fast, good for simple tasks",
            size=ModelSize.TINY,
            domain=ModelDomain.GENERAL,
            parameters="3.8B",
            context_length=4096,
            recommended_for=["query_rewriting", "fast_inference"],
            ollama_name="phi3:mini",
            huggingface_id="microsoft/Phi-3-mini-4k-instruct",
            min_ram_gb=4,
        ),
        ModelInfo(
            name="gemma-2b",
            display_name="Gemma 2B",
            description="Google's small model, efficient and capable",
            size=ModelSize.TINY,
            domain=ModelDomain.GENERAL,
            parameters="2B",
            context_length=8192,
            recommended_for=["fast_inference", "low_memory"],
            ollama_name="gemma:2b",
            huggingface_id="google/gemma-2b-it",
            min_ram_gb=3,
        ),
        ModelInfo(
            name="qwen2-1.5b",
            display_name="Qwen2 1.5B",
            description="Alibaba's tiny model, very fast",
            size=ModelSize.TINY,
            domain=ModelDomain.GENERAL,
            parameters="1.5B",
            context_length=32768,
            recommended_for=["fast_inference", "low_memory"],
            ollama_name="qwen2:1.5b",
            huggingface_id="Qwen/Qwen2-1.5B-Instruct",
            min_ram_gb=2,
        ),
        
        # Small Models (3-8B) - Balanced Performance
        ModelInfo(
            name="mistral-7b-instruct",
            display_name="Mistral 7B Instruct",
            description="Balanced performance and efficiency, great for general tasks",
            size=ModelSize.SMALL,
            domain=ModelDomain.GENERAL,
            parameters="7B",
            context_length=8192,
            recommended_for=["query_rewriting", "judge", "general"],
            ollama_name="mistral:7b-instruct",
            huggingface_id="mistralai/Mistral-7B-Instruct-v0.2",
            min_ram_gb=8,
        ),
        ModelInfo(
            name="llama3-8b-instruct",
            display_name="Llama 3 8B Instruct",
            description="Strong instruction following and reasoning capabilities",
            size=ModelSize.SMALL,
            domain=ModelDomain.GENERAL,
            parameters="8B",
            context_length=8192,
            recommended_for=["query_rewriting", "judge", "general"],
            ollama_name="llama3:8b-instruct",
            huggingface_id="meta-llama/Meta-Llama-3-8B-Instruct",
            min_ram_gb=8,
        ),
        ModelInfo(
            name="llama3.1-8b",
            display_name="Llama 3.1 8B Instruct",
            description="Updated Llama 3 with longer context and better performance",
            size=ModelSize.SMALL,
            domain=ModelDomain.GENERAL,
            parameters="8B",
            context_length=128000,
            recommended_for=["query_rewriting", "judge", "long_context"],
            ollama_name="llama3.1:8b",
            huggingface_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            min_ram_gb=8,
        ),
        ModelInfo(
            name="gemma-7b",
            display_name="Gemma 7B",
            description="Google's 7B model with strong performance",
            size=ModelSize.SMALL,
            domain=ModelDomain.GENERAL,
            parameters="7B",
            context_length=8192,
            recommended_for=["query_rewriting", "judge"],
            ollama_name="gemma:7b",
            huggingface_id="google/gemma-7b-it",
            min_ram_gb=8,
        ),
        ModelInfo(
            name="qwen2-7b",
            display_name="Qwen2 7B",
            description="Alibaba's balanced model with long context support",
            size=ModelSize.SMALL,
            domain=ModelDomain.GENERAL,
            parameters="7B",
            context_length=131072,
            recommended_for=["query_rewriting", "judge", "long_context"],
            ollama_name="qwen2:7b",
            huggingface_id="Qwen/Qwen2-7B-Instruct",
            min_ram_gb=8,
        ),
        
        # Medium Models (8-15B)
        ModelInfo(
            name="mixtral-8x7b",
            display_name="Mixtral 8x7B MoE",
            description="Mixture of Experts model, high quality with reasonable resources",
            size=ModelSize.MEDIUM,
            domain=ModelDomain.GENERAL,
            parameters="47B (8x7B MoE)",
            context_length=32768,
            recommended_for=["judge", "high_quality"],
            ollama_name="mixtral:8x7b",
            huggingface_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            min_ram_gb=24,
        ),
        ModelInfo(
            name="solar-10.7b",
            display_name="SOLAR 10.7B",
            description="Upstage's high-performance model",
            size=ModelSize.MEDIUM,
            domain=ModelDomain.GENERAL,
            parameters="10.7B",
            context_length=4096,
            recommended_for=["judge", "general"],
            ollama_name="solar:10.7b",
            huggingface_id="upstage/SOLAR-10.7B-Instruct-v1.0",
            min_ram_gb=12,
        ),
        
        # Large Models (15-30B)
        ModelInfo(
            name="llama3-70b-instruct",
            display_name="Llama 3 70B Instruct",
            description="High-quality responses, requires significant resources",
            size=ModelSize.LARGE,
            domain=ModelDomain.GENERAL,
            parameters="70B",
            context_length=8192,
            recommended_for=["judge", "high_quality"],
            ollama_name="llama3:70b-instruct",
            huggingface_id="meta-llama/Meta-Llama-3-70B-Instruct",
            min_ram_gb=48,
        ),
        ModelInfo(
            name="llama3.1-70b",
            display_name="Llama 3.1 70B Instruct",
            description="Latest Llama with 128K context and best performance",
            size=ModelSize.LARGE,
            domain=ModelDomain.GENERAL,
            parameters="70B",
            context_length=128000,
            recommended_for=["judge", "high_quality", "long_context"],
            ollama_name="llama3.1:70b",
            huggingface_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            min_ram_gb=48,
        ),
        
        # ===== MEDICAL DOMAIN MODELS =====
        
        # Note: BioMistral not in official Ollama library - removed
        # Use Meditron (official) or general models with medical prompting
        
        ModelInfo(
            name="meditron-7b",
            display_name="Meditron 7B (Medical - VERIFIED)",
            description="Medical domain model trained on medical guidelines and papers. Official Ollama model.",
            size=ModelSize.SMALL,
            domain=ModelDomain.MEDICAL,
            parameters="7B",
            context_length=4096,
            recommended_for=["medical", "judge"],
            ollama_name="meditron:7b",
            huggingface_id="epfl-llm/meditron-7b",
            min_ram_gb=8,
        ),
        # Note: medllama2 not reliably available in Ollama - use meditron instead
        
        
        # ===== QUANTIZED VERSIONS (Low Memory) =====
        
        ModelInfo(
            name="mistral-7b-q4",
            display_name="Mistral 7B (Q4)",
            description="4-bit quantized, 50% less memory usage",
            size=ModelSize.SMALL,
            domain=ModelDomain.GENERAL,
            parameters="7B",
            context_length=8192,
            recommended_for=["query_rewriting", "judge", "low_memory"],
            ollama_name="mistral:7b-instruct-q4_0",
            huggingface_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            min_ram_gb=4,
            quantization="Q4_0",
        ),
        ModelInfo(
            name="llama3-8b-q4",
            display_name="Llama 3 8B (Q4)",
            description="4-bit quantized, reduced memory footprint",
            size=ModelSize.SMALL,
            domain=ModelDomain.GENERAL,
            parameters="8B",
            context_length=8192,
            recommended_for=["query_rewriting", "judge", "low_memory"],
            ollama_name="llama3:8b-instruct-q4_0",
            min_ram_gb=4,
            quantization="Q4_0",
        ),
        ModelInfo(
            name="llama3-70b-q4",
            display_name="Llama 3 70B (Q4)",
            description="4-bit quantized 70B, high quality with reduced memory",
            size=ModelSize.LARGE,
            domain=ModelDomain.GENERAL,
            parameters="70B",
            context_length=8192,
            recommended_for=["judge", "high_quality", "low_memory"],
            ollama_name="llama3:70b-instruct-q4_0",
            min_ram_gb=28,
            quantization="Q4_0",
        ),
        ModelInfo(
            name="mixtral-8x7b-q4",
            display_name="Mixtral 8x7B (Q4)",
            description="4-bit quantized MoE, excellent quality-to-memory ratio",
            size=ModelSize.MEDIUM,
            domain=ModelDomain.GENERAL,
            parameters="47B",
            context_length=32768,
            recommended_for=["judge", "high_quality", "low_memory"],
            ollama_name="mixtral:8x7b-instruct-q4_0",
            min_ram_gb=14,
            quantization="Q4_0",
        ),
        
        # ===== CODE & SCIENCE MODELS =====
        
        ModelInfo(
            name="codellama-7b",
            display_name="CodeLlama 7B",
            description="Specialized for code generation and understanding",
            size=ModelSize.SMALL,
            domain=ModelDomain.CODE,
            parameters="7B",
            context_length=16384,
            recommended_for=["code", "general"],
            ollama_name="codellama:7b-instruct",
            huggingface_id="codellama/CodeLlama-7b-Instruct-hf",
            min_ram_gb=8,
        ),
        ModelInfo(
            name="deepseek-coder-6.7b",
            display_name="DeepSeek Coder 6.7B",
            description="Strong code generation and analysis capabilities",
            size=ModelSize.SMALL,
            domain=ModelDomain.CODE,
            parameters="6.7B",
            context_length=16384,
            recommended_for=["code"],
            ollama_name="deepseek-coder:6.7b",
            huggingface_id="deepseek-ai/deepseek-coder-6.7b-instruct",
            min_ram_gb=8,
        ),
        
        # ===== ADDITIONAL MODELS FOR VARIETY =====
        
        # More Tiny Models
        ModelInfo(
            name="tinyllama-1.1b",
            display_name="TinyLlama 1.1B",
            description="Extremely small and fast, good for testing",
            size=ModelSize.TINY,
            domain=ModelDomain.GENERAL,
            parameters="1.1B",
            context_length=2048,
            recommended_for=["fast_inference", "testing"],
            ollama_name="tinyllama:1.1b",
            huggingface_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            min_ram_gb=1,
        ),
        ModelInfo(
            name="orca-mini-3b",
            display_name="Orca Mini 3B",
            description="Small but capable instruction-following model",
            size=ModelSize.TINY,
            domain=ModelDomain.GENERAL,
            parameters="3B",
            context_length=2048,
            recommended_for=["fast_inference", "low_memory"],
            ollama_name="orca-mini:3b",
            min_ram_gb=3,
        ),
        
        # More Small Models
        ModelInfo(
            name="vicuna-7b",
            display_name="Vicuna 7B",
            description="Fine-tuned LLaMA with strong conversational ability",
            size=ModelSize.SMALL,
            domain=ModelDomain.GENERAL,
            parameters="7B",
            context_length=2048,
            recommended_for=["general", "conversation"],
            ollama_name="vicuna:7b",
            min_ram_gb=8,
        ),
        ModelInfo(
            name="openchat-7b",
            display_name="OpenChat 7B",
            description="High-quality conversational model",
            size=ModelSize.SMALL,
            domain=ModelDomain.GENERAL,
            parameters="7B",
            context_length=8192,
            recommended_for=["conversation", "judge"],
            ollama_name="openchat:7b",
            min_ram_gb=8,
        ),
        ModelInfo(
            name="neural-chat-7b",
            display_name="Neural Chat 7B",
            description="Intel's conversational model",
            size=ModelSize.SMALL,
            domain=ModelDomain.GENERAL,
            parameters="7B",
            context_length=8192,
            recommended_for=["conversation", "general"],
            ollama_name="neural-chat:7b",
            min_ram_gb=8,
        ),
        ModelInfo(
            name="starling-7b",
            display_name="Starling 7B",
            description="RLAIF-trained model with strong performance",
            size=ModelSize.SMALL,
            domain=ModelDomain.GENERAL,
            parameters="7B",
            context_length=8192,
            recommended_for=["general", "judge"],
            ollama_name="starling-lm:7b",
            min_ram_gb=8,
        ),
        
        # More Medium Models
        ModelInfo(
            name="codellama-13b",
            display_name="CodeLlama 13B",
            description="Larger code model for complex tasks",
            size=ModelSize.MEDIUM,
            domain=ModelDomain.CODE,
            parameters="13B",
            context_length=16384,
            recommended_for=["code"],
            ollama_name="codellama:13b-instruct",
            min_ram_gb=16,
        ),
        ModelInfo(
            name="wizardlm-13b",
            display_name="WizardLM 13B",
            description="Strong performance on complex instructions",
            size=ModelSize.MEDIUM,
            domain=ModelDomain.GENERAL,
            parameters="13B",
            context_length=4096,
            recommended_for=["general", "judge"],
            ollama_name="wizardlm:13b",
            min_ram_gb=16,
        ),
        
        # Extra Large Models
        ModelInfo(
            name="llama3.1-405b",
            display_name="Llama 3.1 405B",
            description="Meta's largest and most capable model (requires multi-GPU)",
            size=ModelSize.XLARGE,
            domain=ModelDomain.GENERAL,
            parameters="405B",
            context_length=128000,
            recommended_for=["high_quality", "research"],
            ollama_name="llama3.1:405b",
            huggingface_id="meta-llama/Meta-Llama-3.1-405B-Instruct",
            min_ram_gb=256,
        ),
        
        # More Quantized Versions
        ModelInfo(
            name="gemma-7b-q4",
            display_name="Gemma 7B (Q4)",
            description="Quantized Gemma for lower memory usage",
            size=ModelSize.SMALL,
            domain=ModelDomain.GENERAL,
            parameters="7B",
            context_length=8192,
            recommended_for=["general", "low_memory"],
            ollama_name="gemma:7b-instruct-q4_0",
            min_ram_gb=4,
            quantization="Q4_0",
        ),
        ModelInfo(
            name="qwen2-7b-q4",
            display_name="Qwen2 7B (Q4)",
            description="Quantized Qwen2 with long context",
            size=ModelSize.SMALL,
            domain=ModelDomain.GENERAL,
            parameters="7B",
            context_length=131072,
            recommended_for=["long_context", "low_memory"],
            ollama_name="qwen2:7b-instruct-q4_0",
            min_ram_gb=4,
            quantization="Q4_0",
        ),
        # biomistral removed - not in Ollama library
    ]
    
    @classmethod
    def get_all_models(cls) -> List[ModelInfo]:
        """Get all available models."""
        return cls.MODELS
    
    @classmethod
    def get_model(cls, name: str) -> Optional[ModelInfo]:
        """Get model by name."""
        for model in cls.MODELS:
            if model.name == name or model.ollama_name == name:
                return model
        return None
    
    @classmethod
    def get_models_by_domain(cls, domain: ModelDomain) -> List[ModelInfo]:
        """Get models filtered by domain."""
        return [m for m in cls.MODELS if m.domain == domain]
    
    @classmethod
    def get_models_by_size(cls, size: ModelSize) -> List[ModelInfo]:
        """Get models filtered by size."""
        return [m for m in cls.MODELS if m.size == size]
    
    @classmethod
    def get_recommended_for_task(cls, task: str) -> List[ModelInfo]:
        """
        Get models recommended for a specific task.
        
        Args:
            task: Task name (e.g., "judge", "query_rewriting", "medical")
            
        Returns:
            List of recommended models
        """
        return [m for m in cls.MODELS if task in m.recommended_for]
    
    @classmethod
    def get_models_within_ram(cls, max_ram_gb: int) -> List[ModelInfo]:
        """Get models that fit within specified RAM."""
        return [m for m in cls.MODELS if m.min_ram_gb <= max_ram_gb]
    
    @classmethod
    def get_default_model(cls) -> ModelInfo:
        """Get the default recommended model."""
        return cls.get_model("mistral-7b-instruct")
    
    @classmethod
    def get_default_medical_model(cls) -> ModelInfo:
        """Get the default recommended model for medical domain."""
        medical = cls.get_model("meditron-7b")
        if medical:
            return medical
        return cls.get_default_model()
    
    @classmethod
    def to_dict_list(cls) -> List[Dict]:
        """Convert all models to dictionary format for API/UI."""
        return [
            {
                "name": m.name,
                "display_name": m.display_name,
                "description": m.description,
                "size": m.size.value,
                "domain": m.domain.value,
                "parameters": m.parameters,
                "context_length": m.context_length,
                "recommended_for": m.recommended_for,
                "ollama_name": m.ollama_name,
                "min_ram_gb": m.min_ram_gb,
                "quantization": m.quantization,
            }
            for m in cls.MODELS
        ]
