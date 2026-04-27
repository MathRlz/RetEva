# GPU Management

Advanced GPU allocation and management for multi-GPU systems.

## Overview

The evaluator includes a sophisticated GPU pool system for:

- Automatic device allocation across multiple GPUs
- Load balancing strategies
- Memory-aware scheduling
- Concurrent model execution

## GPU Pool

### Basic Usage

```yaml
gpu:
  gpu_pool:
    enabled: true
    device_ids: [0, 1, 2, 3]
    allocation_strategy: memory_balanced
```

```python
from evaluator.devices import GPUPool

# Initialize GPU pool
pool = GPUPool(device_ids=[0, 1, 2, 3])

# Allocate devices automatically
asr_device = pool.allocate("asr_model")
emb_device = pool.allocate("embedding_model")

print(f"ASR on {asr_device}, Embedding on {emb_device}")
# Output: ASR on cuda:0, Embedding on cuda:1
```

### Allocation Strategies

#### Round Robin

Distributes models evenly across GPUs in order.

```yaml
gpu:
  gpu_pool:
    allocation_strategy: round_robin
```

- **Best for**: Equal-sized models
- **Example**: GPU0: ASR, GPU1: Embedding, GPU2: Reranker, GPU3: ASR (next)

#### Memory Balanced

Allocates based on available GPU memory.

```yaml
gpu:
  gpu_pool:
    allocation_strategy: memory_balanced
```

- **Best for**: Mixed model sizes
- **Behavior**: Assigns to GPU with most free memory
- **Example**: Puts large model on GPU with 20GB free, small model on GPU with 5GB free

#### Least Loaded

Chooses GPU with fewest active models.

```yaml
gpu:
  gpu_pool:
    allocation_strategy: least_loaded
```

- **Best for**: Variable workload
- **Behavior**: Tracks model count per GPU
- **Example**: GPU0: 2 models, GPU1: 1 model → assigns to GPU1

### Manual Device Assignment

Override automatic allocation:

```yaml
model:
  asr_device: cuda:0
  text_emb_device: cuda:1
  audio_emb_device: cuda:2
```

```python
config = EvaluationConfig(
    model_config={
        "asr_device": "cuda:0",
        "text_emb_device": "cuda:1"
    }
)
```

## Memory Management

### Memory Limits

Set maximum memory usage per GPU:

```yaml
gpu:
  max_memory_per_gpu: 0.8  # Use up to 80% of GPU memory
  empty_cache_between_batches: true
```

```python
from evaluator.devices import set_gpu_memory_limit

# Limit each GPU to 20GB
set_gpu_memory_limit(20 * 1024 * 1024 * 1024)

# Or percentage
set_gpu_memory_limit(0.8)  # 80% of available
```

### Cache Management

Clear GPU cache between operations:

```python
import torch

# Manual cache clearing
torch.cuda.empty_cache()

# Automatic clearing in config
config.gpu.empty_cache_between_batches = True
```

### Memory Monitoring

Track GPU memory usage:

```python
from evaluator.devices import GPUMonitor

monitor = GPUMonitor()

# Get current usage
usage = monitor.get_memory_usage()
for gpu_id, (used, total) in usage.items():
    print(f"GPU {gpu_id}: {used/1e9:.2f}GB / {total/1e9:.2f}GB")

# Monitor during evaluation
with monitor.track():
    results = runner.run()
    
peak_memory = monitor.get_peak_memory()
print(f"Peak memory: {peak_memory/1e9:.2f}GB")
```

## Multi-GPU Configurations

### Scenario 1: Two GPUs (Common)

One GPU for ASR, one for embedding:

```yaml
gpu:
  gpu_pool:
    enabled: true
    device_ids: [0, 1]
    allocation_strategy: round_robin

model:
  asr_device: cuda:0
  text_emb_device: cuda:1
```

### Scenario 2: Four GPUs (High Performance)

Distribute all components:

```yaml
gpu:
  gpu_pool:
    enabled: true
    device_ids: [0, 1, 2, 3]
    allocation_strategy: memory_balanced

model:
  asr_device: cuda:0
  text_emb_device: cuda:1
  audio_emb_device: cuda:2
  # GPU 3 available for retrieval/reranking
```

### Scenario 3: Single GPU (Limited Resources)

Optimize for one GPU:

```yaml
gpu:
  gpu_pool:
    enabled: false

model:
  asr_device: cuda:0
  text_emb_device: cuda:0  # Share GPU
  asr_batch_size: 8  # Reduce batch sizes
  text_emb_batch_size: 16
  
cache:
  enabled: true  # Cache embeddings to avoid recomputation
```

### Scenario 4: CPU Fallback

Mixed CPU/GPU usage:

```yaml
model:
  asr_device: cpu  # ASR on CPU
  text_emb_device: cuda:0  # Embedding on GPU
```

## Parallel Execution

### Data Parallelism

Process batches across multiple GPUs:

```python
from evaluator.parallel import DataParallelEvaluator

evaluator = DataParallelEvaluator(
    config=config,
    device_ids=[0, 1, 2, 3]
)

# Automatically splits batches across GPUs
results = evaluator.run()
```

Configuration:

```yaml
parallel:
  enabled: true
  strategy: data_parallel
  device_ids: [0, 1, 2, 3]
  batch_split: true  # Split batches across GPUs
```

### Model Parallelism

Split large models across GPUs:

```python
from evaluator.parallel import ModelParallelWrapper

# For very large models that don't fit on one GPU
large_model = ModelParallelWrapper(
    model_name="very-large-model",
    device_ids=[0, 1]  # Split across GPU 0 and 1
)
```

## Best Practices

### 1. Profile Before Allocating

```python
from evaluator.devices import profile_model_memory

# Check memory requirements
asr_memory = profile_model_memory("whisper-large-v3", "asr")
emb_memory = profile_model_memory("jina_v4", "embedding")

print(f"ASR needs {asr_memory/1e9:.2f}GB")
print(f"Embedding needs {emb_memory/1e9:.2f}GB")
```

### 2. Use Caching with GPU Pooling

Reduce memory pressure:

```yaml
cache:
  enabled: true
  cache_embeddings: true  # Don't recompute embeddings
  cache_asr_results: true  # Don't retranscribe

gpu:
  empty_cache_between_batches: true
```

### 3. Monitor and Adjust

```python
from evaluator.devices import GPUMonitor

monitor = GPUMonitor(alert_threshold=0.9)  # Alert at 90% usage

with monitor.track():
    results = runner.run()
    
if monitor.had_oom():
    print("OOM detected! Reduce batch size or use CPU for some components")
```

### 4. Batch Size Tuning

Find optimal batch size:

```python
from evaluator.benchmarks import find_optimal_batch_size

optimal = find_optimal_batch_size(
    model_type="text_embedding",
    model_name="jina_v4",
    device="cuda:0"
)

print(f"Optimal batch size: {optimal}")
```

### 5. Device Placement Strategy

**Large ASR + Small Embedding**: Put both on same GPU if fits

**Large ASR + Large Embedding**: Use separate GPUs

**Multiple Small Models**: Round-robin across GPUs

**Memory-Constrained**: Put largest model on GPU, rest on CPU

## Troubleshooting

### CUDA Out of Memory

```python
RuntimeError: CUDA out of memory
```

**Solutions**:

1. Reduce batch size:
   ```yaml
   data:
     batch_size: 8  # Down from 32
   ```

2. Use smaller models:
   ```yaml
   model:
     asr_model_name: openai/whisper-base  # Instead of large
   ```

3. Enable gradient checkpointing:
   ```yaml
   model:
     use_gradient_checkpointing: true
   ```

4. Move some components to CPU:
   ```yaml
   model:
     asr_device: cpu
     text_emb_device: cuda:0
   ```

### GPU Not Detected

```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.device_count())  # Number of GPUs
```

If False:
- Check CUDA installation: `nvidia-smi`
- Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### Device Mismatch Errors

```python
RuntimeError: Expected all tensors to be on the same device
```

**Solution**: Ensure consistent device assignment:

```yaml
model:
  asr_device: cuda:0
  text_emb_device: cuda:0  # Same device
```

### Slow Multi-GPU Performance

If multi-GPU is slower than single GPU:

1. Check PCIe bandwidth: `nvidia-smi topo -m`
2. Disable unnecessary device transfers
3. Use data parallelism instead of model parallelism
4. Profile with `nvprof` or `nsys`

## Advanced Features

### Dynamic Device Allocation

Allocate devices based on current load:

```python
from evaluator.devices import DynamicGPUAllocator

allocator = DynamicGPUAllocator()

# Automatically selects best device
device = allocator.allocate_dynamic(
    model_size_gb=5.0,
    priority="speed"  # or "memory"
)
```

### GPU Pool Context Manager

Temporarily use GPU pool:

```python
from evaluator.devices import GPUPool

with GPUPool(device_ids=[0, 1, 2, 3]) as pool:
    device1 = pool.allocate("model1")
    device2 = pool.allocate("model2")
    # Run evaluation
    # Pool automatically cleans up on exit
```

### Mixed Precision

Reduce memory usage with FP16:

```yaml
model:
  use_fp16: true  # Use half precision
```

```python
import torch

with torch.cuda.amp.autocast():
    # Models run in FP16
    results = runner.run()
```

## Next Steps

- Configure [Retrieval Strategies](retrieval.md)
- Learn about [Visualization](visualization.md)
- See [Configuration](configuration.md) for complete GPU options
