"""Comprehensive tests for GPU pool allocation feature."""

import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluator.devices.pool import GPUPool, DeviceUsage
from evaluator.devices.strategy import (
    RoundRobinStrategy,
    MemoryAwareStrategy,
    PackingStrategy,
    ManualStrategy,
    create_strategy,
)
from evaluator.devices.monitor import GPUMonitor, MemoryInfo, GPUInfo


class TestDeviceUsage(unittest.TestCase):
    """Test DeviceUsage dataclass."""
    
    def test_free_memory_calculation(self):
        """Test free memory is calculated correctly."""
        usage = DeviceUsage(device="cuda:0", total_memory_gb=16.0, reserved_memory_gb=4.0)
        self.assertEqual(usage.free_memory_gb, 12.0)
    
    def test_free_memory_never_negative(self):
        """Test free memory cannot go negative."""
        usage = DeviceUsage(device="cuda:0", total_memory_gb=16.0, reserved_memory_gb=20.0)
        self.assertEqual(usage.free_memory_gb, 0.0)
    
    def test_utilization_calculation(self):
        """Test utilization is calculated correctly."""
        usage = DeviceUsage(device="cuda:0", total_memory_gb=16.0, reserved_memory_gb=8.0)
        self.assertEqual(usage.utilization, 0.5)
    
    def test_utilization_capped_at_one(self):
        """Test utilization cannot exceed 1.0."""
        usage = DeviceUsage(device="cuda:0", total_memory_gb=16.0, reserved_memory_gb=20.0)
        self.assertEqual(usage.utilization, 1.0)
    
    def test_utilization_zero_total(self):
        """Test utilization when total memory is zero."""
        usage = DeviceUsage(device="cpu", total_memory_gb=0.0)
        self.assertEqual(usage.utilization, 0.0)


class TestGPUMonitor(unittest.TestCase):
    """Test GPUMonitor class."""
    
    def test_is_available_cpu(self):
        """Test CPU is always available."""
        monitor = GPUMonitor()
        self.assertTrue(monitor.is_available("cpu"))
    
    def test_is_available_invalid_device(self):
        """Test invalid device strings return False."""
        monitor = GPUMonitor()
        self.assertFalse(monitor.is_available("invalid"))
        self.assertFalse(monitor.is_available("mps:0"))
    
    @patch.object(GPUMonitor, 'cuda_available', new_callable=PropertyMock)
    def test_is_available_cuda_when_not_available(self, mock_cuda):
        """Test CUDA device returns False when CUDA unavailable."""
        mock_cuda.return_value = False
        monitor = GPUMonitor()
        self.assertFalse(monitor.is_available("cuda:0"))
    
    @patch.object(GPUMonitor, 'cuda_available', new_callable=PropertyMock)
    def test_get_device_count_no_cuda(self, mock_cuda):
        """Test device count returns 0 when CUDA unavailable."""
        mock_cuda.return_value = False
        monitor = GPUMonitor()
        self.assertEqual(monitor.get_device_count(), 0)
    
    @patch.object(GPUMonitor, 'cuda_available', new_callable=PropertyMock)
    def test_get_all_gpus_no_cuda(self, mock_cuda):
        """Test get_all_gpus returns empty list when CUDA unavailable."""
        mock_cuda.return_value = False
        monitor = GPUMonitor()
        self.assertEqual(monitor.get_all_gpus(), [])


class TestGPUPool(unittest.TestCase):
    """Test GPUPool class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_monitor = MagicMock(spec=GPUMonitor)
        self.mock_monitor.cuda_available = True
        self.mock_monitor.get_device_count.return_value = 2
        
        # Mock memory info for two GPUs with 16GB each
        def get_memory_usage(idx):
            return MemoryInfo(total=16.0, used=0.0, free=16.0)
        self.mock_monitor.get_memory_usage.side_effect = get_memory_usage
    
    def test_init_with_explicit_devices(self):
        """Test pool initialization with explicit device list."""
        pool = GPUPool(["cuda:0", "cuda:1"], monitor=self.mock_monitor)
        self.assertEqual(set(pool.devices), {"cuda:0", "cuda:1"})
    
    def test_init_with_auto(self):
        """Test pool initialization with auto-detect."""
        pool = GPUPool(["auto"], monitor=self.mock_monitor)
        self.assertEqual(set(pool.devices), {"cuda:0", "cuda:1"})
    
    def test_init_auto_fallback_to_cpu(self):
        """Test auto-detect falls back to CPU when no GPUs."""
        self.mock_monitor.get_device_count.return_value = 0
        pool = GPUPool(["auto"], monitor=self.mock_monitor)
        self.assertEqual(pool.devices, ["cpu"])
    
    def test_allocate_basic(self):
        """Test basic allocation."""
        pool = GPUPool(["cuda:0", "cuda:1"], monitor=self.mock_monitor)
        device = pool.allocate("asr", memory_gb=2.0)
        self.assertIn(device, ["cuda:0", "cuda:1"])
    
    def test_allocate_tracks_memory(self):
        """Test allocation tracks memory usage."""
        pool = GPUPool(["cuda:0"], monitor=self.mock_monitor)
        pool.allocate("asr", memory_gb=4.0)
        
        usage = pool.get_usage()
        self.assertEqual(usage["cuda:0"].reserved_memory_gb, 4.0)
        self.assertEqual(usage["cuda:0"].allocations["asr"], 4.0)
    
    def test_release(self):
        """Test releasing an allocation."""
        pool = GPUPool(["cuda:0"], monitor=self.mock_monitor)
        pool.allocate("asr", memory_gb=4.0)
        pool.release("asr")
        
        usage = pool.get_usage()
        self.assertEqual(usage["cuda:0"].reserved_memory_gb, 0.0)
        self.assertNotIn("asr", usage["cuda:0"].allocations)
    
    def test_allocate_cpu_fallback(self):
        """Test fallback to CPU when GPUs are full."""
        # GPU with only 2GB available (after 10% buffer = 1.8GB usable)
        # Clear side_effect and set return_value
        self.mock_monitor.get_memory_usage.side_effect = None
        self.mock_monitor.get_memory_usage.return_value = MemoryInfo(total=2.0, used=0.0, free=2.0)
        pool = GPUPool(["cuda:0"], monitor=self.mock_monitor, allow_cpu_fallback=True, memory_buffer_percent=0.1)
        
        # First allocate all usable GPU memory
        pool.allocate("model1", memory_gb=1.5)
        
        # Request more than remaining - should fall back to CPU
        device = pool.allocate("large_model", memory_gb=1.0)
        self.assertEqual(device, "cpu")
    
    def test_allocate_no_fallback_raises(self):
        """Test allocation raises when no device available and no fallback."""
        # GPU with only 2GB available (after 10% buffer = 1.8GB usable)
        # Clear side_effect and set return_value
        self.mock_monitor.get_memory_usage.side_effect = None
        self.mock_monitor.get_memory_usage.return_value = MemoryInfo(total=2.0, used=0.0, free=2.0)
        pool = GPUPool(["cuda:0"], monitor=self.mock_monitor, allow_cpu_fallback=False, memory_buffer_percent=0.1)
        
        # First allocate all usable GPU memory
        pool.allocate("model1", memory_gb=1.5)
        
        with self.assertRaises(RuntimeError):
            pool.allocate("large_model", memory_gb=1.0)
    
    def test_get_device_for_model(self):
        """Test getting device for an allocated model."""
        pool = GPUPool(["cuda:0", "cuda:1"], monitor=self.mock_monitor)
        pool.allocate("asr", memory_gb=2.0)
        
        device = pool.get_device_for_model("asr")
        self.assertIn(device, ["cuda:0", "cuda:1"])
        
        # Unknown model returns None
        self.assertIsNone(pool.get_device_for_model("unknown"))
    
    def test_gpu_devices_excludes_cpu(self):
        """Test gpu_devices property excludes CPU."""
        pool = GPUPool(["cuda:0", "cpu"], monitor=self.mock_monitor)
        self.assertEqual(pool.gpu_devices, ["cuda:0"])


class TestRoundRobinStrategy(unittest.TestCase):
    """Test RoundRobinStrategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_monitor = MagicMock(spec=GPUMonitor)
        self.mock_monitor.cuda_available = True
        self.mock_monitor.get_device_count.return_value = 2
        self.mock_monitor.get_memory_usage.return_value = MemoryInfo(total=16.0, used=0.0, free=16.0)
    
    def test_distributes_sequentially(self):
        """Test round-robin distributes across GPUs."""
        pool = GPUPool(["cuda:0", "cuda:1"], monitor=self.mock_monitor)
        strategy = RoundRobinStrategy()
        pool.set_strategy(strategy)
        
        devices = []
        for i in range(4):
            device = pool.allocate(f"model_{i}", memory_gb=1.0)
            devices.append(device)
        
        # Should alternate between GPUs
        self.assertEqual(devices[0], "cuda:0")
        self.assertEqual(devices[1], "cuda:1")
        self.assertEqual(devices[2], "cuda:0")
        self.assertEqual(devices[3], "cuda:1")


class TestMemoryAwareStrategy(unittest.TestCase):
    """Test MemoryAwareStrategy."""
    
    def test_allocates_to_most_free(self):
        """Test allocation goes to GPU with most free memory."""
        mock_monitor = MagicMock(spec=GPUMonitor)
        mock_monitor.cuda_available = True
        mock_monitor.get_device_count.return_value = 2
        
        # cuda:0 has 16GB, cuda:1 has 8GB
        def get_memory_usage(idx):
            if idx == 0:
                return MemoryInfo(total=16.0, used=0.0, free=16.0)
            return MemoryInfo(total=8.0, used=0.0, free=8.0)
        mock_monitor.get_memory_usage.side_effect = get_memory_usage
        
        pool = GPUPool(["cuda:0", "cuda:1"], monitor=mock_monitor)
        strategy = MemoryAwareStrategy()
        pool.set_strategy(strategy)
        
        # First allocation should go to cuda:0 (more free memory)
        device = pool.allocate("model", memory_gb=2.0)
        self.assertEqual(device, "cuda:0")


class TestPackingStrategy(unittest.TestCase):
    """Test PackingStrategy."""
    
    def test_packs_onto_utilized_gpu(self):
        """Test packing fills up GPUs before moving to next."""
        mock_monitor = MagicMock(spec=GPUMonitor)
        mock_monitor.cuda_available = True
        mock_monitor.get_device_count.return_value = 2
        mock_monitor.get_memory_usage.return_value = MemoryInfo(total=16.0, used=0.0, free=16.0)
        
        pool = GPUPool(["cuda:0", "cuda:1"], monitor=mock_monitor)
        strategy = PackingStrategy()
        pool.set_strategy(strategy)
        
        # Allocate first model
        device1 = pool.allocate("model_1", memory_gb=4.0)
        
        # Second allocation should go to same GPU (packing)
        device2 = pool.allocate("model_2", memory_gb=4.0)
        
        self.assertEqual(device1, device2)


class TestManualStrategy(unittest.TestCase):
    """Test ManualStrategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_monitor = MagicMock(spec=GPUMonitor)
        self.mock_monitor.cuda_available = True
        self.mock_monitor.get_device_count.return_value = 2
        self.mock_monitor.get_memory_usage.return_value = MemoryInfo(total=16.0, used=0.0, free=16.0)
    
    def test_uses_override(self):
        """Test manual strategy uses specified overrides."""
        pool = GPUPool(["cuda:0", "cuda:1"], monitor=self.mock_monitor)
        strategy = ManualStrategy({"asr": "cuda:1", "text_embedding": "cuda:0"})
        pool.set_strategy(strategy)
        
        asr_device = pool.allocate("asr", memory_gb=2.0)
        text_device = pool.allocate("text_embedding", memory_gb=1.0)
        
        self.assertEqual(asr_device, "cuda:1")
        self.assertEqual(text_device, "cuda:0")
    
    def test_fallback_for_unmapped(self):
        """Test unmapped models use default allocation."""
        pool = GPUPool(["cuda:0", "cuda:1"], monitor=self.mock_monitor)
        strategy = ManualStrategy({"asr": "cuda:1"})
        pool.set_strategy(strategy)
        
        # Unmapped model should use default behavior
        device = pool.allocate("unknown_model", memory_gb=1.0)
        self.assertIn(device, ["cuda:0", "cuda:1"])


class TestCreateStrategy(unittest.TestCase):
    """Test create_strategy factory function."""
    
    def test_create_round_robin(self):
        """Test creating round robin strategy."""
        strategy = create_strategy("round_robin")
        self.assertIsInstance(strategy, RoundRobinStrategy)
    
    def test_create_memory_aware(self):
        """Test creating memory aware strategy."""
        strategy = create_strategy("memory_aware")
        self.assertIsInstance(strategy, MemoryAwareStrategy)
    
    def test_create_packing(self):
        """Test creating packing strategy."""
        strategy = create_strategy("packing")
        self.assertIsInstance(strategy, PackingStrategy)
    
    def test_create_manual(self):
        """Test creating manual strategy with overrides."""
        strategy = create_strategy("manual", overrides={"asr": "cuda:0"})
        self.assertIsInstance(strategy, ManualStrategy)
        self.assertEqual(strategy.overrides, {"asr": "cuda:0"})
    
    def test_create_unknown_raises(self):
        """Test creating unknown strategy raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            create_strategy("unknown")
        self.assertIn("Unknown allocation strategy", str(ctx.exception))


class TestDevicePoolConfig(unittest.TestCase):
    """Test DevicePoolConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        from evaluator.config import DevicePoolConfig
        
        config = DevicePoolConfig()
        self.assertEqual(config.available_devices, ["auto"])
        self.assertEqual(config.allocation_strategy, "memory_aware")
        self.assertEqual(config.memory_buffer_percent, 0.1)
        self.assertTrue(config.allow_cpu_fallback)
        self.assertEqual(config.model_device_overrides, {})
    
    def test_invalid_strategy_raises(self):
        """Test invalid strategy raises ValueError."""
        from evaluator.config import DevicePoolConfig
        
        with self.assertRaises(ValueError):
            DevicePoolConfig(allocation_strategy="invalid")
    
    def test_invalid_buffer_raises(self):
        """Test invalid buffer percentage raises ValueError."""
        from evaluator.config import DevicePoolConfig
        
        with self.assertRaises(ValueError):
            DevicePoolConfig(memory_buffer_percent=1.5)


class TestEvaluationConfigWithDevicePool(unittest.TestCase):
    """Test EvaluationConfig integration with device pool."""
    
    def test_from_dict_with_device_pool(self):
        """Test creating config from dict with device_pool."""
        from evaluator.config import EvaluationConfig
        
        config_dict = {
            "experiment_name": "test",
            "device_pool": {
                "available_devices": ["cuda:0", "cuda:1"],
                "allocation_strategy": "round_robin",
            }
        }
        
        config = EvaluationConfig.from_dict(config_dict, validate=False)
        
        self.assertIsNotNone(config.device_pool)
        self.assertEqual(config.device_pool.available_devices, ["cuda:0", "cuda:1"])
        self.assertEqual(config.device_pool.allocation_strategy, "round_robin")
    
    def test_from_dict_without_device_pool(self):
        """Test creating config from dict without device_pool."""
        from evaluator.config import EvaluationConfig
        
        config_dict = {"experiment_name": "test"}
        config = EvaluationConfig.from_dict(config_dict, validate=False)
        
        self.assertIsNone(config.device_pool)
    
    def test_runtime_dict_includes_device_pool(self):
        """Test runtime config includes device_pool info when configured."""
        from evaluator.config import EvaluationConfig, DevicePoolConfig
        
        config = EvaluationConfig(
            experiment_name="test",
            device_pool=DevicePoolConfig(allocation_strategy="packing")
        )
        
        runtime_dict = config.to_runtime_dict()
        self.assertIn("device_pool", runtime_dict)
        self.assertEqual(runtime_dict["device_pool"]["allocation_strategy"], "packing")


class TestCLIArgsWithDevicePool(unittest.TestCase):
    """Test CLI argument handling for device pool."""
    
    def test_apply_devices_arg(self):
        """Test --devices argument creates device pool config."""
        from evaluator.config import EvaluationConfig
        from evaluator.cli.parser import apply_args_to_config
        import argparse
        
        config = EvaluationConfig()
        args = argparse.Namespace(
            devices="cuda:0,cuda:1",
            allocation_strategy="round_robin",
            # Set all other args to None
            asr_model_type=None, asr_model_name=None, asr_adapter_path=None,
            text_emb_model_type=None, text_emb_model_name=None, text_emb_adapter_path=None,
            audio_emb_model_type=None, audio_emb_model_name=None, audio_emb_adapter_path=None,
            pipeline_mode=None, asr_device=None, text_emb_device=None, audio_emb_device=None,
            dataset_name=None, batch_size=None, trace_limit=None, questions_path=None,
            corpus_path=None, skip_dataset_validation=False, db_type=None, k=None,
            retrieval_mode=None, hybrid_dense_weight=None, reranker_mode=None,
            reranker_top_k=None, reranker_weight=None, no_cache=False, cache_dir=None,
            log_level=None, experiment_name=None, output_dir=None, no_checkpoint=False,
            checkpoint_interval=None, judge_enabled=False, judge_model=None,
            judge_api_base=None, judge_api_key_env=None, judge_max_cases=None,
            judge_timeout_s=None, judge_temperature=None,
        )
        
        apply_args_to_config(args, config)
        
        self.assertIsNotNone(config.device_pool)
        self.assertEqual(config.device_pool.available_devices, ["cuda:0", "cuda:1"])
        self.assertEqual(config.device_pool.allocation_strategy, "round_robin")


class TestPipelineBundleWithDevicePool(unittest.TestCase):
    """Test PipelineBundle with device_pool attribute."""
    
    def test_bundle_has_device_pool(self):
        """Test PipelineBundle can hold device pool reference."""
        from evaluator.pipeline.types import PipelineBundle
        
        mock_pool = MagicMock()
        bundle = PipelineBundle(
            mode="asr_text_retrieval",
            device_pool=mock_pool
        )
        
        self.assertIs(bundle.device_pool, mock_pool)
    
    def test_bundle_device_pool_optional(self):
        """Test device_pool is optional and defaults to None."""
        from evaluator.pipeline.types import PipelineBundle
        
        bundle = PipelineBundle(mode="asr_only")
        self.assertIsNone(bundle.device_pool)


if __name__ == "__main__":
    unittest.main()
