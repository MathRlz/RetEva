"""Parallel evaluation across multiple GPUs.

This module provides the ParallelEvaluator class for distributing
evaluation workloads across multiple GPU devices.
"""

from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from ..logging_config import get_logger
from ..config import get_available_gpu_count
from .batch_distributor import BatchDistributor, Worker, WorkerResult, aggregate_results

logger = get_logger(__name__)


def _worker_process(
    worker_id: int,
    device: str,
    indices: List[int],
    dataset_path: str,
    dataset_class: str,
    dataset_kwargs: Dict[str, Any],
    config_dict: Dict[str, Any],
    k: int,
    batch_size: int,
    trace_limit: int,
) -> Dict[str, Any]:
    """Worker process function for multiprocessing.
    
    This function runs in a separate process and handles:
    1. Loading the dataset subset
    2. Creating pipelines with correct device
    3. Running evaluation
    
    Args:
        worker_id: Unique identifier for this worker.
        device: Device string (e.g., "cuda:0").
        indices: Dataset indices to process.
        dataset_path: Path to dataset.
        dataset_class: Class name of the dataset.
        dataset_kwargs: Kwargs for dataset instantiation.
        config_dict: Serialized config dictionary.
        k: Number of retrieval results.
        batch_size: Batch size for processing.
        trace_limit: Query trace limit.
        
    Returns:
        Dictionary with worker results or error info.
    """
    import torch
    
    # Set CUDA device for this process
    if device.startswith("cuda"):
        device_idx = int(device.split(":")[1]) if ":" in device else 0
        torch.cuda.set_device(device_idx)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_idx)
    
    try:
        from torch.utils.data import Subset
        from ..config import EvaluationConfig
        from ..pipeline import create_pipeline_from_config
        from ..storage.cache import CacheManager
        from ..evaluation.phased import evaluate_phased
        
        # Reconstruct config with worker-specific device
        config = EvaluationConfig.from_dict(config_dict)
        config.model.asr_device = device
        config.model.text_emb_device = device
        config.model.audio_emb_device = device
        
        # Create cache manager for this worker
        cache_config = config.cache
        cache_manager = CacheManager(
            cache_dir=cache_config.cache_dir,
            enabled=cache_config.enabled,
        ) if cache_config.enabled else None
        
        # Create dataset
        from ..datasets import get_dataset_class
        dataset_cls = get_dataset_class(dataset_class)
        full_dataset = dataset_cls(**dataset_kwargs)
        
        # Create subset for this worker
        subset = Subset(full_dataset, indices)
        
        # Create pipelines
        bundle = create_pipeline_from_config(config, cache_manager)
        
        # Run evaluation
        results = evaluate_phased(
            dataset=subset,
            retrieval_pipeline=bundle.retrieval_pipeline,
            asr_pipeline=bundle.asr_pipeline,
            text_embedding_pipeline=bundle.text_embedding_pipeline,
            audio_embedding_pipeline=bundle.audio_embedding_pipeline,
            cache_manager=cache_manager,
            k=k,
            batch_size=batch_size,
            trace_limit=trace_limit,
            num_workers=0,  # No nested parallelism
        )
        
        return {
            "worker_id": worker_id,
            "device": device,
            "results": results,
            "num_samples": len(indices),
            "error": None,
        }
        
    except Exception as e:
        import traceback
        return {
            "worker_id": worker_id,
            "device": device,
            "results": None,
            "num_samples": 0,
            "error": f"{str(e)}\n{traceback.format_exc()}",
        }


class ParallelEvaluator:
    """Parallel evaluator for multi-GPU evaluation.
    
    Distributes evaluation workloads across multiple GPUs using
    Python's multiprocessing. Each GPU worker processes a subset
    of the dataset independently, and results are aggregated.
    
    Attributes:
        config: EvaluationConfig instance.
        num_workers: Number of GPU workers to use.
        devices: List of device strings for each worker.
        
    Examples:
        Basic usage::
        
            >>> from evaluator.parallel import ParallelEvaluator
            >>> evaluator = ParallelEvaluator(config, num_workers=2)
            >>> results = evaluator.evaluate_parallel(dataset)
        
        Auto-detect GPU count::
        
            >>> evaluator = ParallelEvaluator(config)  # Uses all available GPUs
            >>> results = evaluator.evaluate_parallel(dataset)
    """
    
    def __init__(
        self,
        config: Any,
        num_workers: Optional[int] = None,
    ):
        """Initialize parallel evaluator.
        
        Args:
            config: EvaluationConfig instance.
            num_workers: Number of GPU workers. If None, auto-detect GPU count.
        """
        self.config = config
        
        # Auto-detect GPU count if not specified
        if num_workers is None or num_workers == 0:
            self.num_workers = get_available_gpu_count()
            if self.num_workers == 0:
                self.num_workers = 1  # Fall back to single CPU
                self.devices = ["cpu"]
            else:
                self.devices = [f"cuda:{i}" for i in range(self.num_workers)]
        else:
            self.num_workers = num_workers
            available = get_available_gpu_count()
            if available == 0:
                self.devices = ["cpu"] * num_workers
            else:
                self.devices = [f"cuda:{i % available}" for i in range(num_workers)]
        
        self.distributor = BatchDistributor()
        
        logger.info(f"ParallelEvaluator initialized with {self.num_workers} workers")
        logger.info(f"Devices: {self.devices}")
    
    def evaluate_parallel(
        self,
        dataset: Any,
        k: int = 10,
        batch_size: int = 32,
        trace_limit: int = 0,
    ) -> Dict[str, Any]:
        """Run parallel evaluation across multiple GPUs.
        
        Distributes the dataset across workers, runs evaluation on each,
        and aggregates the results.
        
        Args:
            dataset: QueryDataset instance to evaluate.
            k: Number of retrieval results per query. Default: 10.
            batch_size: Batch size per worker. Default: 32.
            trace_limit: Number of query traces to keep. Default: 0.
            
        Returns:
            Aggregated evaluation results dictionary.
            
        Notes:
            - Each worker loads models independently
            - Results are aggregated using weighted averages
            - Failed workers are reported in results
        """
        n_samples = len(dataset)
        
        if n_samples == 0:
            logger.warning("Empty dataset provided")
            return {"total_samples": 0, "parallel": True}
        
        logger.info(f"Starting parallel evaluation: {n_samples} samples across {self.num_workers} workers")
        
        # Distribute indices across workers
        index_batches = self.distributor.distribute_indices(n_samples, self.num_workers)
        
        for i, batch in enumerate(index_batches):
            logger.info(f"Worker {i} ({self.devices[i]}): {len(batch)} samples")
        
        # Run workers
        worker_results = self._run_workers(
            dataset=dataset,
            index_batches=index_batches,
            k=k,
            batch_size=batch_size,
            trace_limit=trace_limit,
        )
        
        # Aggregate results
        aggregated = aggregate_results(worker_results)
        
        logger.info(f"Parallel evaluation complete: {aggregated.get('total_samples', 0)} samples processed")
        
        return aggregated
    
    def _run_workers(
        self,
        dataset: Any,
        index_batches: List[List[int]],
        k: int,
        batch_size: int,
        trace_limit: int,
    ) -> List[WorkerResult]:
        """Run evaluation workers.
        
        Uses concurrent.futures for process-based parallelism.
        
        Args:
            dataset: The dataset to evaluate.
            index_batches: List of index batches, one per worker.
            k: Number of retrieval results.
            batch_size: Batch size for processing.
            trace_limit: Query trace limit.
            
        Returns:
            List of WorkerResult objects.
        """
        worker_results: List[WorkerResult] = []
        
        # For single worker, run directly without multiprocessing overhead
        if self.num_workers == 1:
            result = self._run_single_worker(
                worker_id=0,
                device=self.devices[0],
                indices=index_batches[0],
                dataset=dataset,
                k=k,
                batch_size=batch_size,
                trace_limit=trace_limit,
            )
            worker_results.append(result)
            return worker_results
        
        # Use spawn method for CUDA compatibility
        ctx = mp.get_context("spawn")
        
        # Serialize config and dataset info for workers
        config_dict = {
            "runtime": self.config.to_runtime_dict(),
            "experiment": self.config.to_experiment_dict(),
        }
        dataset_info = self._serialize_dataset_info(dataset)
        
        with ProcessPoolExecutor(
            max_workers=self.num_workers,
            mp_context=ctx,
        ) as executor:
            futures = {}
            
            for i, (device, indices) in enumerate(zip(self.devices, index_batches)):
                if not indices:
                    continue
                
                future = executor.submit(
                    _worker_process,
                    worker_id=i,
                    device=device,
                    indices=indices,
                    dataset_path=dataset_info["path"],
                    dataset_class=dataset_info["class"],
                    dataset_kwargs=dataset_info["kwargs"],
                    config_dict=config_dict,
                    k=k,
                    batch_size=batch_size,
                    trace_limit=trace_limit,
                )
                futures[future] = i
            
            for future in as_completed(futures):
                worker_id = futures[future]
                try:
                    result_dict = future.result()
                    worker_results.append(WorkerResult(
                        worker_id=result_dict["worker_id"],
                        device=result_dict["device"],
                        results=result_dict["results"],
                        num_samples=result_dict["num_samples"],
                        error=result_dict["error"],
                    ))
                except Exception as e:
                    worker_results.append(WorkerResult(
                        worker_id=worker_id,
                        device=self.devices[worker_id],
                        results=None,
                        num_samples=0,
                        error=str(e),
                    ))
        
        return worker_results
    
    def _run_single_worker(
        self,
        worker_id: int,
        device: str,
        indices: List[int],
        dataset: Any,
        k: int,
        batch_size: int,
        trace_limit: int,
    ) -> WorkerResult:
        """Run evaluation on a single worker without multiprocessing.
        
        More efficient for single-GPU or debugging scenarios.
        """
        from torch.utils.data import Subset
        from ..pipeline import create_pipeline_from_config
        from ..storage.cache import CacheManager
        from ..evaluation.phased import evaluate_phased
        
        try:
            # Update config devices
            self.config.model.asr_device = device
            self.config.model.text_emb_device = device
            self.config.model.audio_emb_device = device
            
            # Create cache manager
            cache_config = self.config.cache
            cache_manager = CacheManager(
                cache_dir=cache_config.cache_dir,
                enabled=cache_config.enabled,
            ) if cache_config.enabled else None
            
            # Create subset
            subset = Subset(dataset, indices)
            
            # Create pipelines
            bundle = create_pipeline_from_config(self.config, cache_manager)
            
            # Run evaluation
            results = evaluate_phased(
                dataset=subset,
                retrieval_pipeline=bundle.retrieval_pipeline,
                asr_pipeline=bundle.asr_pipeline,
                text_embedding_pipeline=bundle.text_embedding_pipeline,
                audio_embedding_pipeline=bundle.audio_embedding_pipeline,
                cache_manager=cache_manager,
                k=k,
                batch_size=batch_size,
                trace_limit=trace_limit,
                num_workers=0,
            )
            
            return WorkerResult(
                worker_id=worker_id,
                device=device,
                results=results,
                num_samples=len(indices),
                error=None,
            )
            
        except Exception as e:
            import traceback
            return WorkerResult(
                worker_id=worker_id,
                device=device,
                results=None,
                num_samples=0,
                error=f"{str(e)}\n{traceback.format_exc()}",
            )
    
    def _serialize_dataset_info(self, dataset: Any) -> Dict[str, Any]:
        """Extract serializable dataset information for worker processes.
        
        Args:
            dataset: The dataset instance.
            
        Returns:
            Dictionary with dataset class and construction args.
        """
        # Get the class name
        class_name = dataset.__class__.__name__
        
        # Try to extract constructor arguments
        kwargs = {}
        
        if hasattr(dataset, "questions_path"):
            kwargs["questions_path"] = str(dataset.questions_path)
        if hasattr(dataset, "corpus_path"):
            kwargs["corpus_path"] = str(dataset.corpus_path)
        if hasattr(dataset, "corpus"):
            # For AdmedQueryDataset, we'd need to handle this differently
            pass
        
        return {
            "class": class_name,
            "path": getattr(dataset, "data_path", ""),
            "kwargs": kwargs,
        }
