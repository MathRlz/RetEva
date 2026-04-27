"""Tests for configuration presets and grid search."""

import pytest
from evaluator.config import EvaluationConfig, VectorDBConfig, EmbeddingFusionConfig
from evaluator.config.feature_presets import (
    get_preset,
    list_presets,
    apply_preset,
    PRESET_REGISTRY,
)
from evaluator.analysis.grid_search import (
    GridSearch,
    create_fusion_grid,
    create_retrieval_grid,
    create_advanced_rag_grid,
    run_grid_search,
    analyze_grid_results,
)


class TestPresets:
    """Tests for configuration presets."""
    
    def test_get_preset_valid(self):
        """Test getting valid preset."""
        preset = get_preset("medical_optimized")
        
        assert "embedding_fusion" in preset or "experiment_name" in preset
    
    def test_get_preset_invalid(self):
        """Test error on invalid preset."""
        with pytest.raises(KeyError, match="Unknown preset"):
            get_preset("nonexistent_preset")
    
    def test_list_presets(self):
        """Test listing all presets."""
        presets = list_presets()
        
        assert isinstance(presets, dict)
        assert len(presets) > 0
        assert "medical_optimized" in presets
    
    def test_preset_registry_complete(self):
        """Test all presets in registry have descriptions."""
        descriptions = list_presets()
        
        for preset_name in PRESET_REGISTRY.keys():
            assert preset_name in descriptions
    
    def test_apply_preset_fusion(self):
        """Test applying fusion preset."""
        config = EvaluationConfig()
        
        config = apply_preset(config, "fusion_balanced")
        
        assert config.embedding_fusion.enabled is True
        assert config.embedding_fusion.audio_weight == 0.5
    
    def test_apply_preset_query(self):
        """Test applying query optimization preset."""
        config = EvaluationConfig()
        
        config = apply_preset(config, "query_rewrite")
        
        assert config.query_optimization.enabled is True
        assert config.query_optimization.method == "rewrite"
    
    def test_apply_preset_hybrid(self):
        """Test applying hybrid retrieval preset."""
        config = EvaluationConfig()
        
        config = apply_preset(config, "hybrid_rrf")
        
        assert config.vector_db.retrieval_mode == "hybrid"
        assert config.vector_db.hybrid_fusion_method == "rrf"
    
    def test_full_stack_presets_valid(self):
        """Test full stack presets are valid."""
        full_stack_presets = [
            "medical_optimized",
            "full_rag_advanced",
            "fast_baseline",
            "quality_focused",
        ]
        
        for preset_name in full_stack_presets:
            preset = get_preset(preset_name)
            # Should be able to create config
            config = EvaluationConfig(**preset)
            assert config.experiment_name is not None


class TestGridSearch:
    """Tests for grid search functionality."""
    
    def test_grid_initialization(self):
        """Test grid search initialization."""
        base_config = EvaluationConfig()
        grid = GridSearch(base_config)
        
        assert grid.base_config == base_config
        assert len(grid.param_grid) == 0
    
    def test_add_param(self):
        """Test adding parameters to grid."""
        base_config = EvaluationConfig()
        grid = GridSearch(base_config)
        
        grid.add_param("vector_db.k", [5, 10, 20])
        
        assert "vector_db.k" in grid.param_grid
        assert len(grid.param_grid["vector_db.k"]) == 3
    
    def test_add_param_chaining(self):
        """Test method chaining for add_param."""
        base_config = EvaluationConfig()
        grid = GridSearch(base_config)
        
        grid.add_param("vector_db.k", [5, 10]) \
            .add_param("vector_db.hybrid_dense_weight", [0.5, 0.7])
        
        assert len(grid.param_grid) == 2
    
    def test_get_size(self):
        """Test computing grid size."""
        base_config = EvaluationConfig()
        grid = GridSearch(base_config)
        
        grid.add_param("vector_db.k", [5, 10, 20])  # 3 values
        grid.add_param("vector_db.hybrid_dense_weight", [0.5, 0.7])  # 2 values
        
        assert grid.get_size() == 6  # 3 * 2
    
    def test_get_size_empty(self):
        """Test grid size when no params."""
        base_config = EvaluationConfig()
        grid = GridSearch(base_config)
        
        assert grid.get_size() == 1
    
    def test_generate_configs(self):
        """Test generating configurations."""
        base_config = EvaluationConfig()
        grid = GridSearch(base_config)
        
        grid.add_param("vector_db.k", [5, 10])
        
        configs = list(grid.generate_configs())
        
        assert len(configs) == 2
        assert configs[0][1].vector_db.k == 5
        assert configs[1][1].vector_db.k == 10
    
    def test_generate_configs_multiple_params(self):
        """Test generating configs with multiple parameters."""
        base_config = EvaluationConfig()
        grid = GridSearch(base_config)
        
        grid.add_param("vector_db.k", [5, 10])
        grid.add_param("vector_db.hybrid_dense_weight", [0.3, 0.7])
        
        configs = list(grid.generate_configs())
        
        assert len(configs) == 4  # 2 * 2
        
        # Check combinations exist
        param_dicts = [params for params, _ in configs]
        assert {"vector_db.k": 5, "vector_db.hybrid_dense_weight": 0.3} in param_dicts
        assert {"vector_db.k": 10, "vector_db.hybrid_dense_weight": 0.7} in param_dicts
    
    def test_summary(self):
        """Test grid summary."""
        base_config = EvaluationConfig()
        grid = GridSearch(base_config)
        
        grid.add_param("vector_db.k", [5, 10, 20])
        grid.add_param("vector_db.hybrid_dense_weight", [0.5, 0.7])
        
        summary = grid.summary()
        
        assert summary["total_configs"] == 6
        assert "parameters" in summary
        assert summary["parameters"]["vector_db.k"] == 3


class TestGridCreators:
    """Tests for grid creator functions."""
    
    def test_create_fusion_grid(self):
        """Test fusion grid creator."""
        base_config = EvaluationConfig()
        grid = create_fusion_grid(base_config)
        
        assert grid.get_size() > 1
        assert any("fusion" in key for key in grid.param_grid.keys())
    
    def test_create_retrieval_grid(self):
        """Test retrieval grid creator."""
        base_config = EvaluationConfig()
        grid = create_retrieval_grid(base_config)
        
        assert grid.get_size() > 1
        assert "vector_db.k" in grid.param_grid
    
    def test_create_advanced_rag_grid(self):
        """Test advanced RAG grid creator."""
        base_config = EvaluationConfig()
        grid = create_advanced_rag_grid(base_config)
        
        assert grid.get_size() > 1
        assert any("mmr" in key or "expansion" in key or "feedback" in key 
                   for key in grid.param_grid.keys())


class TestGridSearchRun:
    """Tests for running grid search."""
    
    def test_run_grid_search_basic(self):
        """Test basic grid search run."""
        base_config = EvaluationConfig()
        grid = GridSearch(base_config)
        grid.add_param("vector_db.k", [5, 10])
        
        # Mock evaluation function
        def mock_eval(config):
            return {"recall@5": 0.8, "mrr": 0.7}
        
        results = run_grid_search(grid, mock_eval)
        
        assert len(results) == 2
        assert all(r["success"] for r in results)
    
    def test_run_grid_search_with_errors(self):
        """Test grid search handling errors."""
        base_config = EvaluationConfig()
        grid = GridSearch(base_config)
        grid.add_param("vector_db.k", [5, 10])
        
        # Mock evaluation that fails
        def failing_eval(config):
            if config.vector_db.k == 10:
                raise ValueError("Test error")
            return {"recall@5": 0.8}
        
        results = run_grid_search(grid, failing_eval)
        
        assert len(results) == 2
        assert results[0]["success"] is True
        assert results[1]["success"] is False
        assert "error" in results[1]


class TestResultsAnalysis:
    """Tests for analyzing grid search results."""
    
    def test_analyze_results_basic(self):
        """Test basic results analysis."""
        results = [
            {
                "config_id": 1,
                "params": {"k": 5},
                "metrics": {"recall@5": 0.7},
                "success": True
            },
            {
                "config_id": 2,
                "params": {"k": 10},
                "metrics": {"recall@5": 0.9},
                "success": True
            },
        ]
        
        analysis = analyze_grid_results(results, metric_name="recall@5")
        
        assert analysis["best_config"]["params"] == {"k": 10}
        assert analysis["best_config"]["recall@5"] == 0.9
        assert len(analysis["top_configs"]) == 2
    
    def test_analyze_results_with_failures(self):
        """Test analysis with failed runs."""
        results = [
            {
                "config_id": 1,
                "params": {"k": 5},
                "metrics": {"recall@5": 0.8},
                "success": True
            },
            {
                "config_id": 2,
                "params": {"k": 10},
                "error": "Test error",
                "success": False
            },
        ]
        
        analysis = analyze_grid_results(results, metric_name="recall@5")
        
        assert analysis["summary"]["successful_runs"] == 1
        assert analysis["summary"]["failed_runs"] == 1
        assert analysis["best_config"]["params"] == {"k": 5}
    
    def test_analyze_results_all_failures(self):
        """Test analysis when all runs failed."""
        results = [
            {
                "config_id": 1,
                "error": "Error 1",
                "success": False
            },
            {
                "config_id": 2,
                "error": "Error 2",
                "success": False
            },
        ]
        
        analysis = analyze_grid_results(results, metric_name="recall@5")
        
        assert analysis["best_config"] is None
        assert len(analysis["top_configs"]) == 0
        assert analysis["summary"]["successful_runs"] == 0
    
    def test_analyze_results_top_k(self):
        """Test top-k selection in analysis."""
        results = [
            {"params": {}, "metrics": {"recall@5": 0.5}, "success": True},
            {"params": {}, "metrics": {"recall@5": 0.7}, "success": True},
            {"params": {}, "metrics": {"recall@5": 0.9}, "success": True},
            {"params": {}, "metrics": {"recall@5": 0.6}, "success": True},
        ]
        
        analysis = analyze_grid_results(results, metric_name="recall@5", top_k=2)
        
        assert len(analysis["top_configs"]) == 2
        assert analysis["top_configs"][0]["recall@5"] == 0.9
        assert analysis["top_configs"][1]["recall@5"] == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
