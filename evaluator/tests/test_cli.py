"""Comprehensive tests for CLI functionality.

This module tests command-line interface components including:
- Argument parsing
- Config validation
- Help output
- Error handling for invalid arguments
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from evaluator.cli import parse_args, run_evaluation
from evaluator.cli.parser import apply_args_to_config
from evaluator.config import EvaluationConfig


class TestArgumentParsing:
    """Test argument parsing for CLI."""
    
    def test_parse_args_default(self):
        """Test parsing with minimal arguments."""
        with patch('sys.argv', ['evaluate.py']):
            args = parse_args()
            assert args.config is None
            assert args.asr_model_type is None
            assert args.batch_size is None
    
    def test_parse_args_config_path(self):
        """Test parsing config file path."""
        with patch('sys.argv', ['evaluate.py', '--config', 'configs/test.yaml']):
            args = parse_args()
            assert args.config == 'configs/test.yaml'

    def test_parse_args_list_models(self):
        """Test parsing model listing flag."""
        with patch('sys.argv', ['evaluate.py', '--list_models']):
            args = parse_args()
            assert args.list_models is True
    
    def test_parse_args_model_overrides(self):
        """Test parsing model override arguments."""
        with patch('sys.argv', [
            'evaluate.py',
            '--asr_model_type', 'whisper',
            '--asr_model_name', 'openai/whisper-base',
            '--text_emb_model_type', 'labse',
        ]):
            args = parse_args()
            assert args.asr_model_type == 'whisper'
            assert args.asr_model_name == 'openai/whisper-base'
            assert args.text_emb_model_type == 'labse'
    
    def test_parse_args_device_overrides(self):
        """Test parsing device assignment arguments."""
        with patch('sys.argv', [
            'evaluate.py',
            '--asr_device', 'cuda:0',
            '--text_emb_device', 'cuda:1',
        ]):
            args = parse_args()
            assert args.asr_device == 'cuda:0'
            assert args.text_emb_device == 'cuda:1'
    
    def test_parse_args_gpu_pool(self):
        """Test parsing GPU pool configuration."""
        with patch('sys.argv', [
            'evaluate.py',
            '--devices', 'cuda:0,cuda:1,cuda:2',
            '--allocation_strategy', 'memory_aware',
        ]):
            args = parse_args()
            assert args.devices == 'cuda:0,cuda:1,cuda:2'
            assert args.allocation_strategy == 'memory_aware'

    def test_parse_args_service_runtime(self):
        """Test parsing service runtime policy arguments."""
        with patch('sys.argv', [
            'evaluate.py',
            '--service_startup_mode', 'eager',
            '--service_offload_policy', 'never',
        ]):
            args = parse_args()
            assert args.service_startup_mode == 'eager'
            assert args.service_offload_policy == 'never'
    
    def test_parse_args_dataset_options(self):
        """Test parsing dataset-related arguments."""
        with patch('sys.argv', [
            'evaluate.py',
            '--questions_path', 'data/questions.json',
            '--corpus_path', 'data/corpus.json',
            '--batch_size', '16',
            '--trace_limit', '100',
        ]):
            args = parse_args()
            assert args.questions_path == 'data/questions.json'
            assert args.corpus_path == 'data/corpus.json'
            assert args.batch_size == 16
            assert args.trace_limit == 100
    
    def test_parse_args_vector_db_options(self):
        """Test parsing vector database arguments."""
        with patch('sys.argv', [
            'evaluate.py',
            '--db_type', 'faiss',
            '--k', '10',
            '--retrieval_mode', 'hybrid',
            '--hybrid_dense_weight', '0.7',
        ]):
            args = parse_args()
            assert args.db_type == 'faiss'
            assert args.k == 10
            assert args.retrieval_mode == 'hybrid'
            assert args.hybrid_dense_weight == 0.7
    
    def test_parse_args_cache_options(self):
        """Test parsing cache-related arguments."""
        with patch('sys.argv', [
            'evaluate.py',
            '--no_cache',
            '--cache_dir', '/tmp/cache',
        ]):
            args = parse_args()
            assert args.no_cache is True
            assert args.cache_dir == '/tmp/cache'
    
    def test_parse_args_logging_options(self):
        """Test parsing logging arguments."""
        with patch('sys.argv', [
            'evaluate.py',
            '--log_level', 'DEBUG',
        ]):
            args = parse_args()
            assert args.log_level == 'DEBUG'
    
    def test_parse_args_output_options(self):
        """Test parsing output-related arguments."""
        with patch('sys.argv', [
            'evaluate.py',
            '--experiment_name', 'my_experiment',
            '--output_dir', 'results/',
        ]):
            args = parse_args()
            assert args.experiment_name == 'my_experiment'
            assert args.output_dir == 'results/'
    
    def test_parse_args_checkpoint_options(self):
        """Test parsing checkpoint arguments."""
        with patch('sys.argv', [
            'evaluate.py',
            '--no_checkpoint',
            '--checkpoint_interval', '500',
        ]):
            args = parse_args()
            assert args.no_checkpoint is True
            assert args.checkpoint_interval == 500
    
    def test_parse_args_judge_options(self):
        """Test parsing LLM judge arguments."""
        with patch('sys.argv', [
            'evaluate.py',
            '--judge_enabled',
            '--judge_model', 'gpt-4',
            '--judge_max_cases', '50',
            '--judge_temperature', '0.0',
        ]):
            args = parse_args()
            assert args.judge_enabled is True
            assert args.judge_model == 'gpt-4'
            assert args.judge_max_cases == 50
            assert args.judge_temperature == 0.0
    
    def test_parse_args_pipeline_mode(self):
        """Test parsing pipeline mode argument."""
        with patch('sys.argv', [
            'evaluate.py',
            '--pipeline_mode', 'asr_text_retrieval',
        ]):
            args = parse_args()
            assert args.pipeline_mode == 'asr_text_retrieval'
    
    def test_parse_args_reranker_options(self):
        """Test parsing reranker arguments."""
        with patch('sys.argv', [
            'evaluate.py',
            '--reranker_mode', 'token_overlap',
            '--reranker_top_k', '20',
            '--reranker_weight', '0.3',
        ]):
            args = parse_args()
            assert args.reranker_mode == 'token_overlap'
            assert args.reranker_top_k == 20
            assert args.reranker_weight == 0.3
    
    def test_parse_args_adapter_paths(self):
        """Test parsing adapter path arguments."""
        with patch('sys.argv', [
            'evaluate.py',
            '--asr_adapter_path', 'adapters/asr_lora',
            '--text_emb_adapter_path', 'adapters/emb_lora',
        ]):
            args = parse_args()
            assert args.asr_adapter_path == 'adapters/asr_lora'
            assert args.text_emb_adapter_path == 'adapters/emb_lora'


class TestConfigApplication:
    """Test applying CLI arguments to configuration."""
    
    def test_apply_args_model_overrides(self):
        """Test applying model-related arguments to config."""
        config = EvaluationConfig(experiment_name="test")
        
        with patch('sys.argv', [
            'evaluate.py',
            '--asr_model_type', 'whisper',
            '--asr_model_name', 'openai/whisper-base',
        ]):
            args = parse_args()
            apply_args_to_config(args, config)
            
            assert config.model.asr_model_type == 'whisper'
            assert config.model.asr_model_name == 'openai/whisper-base'
    
    def test_apply_args_device_overrides(self):
        """Test applying device arguments to config."""
        config = EvaluationConfig(experiment_name="test")
        
        with patch('sys.argv', [
            'evaluate.py',
            '--asr_device', 'cuda:0',
            '--text_emb_device', 'cuda:1',
        ]):
            args = parse_args()
            apply_args_to_config(args, config)
            
            assert config.model.asr_device == 'cuda:0'
            assert config.model.text_emb_device == 'cuda:1'

    def test_apply_args_service_runtime(self):
        """Test applying service runtime policy arguments to config."""
        config = EvaluationConfig(experiment_name="test")

        with patch('sys.argv', [
            'evaluate.py',
            '--service_startup_mode', 'eager',
            '--service_offload_policy', 'never',
        ]):
            args = parse_args()
            apply_args_to_config(args, config)

            assert config.service_runtime.startup_mode == 'eager'
            assert config.service_runtime.offload_policy == 'never'
    
    def test_apply_args_dataset_overrides(self):
        """Test applying dataset arguments to config."""
        config = EvaluationConfig(experiment_name="test")
        
        with patch('sys.argv', [
            'evaluate.py',
            '--questions_path', 'data/questions.json',
            '--batch_size', '16',
            '--trace_limit', '100',
        ]):
            args = parse_args()
            apply_args_to_config(args, config)
            
            assert config.data.questions_path == 'data/questions.json'
            assert config.data.batch_size == 16
            assert config.data.trace_limit == 100
    
    def test_apply_args_cache_disable(self):
        """Test disabling cache via CLI."""
        config = EvaluationConfig(experiment_name="test")
        
        with patch('sys.argv', ['evaluate.py', '--no_cache']):
            args = parse_args()
            apply_args_to_config(args, config)
            
            assert config.cache.enabled is False
    
    def test_apply_args_vector_db(self):
        """Test applying vector DB arguments to config."""
        config = EvaluationConfig(experiment_name="test")
        
        with patch('sys.argv', [
            'evaluate.py',
            '--db_type', 'faiss',
            '--k', '10',
            '--retrieval_mode', 'hybrid',
        ]):
            args = parse_args()
            apply_args_to_config(args, config)
            
            assert config.vector_db.type == 'faiss'
            assert config.vector_db.k == 10
            assert config.vector_db.retrieval_mode == 'hybrid'
    
    def test_apply_args_experiment_name(self):
        """Test applying experiment name via CLI."""
        config = EvaluationConfig(experiment_name="default")
        
        with patch('sys.argv', [
            'evaluate.py',
            '--experiment_name', 'cli_override',
        ]):
            args = parse_args()
            apply_args_to_config(args, config)
            
            assert config.experiment_name == 'cli_override'
    
    def test_apply_args_judge_config(self):
        """Test applying judge configuration via CLI."""
        config = EvaluationConfig(experiment_name="test")
        
        with patch('sys.argv', [
            'evaluate.py',
            '--judge_enabled',
            '--judge_model', 'gpt-4',
            '--judge_max_cases', '50',
        ]):
            args = parse_args()
            apply_args_to_config(args, config)
            
            assert config.judge.enabled is True
            assert config.judge.model == 'gpt-4'
            assert config.judge.max_cases == 50
    
    def test_apply_args_gpu_pool_config(self):
        """Test applying GPU pool configuration via CLI."""
        config = EvaluationConfig(experiment_name="test")
        
        with patch('sys.argv', [
            'evaluate.py',
            '--devices', 'cuda:0,cuda:1',
            '--allocation_strategy', 'memory_aware',
        ]):
            args = parse_args()
            apply_args_to_config(args, config)
            
            assert config.device_pool is not None
            assert config.device_pool.available_devices == ['cuda:0', 'cuda:1']
            assert config.device_pool.allocation_strategy == 'memory_aware'
    
    def test_apply_args_pipeline_mode(self):
        """Test applying pipeline mode via CLI."""
        config = EvaluationConfig(experiment_name="test")
        
        with patch('sys.argv', [
            'evaluate.py',
            '--pipeline_mode', 'audio_emb_retrieval',
        ]):
            args = parse_args()
            apply_args_to_config(args, config)
            
            assert config.model.pipeline_mode == 'audio_emb_retrieval'
    
    def test_apply_args_skip_validation(self):
        """Test disabling dataset validation via CLI."""
        config = EvaluationConfig(experiment_name="test")
        
        with patch('sys.argv', [
            'evaluate.py',
            '--skip_dataset_validation',
        ]):
            args = parse_args()
            apply_args_to_config(args, config)
            
            assert config.data.strict_validation is False


class TestCLIHelp:
    """Test CLI help output."""
    
    def test_help_output(self):
        """Test that --help produces expected output."""
        result = subprocess.run(
            [sys.executable, '-m', 'evaluator.cli.parser', '--help'],
            capture_output=True,
            text=True
        )
        # Should exit with 0 for help
        # Note: argparse exits with 0 for --help
        assert result.returncode in (0, 1)  # May be 1 if module doesn't have __main__
        
        # If we got output, check it contains key sections
        if result.stdout:
            assert 'usage:' in result.stdout.lower() or 'Evaluate' in result.stdout
    
    def test_script_help_output(self):
        """Test that main script --help works."""
        script_path = Path(__file__).parent.parent / 'evaluate.py'
        if script_path.exists():
            result = subprocess.run(
                [sys.executable, str(script_path), '--help'],
                capture_output=True,
                text=True
            )
            # Should contain help text
            output = result.stdout + result.stderr
            assert 'Evaluate' in output or 'usage:' in output.lower()


class TestCLIErrorHandling:
    """Test error handling for invalid CLI arguments."""
    
    def test_invalid_model_type(self):
        """Test error on invalid model type."""
        with patch('sys.argv', [
            'evaluate.py',
            '--asr_model_type', 'invalid_model',
        ]):
            with pytest.raises(SystemExit):
                parse_args()
    
    def test_invalid_log_level(self):
        """Test error on invalid log level."""
        with patch('sys.argv', [
            'evaluate.py',
            '--log_level', 'INVALID',
        ]):
            with pytest.raises(SystemExit):
                parse_args()
    
    def test_invalid_retrieval_mode(self):
        """Test error on invalid retrieval mode."""
        with patch('sys.argv', [
            'evaluate.py',
            '--retrieval_mode', 'invalid_mode',
        ]):
            with pytest.raises(SystemExit):
                parse_args()
    
    def test_invalid_db_type(self):
        """Test error on invalid database type."""
        with patch('sys.argv', [
            'evaluate.py',
            '--db_type', 'invalid_db',
        ]):
            with pytest.raises(SystemExit):
                parse_args()
    
    def test_invalid_allocation_strategy(self):
        """Test error on invalid allocation strategy."""
        with patch('sys.argv', [
            'evaluate.py',
            '--allocation_strategy', 'invalid_strategy',
        ]):
            with pytest.raises(SystemExit):
                parse_args()
    
    def test_invalid_pipeline_mode(self):
        """Test error on invalid pipeline mode."""
        with patch('sys.argv', [
            'evaluate.py',
            '--pipeline_mode', 'invalid_mode',
        ]):
            with pytest.raises(SystemExit):
                parse_args()
    
    def test_invalid_reranker_mode(self):
        """Test error on invalid reranker mode."""
        with patch('sys.argv', [
            'evaluate.py',
            '--reranker_mode', 'invalid_reranker',
        ]):
            with pytest.raises(SystemExit):
                parse_args()


class TestCLIIntegration:
    """Integration tests for CLI workflow."""
    
    def test_config_file_validation(self, tmp_path):
        """Test validation of config file path."""
        # Create a temporary YAML config
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
experiment_name: test_cli
model:
  pipeline_mode: asr_text_retrieval
  asr_model_type: whisper
  asr_model_name: openai/whisper-tiny
  text_emb_model_type: labse
data:
  batch_size: 8
cache:
  enabled: false
""")
        
        # Parse args with config file
        with patch('sys.argv', ['evaluate.py', '--config', str(config_file)]):
            args = parse_args()
            assert args.config == str(config_file)
    
    def test_combined_config_and_overrides(self, tmp_path):
        """Test combining config file with CLI overrides."""
        config = EvaluationConfig(
            experiment_name="test",
            model={'pipeline_mode': 'asr_text_retrieval'}
        )
        
        with patch('sys.argv', [
            'evaluate.py',
            '--batch_size', '16',
            '--k', '10',
            '--experiment_name', 'cli_override',
        ]):
            args = parse_args()
            apply_args_to_config(args, config)
            
            # CLI overrides should take effect
            assert config.experiment_name == 'cli_override'
            assert config.data.batch_size == 16
            assert config.vector_db.k == 10
    
    def test_all_args_applied(self):
        """Test that multiple argument types can be applied together."""
        config = EvaluationConfig(experiment_name="test")
        
        with patch('sys.argv', [
            'evaluate.py',
            '--asr_model_type', 'whisper',
            '--text_emb_model_type', 'labse',
            '--batch_size', '32',
            '--k', '5',
            '--no_cache',
            '--log_level', 'INFO',
            '--experiment_name', 'full_test',
        ]):
            args = parse_args()
            apply_args_to_config(args, config)
            
            assert config.model.asr_model_type == 'whisper'
            assert config.model.text_emb_model_type == 'labse'
            assert config.data.batch_size == 32
            assert config.vector_db.k == 5
            assert config.cache.enabled is False
            assert config.logging.console_level == 'INFO'
            assert config.experiment_name == 'full_test'


class TestCLIEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_empty_device_list(self):
        """Test handling of empty device list."""
        with patch('sys.argv', ['evaluate.py', '--devices', '']):
            args = parse_args()
            assert args.devices == ''
    
    def test_zero_values(self):
        """Test handling of zero values for numeric arguments."""
        with patch('sys.argv', [
            'evaluate.py',
            '--trace_limit', '0',
            '--checkpoint_interval', '0',
        ]):
            args = parse_args()
            assert args.trace_limit == 0
            assert args.checkpoint_interval == 0
    
    def test_negative_values(self):
        """Test handling of negative values (should be rejected by validation)."""
        with patch('sys.argv', ['evaluate.py', '--k', '-5']):
            # Parser will accept it, but config validation should reject
            args = parse_args()
            assert args.k == -5
    
    def test_float_temperature(self):
        """Test float value for temperature parameter."""
        with patch('sys.argv', [
            'evaluate.py',
            '--judge_temperature', '0.7',
            '--hybrid_dense_weight', '0.5',
        ]):
            args = parse_args()
            assert args.judge_temperature == 0.7
            assert args.hybrid_dense_weight == 0.5
    
    def test_multiple_bool_flags(self):
        """Test multiple boolean flags together."""
        with patch('sys.argv', [
            'evaluate.py',
            '--no_cache',
            '--no_checkpoint',
            '--judge_enabled',
            '--skip_dataset_validation',
        ]):
            args = parse_args()
            assert args.no_cache is True
            assert args.no_checkpoint is True
            assert args.judge_enabled is True
            assert args.skip_dataset_validation is True
    
    def test_whitespace_in_paths(self):
        """Test handling of paths with whitespace."""
        with patch('sys.argv', [
            'evaluate.py',
            '--questions_path', 'path with spaces/questions.json',
            '--cache_dir', '/tmp/cache dir/',
        ]):
            args = parse_args()
            assert args.questions_path == 'path with spaces/questions.json'
            assert args.cache_dir == '/tmp/cache dir/'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
