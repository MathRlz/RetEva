"""Unit tests for PipelineBundle dataclass."""
import unittest
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluator.pipeline.types import PipelineBundle


class TestPipelineBundle(unittest.TestCase):
    """Test PipelineBundle dataclass."""
    
    def test_default_initialization(self):
        """Test that PipelineBundle initializes with None values."""
        bundle = PipelineBundle()
        
        self.assertIsNone(bundle.asr_pipeline)
        self.assertIsNone(bundle.text_embedding_pipeline)
        self.assertIsNone(bundle.audio_embedding_pipeline)
        self.assertIsNone(bundle.retrieval_pipeline)
        self.assertEqual(bundle.mode, "")
    
    def test_initialization_with_values(self):
        """Test that PipelineBundle accepts values."""
        mock_asr = MagicMock()
        mock_text = MagicMock()
        mock_audio = MagicMock()
        mock_retrieval = MagicMock()
        
        bundle = PipelineBundle(
            asr_pipeline=mock_asr,
            text_embedding_pipeline=mock_text,
            audio_embedding_pipeline=mock_audio,
            retrieval_pipeline=mock_retrieval,
            mode="asr_text_retrieval",
        )
        
        self.assertEqual(bundle.asr_pipeline, mock_asr)
        self.assertEqual(bundle.text_embedding_pipeline, mock_text)
        self.assertEqual(bundle.audio_embedding_pipeline, mock_audio)
        self.assertEqual(bundle.retrieval_pipeline, mock_retrieval)
        self.assertEqual(bundle.mode, "asr_text_retrieval")


class TestPipelineBundleHasProperties(unittest.TestCase):
    """Test PipelineBundle has_* properties."""
    
    def test_has_asr_true(self):
        """Test has_asr returns True when pipeline is set."""
        bundle = PipelineBundle(asr_pipeline=MagicMock())
        self.assertTrue(bundle.has_asr)
    
    def test_has_asr_false(self):
        """Test has_asr returns False when pipeline is None."""
        bundle = PipelineBundle()
        self.assertFalse(bundle.has_asr)
    
    def test_has_text_embedding_true(self):
        """Test has_text_embedding returns True when pipeline is set."""
        bundle = PipelineBundle(text_embedding_pipeline=MagicMock())
        self.assertTrue(bundle.has_text_embedding)
    
    def test_has_text_embedding_false(self):
        """Test has_text_embedding returns False when pipeline is None."""
        bundle = PipelineBundle()
        self.assertFalse(bundle.has_text_embedding)
    
    def test_has_audio_embedding_true(self):
        """Test has_audio_embedding returns True when pipeline is set."""
        bundle = PipelineBundle(audio_embedding_pipeline=MagicMock())
        self.assertTrue(bundle.has_audio_embedding)
    
    def test_has_audio_embedding_false(self):
        """Test has_audio_embedding returns False when pipeline is None."""
        bundle = PipelineBundle()
        self.assertFalse(bundle.has_audio_embedding)
    
    def test_has_retrieval_true(self):
        """Test has_retrieval returns True when pipeline is set."""
        bundle = PipelineBundle(retrieval_pipeline=MagicMock())
        self.assertTrue(bundle.has_retrieval)
    
    def test_has_retrieval_false(self):
        """Test has_retrieval returns False when pipeline is None."""
        bundle = PipelineBundle()
        self.assertFalse(bundle.has_retrieval)


class TestPipelineBundleValidation(unittest.TestCase):
    """Test PipelineBundle validation."""
    
    def test_validate_asr_only_success(self):
        """Test validation passes for asr_only mode with ASR pipeline."""
        bundle = PipelineBundle(asr_pipeline=MagicMock())
        # Should not raise
        bundle.validate("asr_only")
    
    def test_validate_asr_only_failure(self):
        """Test validation fails for asr_only mode without ASR pipeline."""
        bundle = PipelineBundle()
        
        with self.assertRaises(ValueError) as context:
            bundle.validate("asr_only")
        
        self.assertIn("asr_pipeline", str(context.exception))
        self.assertIn("asr_only", str(context.exception))
    
    def test_validate_asr_text_retrieval_success(self):
        """Test validation passes for asr_text_retrieval with all pipelines."""
        bundle = PipelineBundle(
            asr_pipeline=MagicMock(),
            text_embedding_pipeline=MagicMock(),
            retrieval_pipeline=MagicMock(),
        )
        # Should not raise
        bundle.validate("asr_text_retrieval")
    
    def test_validate_asr_text_retrieval_missing_asr(self):
        """Test validation fails for asr_text_retrieval without ASR."""
        bundle = PipelineBundle(
            text_embedding_pipeline=MagicMock(),
            retrieval_pipeline=MagicMock(),
        )
        
        with self.assertRaises(ValueError) as context:
            bundle.validate("asr_text_retrieval")
        
        self.assertIn("asr_pipeline", str(context.exception))
    
    def test_validate_asr_text_retrieval_missing_multiple(self):
        """Test validation fails with multiple missing pipelines."""
        bundle = PipelineBundle(asr_pipeline=MagicMock())
        
        with self.assertRaises(ValueError) as context:
            bundle.validate("asr_text_retrieval")
        
        error_msg = str(context.exception)
        self.assertIn("text_embedding_pipeline", error_msg)
        self.assertIn("retrieval_pipeline", error_msg)
    
    def test_validate_audio_emb_retrieval_success(self):
        """Test validation passes for audio_emb_retrieval mode."""
        bundle = PipelineBundle(
            audio_embedding_pipeline=MagicMock(),
            retrieval_pipeline=MagicMock(),
        )
        # Should not raise
        bundle.validate("audio_emb_retrieval")
    
    def test_validate_audio_emb_retrieval_failure(self):
        """Test validation fails for audio_emb_retrieval without pipelines."""
        bundle = PipelineBundle()
        
        with self.assertRaises(ValueError) as context:
            bundle.validate("audio_emb_retrieval")
        
        error_msg = str(context.exception)
        self.assertIn("audio_embedding_pipeline", error_msg)
        self.assertIn("retrieval_pipeline", error_msg)
    
    def test_validate_audio_text_retrieval_success(self):
        """Test validation passes for audio_text_retrieval mode."""
        bundle = PipelineBundle(
            audio_embedding_pipeline=MagicMock(),
            text_embedding_pipeline=MagicMock(),
            retrieval_pipeline=MagicMock(),
        )
        # Should not raise
        bundle.validate("audio_text_retrieval")
    
    def test_validate_audio_text_retrieval_failure(self):
        """Test validation fails for audio_text_retrieval without pipelines."""
        bundle = PipelineBundle(audio_embedding_pipeline=MagicMock())
        
        with self.assertRaises(ValueError) as context:
            bundle.validate("audio_text_retrieval")
        
        error_msg = str(context.exception)
        self.assertIn("text_embedding_pipeline", error_msg)
        self.assertIn("retrieval_pipeline", error_msg)
    
    def test_validate_unknown_mode(self):
        """Test validation raises error for unknown mode."""
        bundle = PipelineBundle()
        
        with self.assertRaises(ValueError) as context:
            bundle.validate("unknown_mode")
        
        self.assertIn("Unknown pipeline mode", str(context.exception))
        self.assertIn("unknown_mode", str(context.exception))


class TestPipelineBundleGetRequiredPipelines(unittest.TestCase):
    """Test _get_required_pipelines static method."""
    
    def test_asr_only_requirements(self):
        """Test asr_only mode requires only ASR."""
        required = PipelineBundle._get_required_pipelines("asr_only")
        self.assertEqual(required, {"asr"})
    
    def test_asr_text_retrieval_requirements(self):
        """Test asr_text_retrieval mode requirements."""
        required = PipelineBundle._get_required_pipelines("asr_text_retrieval")
        self.assertEqual(required, {"asr", "text_embedding", "retrieval"})
    
    def test_audio_emb_retrieval_requirements(self):
        """Test audio_emb_retrieval mode requirements."""
        required = PipelineBundle._get_required_pipelines("audio_emb_retrieval")
        self.assertEqual(required, {"audio_embedding", "retrieval"})
    
    def test_audio_text_retrieval_requirements(self):
        """Test audio_text_retrieval mode requirements."""
        required = PipelineBundle._get_required_pipelines("audio_text_retrieval")
        self.assertEqual(required, {"audio_embedding", "text_embedding", "retrieval"})
    
    def test_unknown_mode_raises_error(self):
        """Test that unknown mode raises ValueError."""
        with self.assertRaises(ValueError) as context:
            PipelineBundle._get_required_pipelines("invalid_mode")
        
        self.assertIn("Unknown pipeline mode", str(context.exception))


if __name__ == "__main__":
    unittest.main()
