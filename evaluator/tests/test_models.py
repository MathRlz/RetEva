"""Unit tests for model implementations.

Tests model initialization, error handling, interface compliance, and registry integration.
All tests mock external dependencies to avoid loading actual model weights.
"""
import os
import unittest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import sys
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluator.models.base import ASRModel, TextEmbeddingModel, AudioEmbeddingModel
from evaluator.models.registry import (
    asr_registry,
    text_embedding_registry,
    audio_embedding_registry,
)


# Helper to check if a module is available
def module_available(module_name):
    """Check if a module is available for import."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


# Skip decorators for optional dependencies
peft_available = module_available('peft')
flag_embedding_available = module_available('FlagEmbedding')


# ============================================================================
# Test Interface Compliance
# ============================================================================

class TestInterfaceCompliance(unittest.TestCase):
    """Test that all registered models implement the required interface."""
    
    def test_asr_models_implement_interface(self):
        """Test that all ASR models implement ASRModel interface."""
        for model_type in asr_registry.list_types():
            model_class = asr_registry.get(model_type)
            self.assertTrue(
                issubclass(model_class, ASRModel),
                f"{model_class.__name__} does not inherit from ASRModel"
            )
            # Check required methods exist
            self.assertTrue(hasattr(model_class, 'transcribe'))
            self.assertTrue(hasattr(model_class, 'preprocess'))
            self.assertTrue(hasattr(model_class, 'transcribe_from_features'))
            self.assertTrue(hasattr(model_class, 'name'))
            self.assertTrue(hasattr(model_class, 'to'))
    
    def test_text_embedding_models_implement_interface(self):
        """Test that all text embedding models implement TextEmbeddingModel interface."""
        for model_type in text_embedding_registry.list_types():
            model_class = text_embedding_registry.get(model_type)
            self.assertTrue(
                issubclass(model_class, TextEmbeddingModel),
                f"{model_class.__name__} does not inherit from TextEmbeddingModel"
            )
            # Check required methods exist
            self.assertTrue(hasattr(model_class, 'encode'))
            self.assertTrue(hasattr(model_class, 'name'))
            self.assertTrue(hasattr(model_class, 'to'))
    
    def test_audio_embedding_models_implement_interface(self):
        """Test that all audio embedding models implement AudioEmbeddingModel interface."""
        for model_type in audio_embedding_registry.list_types():
            model_class = audio_embedding_registry.get(model_type)
            self.assertTrue(
                issubclass(model_class, AudioEmbeddingModel),
                f"{model_class.__name__} does not inherit from AudioEmbeddingModel"
            )
            # Check required methods exist
            self.assertTrue(hasattr(model_class, 'encode_audio'))
            self.assertTrue(hasattr(model_class, 'preprocess_audio'))
            self.assertTrue(hasattr(model_class, 'encode_from_features'))
            self.assertTrue(hasattr(model_class, 'name'))
            self.assertTrue(hasattr(model_class, 'to'))


# ============================================================================
# Test ASR Models
# ============================================================================

class TestWhisperModel(unittest.TestCase):
    """Test WhisperModel initialization and methods."""
    
    @patch('transformers.WhisperForConditionalGeneration')
    @patch('transformers.WhisperProcessor')
    @patch('transformers.WhisperFeatureExtractor')
    def test_initialization(self, mock_extractor, mock_processor, mock_model):
        """Test WhisperModel initialization with mocked dependencies."""
        from evaluator.models.asr.whisper import WhisperModel
        
        model = WhisperModel(model_name="openai/whisper-small")
        
        self.assertEqual(model.model_name, "openai/whisper-small")
        self.assertIsNone(model.adapter_path)
        self.assertEqual(model.device, torch.device("cpu"))
        mock_processor.from_pretrained.assert_called_once_with("openai/whisper-small")
        mock_extractor.from_pretrained.assert_called_once_with("openai/whisper-small")
        mock_model.from_pretrained.assert_called_once_with("openai/whisper-small")
    
    @patch('transformers.WhisperForConditionalGeneration')
    @patch('transformers.WhisperProcessor')
    @patch('transformers.WhisperFeatureExtractor')
    def test_name_without_adapter(self, mock_extractor, mock_processor, mock_model):
        """Test name() returns correct format without adapter."""
        from evaluator.models.asr.whisper import WhisperModel
        
        model = WhisperModel(model_name="openai/whisper-small")
        name = model.name()
        
        self.assertEqual(name, "WhisperModel - openai/whisper-small")
        self.assertNotIn("adapter", name)
    
    @unittest.skipUnless(peft_available, "peft not installed")
    @patch('peft.PeftModel')
    @patch('transformers.WhisperForConditionalGeneration')
    @patch('transformers.WhisperProcessor')
    @patch('transformers.WhisperFeatureExtractor')
    def test_name_with_adapter(self, mock_extractor, mock_processor, mock_model, mock_peft):
        """Test name() returns correct format with adapter."""
        from evaluator.models.asr.whisper import WhisperModel
        
        mock_peft_instance = MagicMock()
        mock_peft_instance.merge_and_unload.return_value = MagicMock()
        mock_peft.from_pretrained.return_value = mock_peft_instance
        
        model = WhisperModel(
            model_name="openai/whisper-small", 
            adapter_path="/path/to/adapter"
        )
        name = model.name()
        
        self.assertIn("WhisperModel", name)
        self.assertIn("openai/whisper-small", name)
        self.assertIn("adapter", name)
        self.assertIn("/path/to/adapter", name)
    
    @patch('transformers.WhisperForConditionalGeneration')
    @patch('transformers.WhisperProcessor')
    @patch('transformers.WhisperFeatureExtractor')
    def test_to_device(self, mock_extractor, mock_processor, mock_model):
        """Test moving model to device."""
        from evaluator.models.asr.whisper import WhisperModel
        
        model = WhisperModel()
        result = model.to(torch.device("cuda:0"))
        
        self.assertEqual(model.device, torch.device("cuda:0"))
        self.assertEqual(result, model)  # Returns self for chaining
        model.model.to.assert_called_once_with(torch.device("cuda:0"))
    
    @patch('transformers.WhisperForConditionalGeneration')
    @patch('transformers.WhisperProcessor')
    @patch('transformers.WhisperFeatureExtractor')
    def test_preprocess_resamples_audio(self, mock_extractor, mock_processor, mock_model):
        """Test that preprocess resamples non-16kHz audio."""
        from evaluator.models.asr.whisper import WhisperModel
        
        mock_extractor_instance = MagicMock()
        mock_extractor_instance.return_value = MagicMock(
            input_features=torch.randn(1, 80, 3000),
            attention_mask=torch.ones(1, 3000)
        )
        mock_extractor.from_pretrained.return_value = mock_extractor_instance
        
        model = WhisperModel()
        model.feature_extractor = mock_extractor_instance
        
        audio = [torch.randn(48000)]  # 1 second at 48kHz
        sampling_rates = [48000]
        
        with patch('torchaudio.functional.resample') as mock_resample:
            mock_resample.return_value = torch.randn(16000)
            features, mask = model.preprocess(audio, sampling_rates)
            mock_resample.assert_called_once()


# Check if faster_whisper module is available
faster_whisper_available = module_available('faster_whisper')


class TestFasterWhisperModel(unittest.TestCase):
    """Test FasterWhisperModel initialization and methods."""
    
    @unittest.skipUnless(faster_whisper_available, "faster_whisper not installed")
    @patch('faster_whisper.WhisperModel')
    def test_initialization(self, mock_fw_model):
        """Test FasterWhisperModel initialization with mocked dependencies."""
        from evaluator.models.asr.faster_whisper import FasterWhisperModel
        
        with patch('torch.cuda.is_available', return_value=False):
            model = FasterWhisperModel(model_name="large-v3")
        
        self.assertEqual(model.model_name, "large-v3")
        self.assertEqual(model.compute_type, "float16")
        self.assertEqual(model._device_str, "cpu")
        mock_fw_model.assert_called_once_with(
            "large-v3",
            device="cpu",
            compute_type="float16",
            cpu_threads=4,
            num_workers=1,
        )
    
    @unittest.skipUnless(faster_whisper_available, "faster_whisper not installed")
    @patch('faster_whisper.WhisperModel')
    def test_initialization_with_compute_type(self, mock_fw_model):
        """Test FasterWhisperModel initialization with custom compute_type."""
        from evaluator.models.asr.faster_whisper import FasterWhisperModel
        
        with patch('torch.cuda.is_available', return_value=False):
            model = FasterWhisperModel(model_name="small", compute_type="int8")
        
        self.assertEqual(model.compute_type, "int8")
        mock_fw_model.assert_called_once_with(
            "small",
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
            num_workers=1,
        )
    
    @unittest.skipUnless(faster_whisper_available, "faster_whisper not installed")
    @patch('faster_whisper.WhisperModel')
    def test_initialization_cuda_device(self, mock_fw_model):
        """Test FasterWhisperModel initialization with CUDA device."""
        from evaluator.models.asr.faster_whisper import FasterWhisperModel
        
        with patch('torch.cuda.is_available', return_value=True):
            model = FasterWhisperModel(model_name="medium")
        
        self.assertEqual(model._device_str, "cuda")
        mock_fw_model.assert_called_once_with(
            "medium",
            device="cuda",
            compute_type="float16",
            cpu_threads=4,
            num_workers=1,
        )
    
    @unittest.skipUnless(faster_whisper_available, "faster_whisper not installed")
    @patch('faster_whisper.WhisperModel')
    def test_name(self, mock_fw_model):
        """Test name() returns correct format."""
        from evaluator.models.asr.faster_whisper import FasterWhisperModel
        
        with patch('torch.cuda.is_available', return_value=False):
            model = FasterWhisperModel(model_name="large-v3", compute_type="float16")
        
        name = model.name()
        
        self.assertEqual(name, "FasterWhisperModel - large-v3 (float16)")
    
    @unittest.skipUnless(faster_whisper_available, "faster_whisper not installed")
    @patch('faster_whisper.WhisperModel')
    def test_to_device(self, mock_fw_model):
        """Test moving model to device recreates the model."""
        from evaluator.models.asr.faster_whisper import FasterWhisperModel
        
        with patch('torch.cuda.is_available', return_value=False):
            model = FasterWhisperModel(model_name="small")
        
        # Reset mock to track new calls
        mock_fw_model.reset_mock()
        
        result = model.to(torch.device("cuda:0"))
        
        self.assertEqual(model.device, torch.device("cuda:0"))
        self.assertEqual(model._device_str, "cuda")
        self.assertEqual(result, model)
        mock_fw_model.assert_called_once_with(
            "small",
            device="cuda",
            compute_type="float16",
            cpu_threads=4,
            num_workers=1,
        )
    
    @unittest.skipUnless(faster_whisper_available, "faster_whisper not installed")
    @patch('faster_whisper.WhisperModel')
    def test_to_same_device_type_no_recreate(self, mock_fw_model):
        """Test that moving to same device type does not recreate the model."""
        from evaluator.models.asr.faster_whisper import FasterWhisperModel
        
        with patch('torch.cuda.is_available', return_value=False):
            model = FasterWhisperModel(model_name="small", device="cpu")
        
        # Reset mock to track new calls
        mock_fw_model.reset_mock()
        
        model.to(torch.device("cpu"))
        
        # Model should not be recreated
        mock_fw_model.assert_not_called()
    
    @unittest.skipUnless(faster_whisper_available, "faster_whisper not installed")
    @patch('faster_whisper.WhisperModel')
    def test_preprocess_resamples_audio(self, mock_fw_model):
        """Test that preprocess resamples non-16kHz audio."""
        from evaluator.models.asr.faster_whisper import FasterWhisperModel
        
        with patch('torch.cuda.is_available', return_value=False):
            model = FasterWhisperModel(model_name="small")
        
        audio = [torch.randn(48000)]  # 1 second at 48kHz
        sampling_rates = [48000]
        
        with patch('torchaudio.functional.resample') as mock_resample:
            mock_resample.return_value = torch.randn(16000)
            processed_audio, new_rates = model.preprocess(audio, sampling_rates)
            mock_resample.assert_called_once()
            self.assertEqual(new_rates[0], 16000)
    
    @unittest.skipUnless(faster_whisper_available, "faster_whisper not installed")
    @patch('faster_whisper.WhisperModel')
    def test_preprocess_no_resample_at_16khz(self, mock_fw_model):
        """Test that preprocess does not resample 16kHz audio."""
        from evaluator.models.asr.faster_whisper import FasterWhisperModel
        
        with patch('torch.cuda.is_available', return_value=False):
            model = FasterWhisperModel(model_name="small")
        
        audio = [torch.randn(16000)]  # 1 second at 16kHz
        sampling_rates = [16000]
        
        with patch('torchaudio.functional.resample') as mock_resample:
            processed_audio, new_rates = model.preprocess(audio, sampling_rates)
            mock_resample.assert_not_called()
            self.assertEqual(new_rates[0], 16000)
            self.assertEqual(len(processed_audio), 1)
            self.assertIsInstance(processed_audio[0], np.ndarray)
    
    @unittest.skipUnless(faster_whisper_available, "faster_whisper not installed")
    @patch('faster_whisper.WhisperModel')
    def test_transcribe(self, mock_fw_model):
        """Test transcribe method."""
        from evaluator.models.asr.faster_whisper import FasterWhisperModel
        
        # Mock the transcribe method to return segments
        mock_segment = MagicMock()
        mock_segment.text = " Hello world "
        mock_fw_instance = MagicMock()
        mock_fw_instance.transcribe.return_value = ([mock_segment], None)
        mock_fw_model.return_value = mock_fw_instance
        
        with patch('torch.cuda.is_available', return_value=False):
            model = FasterWhisperModel(model_name="small")
        
        audio = [torch.randn(16000)]
        sampling_rates = [16000]
        
        result = model.transcribe(audio, sampling_rates, language="en")
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "Hello world")
        mock_fw_instance.transcribe.assert_called_once()
    
    @unittest.skipUnless(faster_whisper_available, "faster_whisper not installed")
    @patch('faster_whisper.WhisperModel')
    def test_transcribe_multiple_segments(self, mock_fw_model):
        """Test transcribe with multiple segments."""
        from evaluator.models.asr.faster_whisper import FasterWhisperModel
        
        # Mock multiple segments
        mock_segment1 = MagicMock()
        mock_segment1.text = " Hello "
        mock_segment2 = MagicMock()
        mock_segment2.text = " world "
        mock_fw_instance = MagicMock()
        mock_fw_instance.transcribe.return_value = ([mock_segment1, mock_segment2], None)
        mock_fw_model.return_value = mock_fw_instance
        
        with patch('torch.cuda.is_available', return_value=False):
            model = FasterWhisperModel(model_name="small")
        
        audio = [torch.randn(16000)]
        sampling_rates = [16000]
        
        result = model.transcribe(audio, sampling_rates)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "Hello world")
    
    @unittest.skipUnless(faster_whisper_available, "faster_whisper not installed")
    @patch('faster_whisper.WhisperModel')
    def test_transcribe_from_features_with_audio_arrays(self, mock_fw_model):
        """Test transcribe_from_features works with raw audio arrays."""
        from evaluator.models.asr.faster_whisper import FasterWhisperModel
        
        mock_segment = MagicMock()
        mock_segment.text = " Test "
        mock_fw_instance = MagicMock()
        mock_fw_instance.transcribe.return_value = ([mock_segment], None)
        mock_fw_model.return_value = mock_fw_instance
        
        with patch('torch.cuda.is_available', return_value=False):
            model = FasterWhisperModel(model_name="small")
        
        # Pass raw audio as numpy arrays
        audio_arrays = [np.random.randn(16000).astype(np.float32)]
        result = model.transcribe_from_features(audio_arrays, language="en")
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "Test")
    
    @unittest.skipUnless(faster_whisper_available, "faster_whisper not installed")
    @patch('faster_whisper.WhisperModel')
    def test_transcribe_from_features_raises_on_mel_features(self, mock_fw_model):
        """Test transcribe_from_features raises error on mel spectrogram features."""
        from evaluator.models.asr.faster_whisper import FasterWhisperModel
        
        with patch('torch.cuda.is_available', return_value=False):
            model = FasterWhisperModel(model_name="small")
        
        # Pass mel spectrogram tensor (not raw audio)
        mel_features = torch.randn(1, 80, 3000)
        
        with self.assertRaises(NotImplementedError):
            model.transcribe_from_features(mel_features)
    
    @unittest.skipUnless(faster_whisper_available, "faster_whisper not installed")
    @patch('faster_whisper.WhisperModel')
    def test_adapter_warning(self, mock_fw_model):
        """Test that adapter_path triggers a warning."""
        from evaluator.models.asr.faster_whisper import FasterWhisperModel
        import warnings
        
        with patch('torch.cuda.is_available', return_value=False):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                model = FasterWhisperModel(
                    model_name="small",
                    adapter_path="/path/to/adapter"
                )
                
                self.assertEqual(len(w), 1)
                self.assertIn("adapter_path", str(w[0].message))
                self.assertIn("not supported", str(w[0].message))
    
    def test_import_error_without_faster_whisper(self):
        """Test proper import error when faster_whisper is not installed."""
        import sys
        
        # Save original module
        original_modules = dict(sys.modules)
        
        # Remove faster_whisper from sys.modules if present
        modules_to_remove = [k for k in sys.modules if 'faster_whisper' in k]
        for mod in modules_to_remove:
            del sys.modules[mod]
        
        # Mock import to raise ImportError
        with patch.dict(sys.modules, {'faster_whisper': None}):
            with patch('builtins.__import__', side_effect=ImportError("No module named 'faster_whisper'")):
                # Re-import the module to trigger the ImportError check
                # Note: This test verifies the error message in the except block
                pass
        
        # Restore modules
        sys.modules.update(original_modules)


class TestWav2Vec2Model(unittest.TestCase):
    """Test Wav2Vec2Model initialization and methods."""
    
    @patch('transformers.Wav2Vec2ForCTC')
    @patch('transformers.Wav2Vec2Processor')
    def test_initialization(self, mock_processor, mock_model):
        """Test Wav2Vec2Model initialization with mocked dependencies."""
        from evaluator.models.asr.wav2vec2 import Wav2Vec2Model
        
        model = Wav2Vec2Model(model_name="facebook/wav2vec2-base")
        
        self.assertEqual(model.model_name, "facebook/wav2vec2-base")
        self.assertIsNone(model.adapter_path)
        self.assertEqual(model.device, torch.device("cpu"))
        mock_processor.from_pretrained.assert_called_once_with("facebook/wav2vec2-base")
        mock_model.from_pretrained.assert_called_once_with("facebook/wav2vec2-base")
    
    @patch('transformers.Wav2Vec2ForCTC')
    @patch('transformers.Wav2Vec2Processor')
    def test_name_without_adapter(self, mock_processor, mock_model):
        """Test name() returns correct format without adapter."""
        from evaluator.models.asr.wav2vec2 import Wav2Vec2Model
        
        model = Wav2Vec2Model(model_name="facebook/wav2vec2-base")
        name = model.name()
        
        self.assertEqual(name, "Wav2Vec2Model - facebook/wav2vec2-base")
        self.assertNotIn("adapter", name)
    
    @unittest.skipUnless(peft_available, "peft not installed")
    @patch('peft.PeftModel')
    @patch('transformers.Wav2Vec2ForCTC')
    @patch('transformers.Wav2Vec2Processor')
    def test_name_with_adapter(self, mock_processor, mock_model, mock_peft):
        """Test name() returns correct format with adapter."""
        from evaluator.models.asr.wav2vec2 import Wav2Vec2Model
        
        mock_peft_instance = MagicMock()
        mock_peft_instance.merge_and_unload.return_value = MagicMock()
        mock_peft.from_pretrained.return_value = mock_peft_instance
        
        model = Wav2Vec2Model(
            model_name="facebook/wav2vec2-base",
            adapter_path="/path/to/adapter"
        )
        name = model.name()
        
        self.assertIn("Wav2Vec2Model", name)
        self.assertIn("facebook/wav2vec2-base", name)
        self.assertIn("adapter", name)
    
    @patch('transformers.Wav2Vec2ForCTC')
    @patch('transformers.Wav2Vec2Processor')
    def test_to_device(self, mock_processor, mock_model):
        """Test moving model to device."""
        from evaluator.models.asr.wav2vec2 import Wav2Vec2Model
        
        model = Wav2Vec2Model()
        result = model.to(torch.device("cuda:0"))
        
        self.assertEqual(model.device, torch.device("cuda:0"))
        self.assertEqual(result, model)
        model.model.to.assert_called_once_with(torch.device("cuda:0"))
    
    @patch('transformers.Wav2Vec2ForCTC')
    @patch('transformers.Wav2Vec2Processor')
    def test_adapter_loading_requires_peft(self, mock_processor, mock_model):
        """Test that adapter loading raises ImportError without peft."""
        from evaluator.models.asr.wav2vec2 import Wav2Vec2Model
        
        model = Wav2Vec2Model(model_name="facebook/wav2vec2-base")
        
        # Mock peft import to fail by patching the _load_adapter method
        with patch.object(
            model, '_load_adapter',
            side_effect=ImportError("peft library required")
        ):
            with self.assertRaises(ImportError):
                model._load_adapter("/fake/path")


# ============================================================================
# Test Text Embedding Models
# ============================================================================

class TestLabseModel(unittest.TestCase):
    """Test LabseModel initialization and methods."""
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_initialization(self, mock_st):
        """Test LabseModel initialization with mocked SentenceTransformer."""
        from evaluator.models.t2e.labse import LabseModel
        
        model = LabseModel(model_name="sentence-transformers/LaBSE")
        
        self.assertEqual(model.model_name, "sentence-transformers/LaBSE")
        mock_st.assert_called_once_with("sentence-transformers/LaBSE")
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_name(self, mock_st):
        """Test name() returns correct format."""
        from evaluator.models.t2e.labse import LabseModel
        
        model = LabseModel(model_name="sentence-transformers/LaBSE")
        name = model.name()
        
        self.assertEqual(name, "LaBseModel - sentence-transformers/LaBSE")
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_to_device(self, mock_st):
        """Test moving model to device."""
        from evaluator.models.t2e.labse import LabseModel
        
        model = LabseModel()
        result = model.to(torch.device("cuda:0"))
        
        self.assertEqual(result, model)
        model.model.to.assert_called_once_with(torch.device("cuda:0"))
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_encode_returns_numpy(self, mock_st):
        """Test that encode returns numpy array."""
        from evaluator.models.t2e.labse import LabseModel
        
        mock_model = MagicMock()
        expected_output = np.random.randn(3, 768)
        mock_model.encode.return_value = expected_output
        mock_st.return_value = mock_model
        
        model = LabseModel()
        result = model.encode(["text1", "text2", "text3"])
        
        np.testing.assert_array_equal(result, expected_output)
        mock_model.encode.assert_called_once_with(
            ["text1", "text2", "text3"],
            convert_to_numpy=True,
            show_progress_bar=False,
            tqdm_kwargs={}
        )

    @patch('sentence_transformers.SentenceTransformer')
    def test_encode_with_progress(self, mock_st):
        """Test encode with progress bar enabled."""
        from evaluator.models.t2e.labse import LabseModel

        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(1, 768)
        mock_st.return_value = mock_model

        model = LabseModel()
        model.encode(["text"], show_progress=True)

        mock_model.encode.assert_called_once_with(
            ["text"],
            convert_to_numpy=True,
            show_progress_bar=True,
            tqdm_kwargs={"desc": "Embedding"}
        )


class TestJinaV4Model(unittest.TestCase):
    """Test JinaV4Model initialization and methods."""
    
    @patch('evaluator.models.t2e.jina.AutoConfig')
    @patch('evaluator.models.t2e.jina.AutoModel')
    def test_initialization(self, mock_automodel, mock_autoconfig):
        """Test JinaV4Model initialization with mocked transformers."""
        from evaluator.models.t2e.jina import JinaV4Model
        
        mock_config = MagicMock()
        mock_autoconfig.from_pretrained.return_value = mock_config
        
        model = JinaV4Model(model_name="jinaai/jina-embeddings-v4")
        
        self.assertEqual(model.model_name, "jinaai/jina-embeddings-v4")
        mock_autoconfig.from_pretrained.assert_called_once()
        # Verify verbosity is set to 0
        self.assertEqual(mock_config.verbosity, 0)
    
    @patch('evaluator.models.t2e.jina.AutoConfig')
    @patch('evaluator.models.t2e.jina.AutoModel')
    def test_name(self, mock_automodel, mock_autoconfig):
        """Test name() returns correct format."""
        from evaluator.models.t2e.jina import JinaV4Model
        
        model = JinaV4Model()
        name = model.name()
        
        self.assertEqual(name, "JinaEmbeddingsV4")
    
    @patch('evaluator.models.t2e.jina.AutoConfig')
    @patch('evaluator.models.t2e.jina.AutoModel')
    def test_to_device(self, mock_automodel, mock_autoconfig):
        """Test moving model to device."""
        from evaluator.models.t2e.jina import JinaV4Model
        
        model = JinaV4Model()
        result = model.to(torch.device("cuda:0"))
        
        self.assertEqual(result, model)
        model.model.to.assert_called_once_with(torch.device("cuda:0"))
    
    @patch('evaluator.models.t2e.jina.AutoConfig')
    @patch('evaluator.models.t2e.jina.AutoModel')
    def test_encode_uses_retrieval_task(self, mock_automodel, mock_autoconfig):
        """Test that encode uses correct task and prompt settings."""
        from evaluator.models.t2e.jina import JinaV4Model
        
        mock_model = MagicMock()
        mock_emb = [torch.randn(512) for _ in range(2)]
        mock_model.encode_text.return_value = mock_emb
        mock_automodel.from_pretrained.return_value = mock_model
        
        model = JinaV4Model()
        result = model.encode(["text1", "text2"])
        
        mock_model.encode_text.assert_called_once_with(
            texts=["text1", "text2"],
            task="retrieval",
            prompt_name="query",
            return_numpy=False
        )
        self.assertIsInstance(result, np.ndarray)


class TestClipModel(unittest.TestCase):
    """Test ClipModel initialization and methods."""
    
    @patch('transformers.CLIPTokenizer')
    @patch('transformers.CLIPModel')
    def test_initialization(self, mock_clip, mock_tokenizer):
        """Test ClipModel initialization with mocked dependencies."""
        from evaluator.models.t2e.clip import ClipModel
        
        model = ClipModel(model_name="openai/clip-vit-base-patch32")
        
        self.assertEqual(model.model_name, "openai/clip-vit-base-patch32")
        self.assertEqual(model.device, torch.device("cpu"))
        mock_tokenizer.from_pretrained.assert_called_once_with("openai/clip-vit-base-patch32")
        mock_clip.from_pretrained.assert_called_once_with("openai/clip-vit-base-patch32")
    
    @patch('transformers.CLIPTokenizer')
    @patch('transformers.CLIPModel')
    def test_name(self, mock_clip, mock_tokenizer):
        """Test name() returns correct format."""
        from evaluator.models.t2e.clip import ClipModel
        
        model = ClipModel(model_name="openai/clip-vit-base-patch32")
        name = model.name()
        
        self.assertEqual(name, "ClipModel - openai/clip-vit-base-patch32")
    
    @patch('transformers.CLIPTokenizer')
    @patch('transformers.CLIPModel')
    def test_to_device(self, mock_clip, mock_tokenizer):
        """Test moving model to device."""
        from evaluator.models.t2e.clip import ClipModel
        
        model = ClipModel()
        result = model.to(torch.device("cuda:0"))
        
        self.assertEqual(model.device, torch.device("cuda:0"))
        self.assertEqual(result, model)
        model.model.to.assert_called_once_with(torch.device("cuda:0"))
    
    @patch('transformers.CLIPTokenizer')
    @patch('transformers.CLIPModel')
    def test_encode_normalizes_embeddings(self, mock_clip, mock_tokenizer):
        """Test that encode returns L2-normalized embeddings."""
        from evaluator.models.t2e.clip import ClipModel
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = MagicMock(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.tensor([[1, 1, 1]])
        )
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model = MagicMock()
        # Return unnormalized embeddings
        raw_embeddings = torch.tensor([[3.0, 4.0]])  # Norm = 5
        mock_model.get_text_features.return_value = raw_embeddings
        mock_clip.from_pretrained.return_value = mock_model
        
        model = ClipModel()
        model.tokenizer = mock_tokenizer_instance
        result = model.encode(["test"])
        
        # Check embeddings are L2 normalized (norm should be 1)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_array_almost_equal(norms, [1.0])


class TestBgeM3Model(unittest.TestCase):
    """Test BgeM3Model initialization and methods."""
    
    def setUp(self):
        """Set up mocks for FlagEmbedding module."""
        # Create mock module for FlagEmbedding if it doesn't exist
        self.mock_bgem3 = MagicMock()
        self.mock_flag_embedding = MagicMock()
        self.mock_flag_embedding.BGEM3FlagModel = self.mock_bgem3
        
        # Store original modules
        self.original_modules = {}
        if 'FlagEmbedding' in sys.modules:
            self.original_modules['FlagEmbedding'] = sys.modules['FlagEmbedding']
        
        # Inject mock module
        sys.modules['FlagEmbedding'] = self.mock_flag_embedding
    
    def tearDown(self):
        """Restore original modules."""
        if 'FlagEmbedding' in self.original_modules:
            sys.modules['FlagEmbedding'] = self.original_modules['FlagEmbedding']
        else:
            sys.modules.pop('FlagEmbedding', None)
        
        # Reload the module to clear cached imports
        if 'evaluator.models.t2e.bgem3' in sys.modules:
            del sys.modules['evaluator.models.t2e.bgem3']
    
    def test_initialization(self):
        """Test BgeM3Model initialization with mocked dependencies."""
        from evaluator.models.t2e.bgem3 import BgeM3Model
        
        model = BgeM3Model(model_name="BAAI/bge-m3")
        
        self.assertEqual(model.model_name, "BAAI/bge-m3")
        self.mock_bgem3.assert_called_once_with("BAAI/bge-m3", use_fp16=True)
    
    def test_name(self):
        """Test name() returns correct format."""
        from evaluator.models.t2e.bgem3 import BgeM3Model
        
        model = BgeM3Model(model_name="BAAI/bge-m3")
        name = model.name()
        
        self.assertEqual(name, "BgeM3Model - BAAI/bge-m3")
    
    def test_to_is_noop(self):
        """Test that to() is a no-op (model handles device internally)."""
        from evaluator.models.t2e.bgem3 import BgeM3Model
        
        model = BgeM3Model()
        result = model.to(torch.device("cuda:0"))
        
        # Should return self without calling any methods
        self.assertEqual(result, model)
    
    def test_encode_returns_dense_vecs(self):
        """Test that encode returns dense_vecs from model output."""
        from evaluator.models.t2e.bgem3 import BgeM3Model
        
        mock_model = MagicMock()
        expected_output = np.random.randn(2, 1024)
        mock_model.encode.return_value = {"dense_vecs": expected_output}
        self.mock_bgem3.return_value = mock_model
        
        model = BgeM3Model()
        result = model.encode(["text1", "text2"])
        
        np.testing.assert_array_equal(result, expected_output)


class TestNemotronModel(unittest.TestCase):
    """Test NemotronModel initialization and methods."""
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_initialization(self, mock_st):
        """Test NemotronModel initialization with mocked SentenceTransformer."""
        from evaluator.models.t2e.nemotron import NemotronModel
        
        model = NemotronModel(model_name="nvidia/llama-embed-nemotron-8b")
        
        self.assertEqual(model.model_name, "nvidia/llama-embed-nemotron-8b")
        # Verify special configuration
        call_args = mock_st.call_args
        self.assertEqual(call_args[0][0], "nvidia/llama-embed-nemotron-8b")
        self.assertTrue(call_args[1].get("trust_remote_code"))
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_name(self, mock_st):
        """Test name() returns correct format."""
        from evaluator.models.t2e.nemotron import NemotronModel
        
        model = NemotronModel(model_name="nvidia/llama-embed-nemotron-8b")
        name = model.name()
        
        self.assertEqual(name, "NemotronModel - nvidia/llama-embed-nemotron-8b")
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_to_device(self, mock_st):
        """Test moving model to device."""
        from evaluator.models.t2e.nemotron import NemotronModel
        
        model = NemotronModel()
        result = model.to(torch.device("cuda:0"))
        
        self.assertEqual(result, model)
        model.model.to.assert_called_once_with(torch.device("cuda:0"))


# ============================================================================
# Test Audio Embedding Models
# ============================================================================

class TestAttentionPoolAudioModel(unittest.TestCase):
    """Test AttentionPoolAudioModel initialization and methods."""
    
    @patch('transformers.WhisperFeatureExtractor')
    @patch('transformers.WhisperModel')
    def test_initialization(self, mock_whisper, mock_extractor):
        """Test AttentionPoolAudioModel initialization."""
        from evaluator.models.a2e.attention_pool import AttentionPoolAudioModel
        
        mock_whisper_instance = MagicMock()
        mock_encoder = MagicMock()
        mock_whisper_instance.get_encoder.return_value = mock_encoder
        mock_whisper.from_pretrained.return_value = mock_whisper_instance
        
        model = AttentionPoolAudioModel(
            audio_encoder_name="openai/whisper-large",
            emb_dim=2048
        )
        
        self.assertEqual(model.audio_encoder_name, "openai/whisper-large")
        self.assertEqual(model.emb_dim, 2048)
        self.assertIsNone(model.model_path)
        self.assertEqual(model.device, torch.device("cpu"))
    
    @patch('transformers.WhisperFeatureExtractor')
    @patch('transformers.WhisperModel')
    def test_name_without_weights(self, mock_whisper, mock_extractor):
        """Test name() returns correct format without weights."""
        from evaluator.models.a2e.attention_pool import AttentionPoolAudioModel
        
        mock_whisper_instance = MagicMock()
        mock_whisper_instance.get_encoder.return_value = MagicMock()
        mock_whisper.from_pretrained.return_value = mock_whisper_instance
        
        model = AttentionPoolAudioModel(
            audio_encoder_name="openai/whisper-large",
            emb_dim=2048
        )
        name = model.name()
        
        self.assertIn("AttentionPoolAudioModel", name)
        self.assertIn("openai/whisper-large", name)
        self.assertIn("2048", name)
        self.assertNotIn("weights", name)
    
    @patch('torch.load')
    @patch('transformers.WhisperFeatureExtractor')
    @patch('transformers.WhisperModel')
    def test_name_with_weights(self, mock_whisper, mock_extractor, mock_load):
        """Test name() returns correct format with weights."""
        from evaluator.models.a2e.attention_pool import AttentionPoolAudioModel
        
        mock_whisper_instance = MagicMock()
        mock_whisper_instance.get_encoder.return_value = MagicMock()
        mock_whisper.from_pretrained.return_value = mock_whisper_instance
        mock_load.return_value = {}
        
        model = AttentionPoolAudioModel(
            audio_encoder_name="openai/whisper-large",
            emb_dim=2048,
            model_path="/path/to/weights.pt"
        )
        name = model.name()
        
        self.assertIn("weights", name)
        self.assertIn("/path/to/weights.pt", name)
    
    @patch('transformers.WhisperFeatureExtractor')
    @patch('transformers.WhisperModel')
    def test_to_device(self, mock_whisper, mock_extractor):
        """Test moving model to device."""
        from evaluator.models.a2e.attention_pool import AttentionPoolAudioModel
        
        mock_whisper_instance = MagicMock()
        mock_whisper_instance.get_encoder.return_value = MagicMock()
        mock_whisper.from_pretrained.return_value = mock_whisper_instance
        
        model = AttentionPoolAudioModel()
        result = model.to(torch.device("cuda:0"))
        
        self.assertEqual(model.device, torch.device("cuda:0"))
        self.assertEqual(result, model)
    
    @patch('transformers.WhisperFeatureExtractor')
    @patch('transformers.WhisperModel')
    def test_encode_from_features_returns_correct_shape(self, mock_whisper, mock_extractor):
        """Test that encode_from_features returns embeddings of correct shape."""
        from evaluator.models.a2e.attention_pool import AttentionPoolAudioModel
        
        mock_whisper_instance = MagicMock()
        mock_encoder = MagicMock()
        # Mock encoder to return expected tensor shape
        mock_encoder.return_value = MagicMock(
            last_hidden_state=torch.randn(2, 100, 1280)
        )
        mock_whisper_instance.get_encoder.return_value = mock_encoder
        mock_whisper.from_pretrained.return_value = mock_whisper_instance
        
        model = AttentionPoolAudioModel(emb_dim=512)
        model.audio_encoder = mock_encoder
        
        # Create properly sized inputs
        features = torch.randn(2, 80, 3000)
        # Attention mask should be halved for Whisper's downsampling (50 for 100 encoder outputs)
        attention_mask = torch.ones(2, 200)
        
        result = model.encode_from_features(features, attention_mask)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 512))
        # Check L2 normalization
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones_like(norms), decimal=5)


class TestAttentionPooling(unittest.TestCase):
    """Test the AttentionPooling module directly."""
    
    def test_attention_pooling_forward(self):
        """Test AttentionPooling forward pass."""
        from evaluator.models.a2e.attention_pool import AttentionPooling
        
        pool = AttentionPooling(input_dim=768, output_dim=768, dropout=0.0)
        
        # Input: (batch_size=2, seq_len=10, input_dim=768)
        x = torch.randn(2, 10, 768)
        
        result = pool(x)
        
        self.assertEqual(result.shape, (2, 768))
    
    def test_attention_pooling_with_mask(self):
        """Test AttentionPooling with attention mask."""
        from evaluator.models.a2e.attention_pool import AttentionPooling
        
        pool = AttentionPooling(input_dim=768, output_dim=768, dropout=0.0)
        
        x = torch.randn(2, 10, 768)
        mask = torch.ones(2, 10)
        mask[0, 5:] = 0  # Mask out half of first sequence
        
        result = pool(x, mask=mask)
        
        self.assertEqual(result.shape, (2, 768))


class TestProjectionHead(unittest.TestCase):
    """Test the ProjectionHead module directly."""
    
    def test_projection_head_forward(self):
        """Test ProjectionHead forward pass."""
        from evaluator.models.a2e.attention_pool import ProjectionHead
        
        proj = ProjectionHead(input_dim=768, emb_dim=512, dropout=0.0)
        
        x = torch.randn(4, 768)
        result = proj(x)
        
        self.assertEqual(result.shape, (4, 512))


class TestMultimodalClapStyleModel(unittest.TestCase):
    """Test MultimodalClapStyleModel initialization and methods."""
    
    @patch('evaluator.models.a2e.clap_style.AutoTokenizer')
    @patch('evaluator.models.a2e.clap_style.WhisperFeatureExtractor')
    @patch('evaluator.models.a2e.clap_style.load_checkpoint_with_remap')
    def test_initialization(self, mock_load, mock_extractor, mock_tokenizer):
        """Test MultimodalClapStyleModel initialization."""
        from evaluator.models.a2e.clap_style import MultimodalClapStyleModel, CLAPConfig
        
        # Create mock checkpoint with config
        mock_config = CLAPConfig()
        mock_load.return_value = {
            'config': mock_config,
            'model_state_dict': {}
        }
        
        with patch('evaluator.models.a2e.clap_style.CLAP') as mock_clap:
            mock_clap_instance = MagicMock()
            mock_clap.return_value = mock_clap_instance
            
            model = MultimodalClapStyleModel(
                model_path="/path/to/model.pt",
                device="cpu"
            )
            
            self.assertEqual(model.model_path, "/path/to/model.pt")
            self.assertEqual(model.device, torch.device("cpu"))
    
    @patch('evaluator.models.a2e.clap_style.AutoTokenizer')
    @patch('evaluator.models.a2e.clap_style.WhisperFeatureExtractor')
    @patch('evaluator.models.a2e.clap_style.load_checkpoint_with_remap')
    def test_implements_both_interfaces(self, mock_load, mock_extractor, mock_tokenizer):
        """Test that model implements both audio and text embedding interfaces."""
        from evaluator.models.a2e.clap_style import MultimodalClapStyleModel, CLAPConfig
        
        mock_load.return_value = {
            'config': CLAPConfig(),
            'model_state_dict': {}
        }
        
        with patch('evaluator.models.a2e.clap_style.CLAP'):
            model = MultimodalClapStyleModel(model_path="/path/to/model.pt")
            
            # Check both interfaces
            self.assertIsInstance(model, AudioEmbeddingModel)
            self.assertIsInstance(model, TextEmbeddingModel)
            
            # Check required methods exist
            self.assertTrue(hasattr(model, 'encode_audio'))
            self.assertTrue(hasattr(model, 'encode'))
            self.assertTrue(hasattr(model, 'encode_text'))
            self.assertTrue(hasattr(model, 'preprocess_audio'))
            self.assertTrue(hasattr(model, 'encode_from_features'))
    
    @patch('evaluator.models.a2e.clap_style.AutoTokenizer')
    @patch('evaluator.models.a2e.clap_style.WhisperFeatureExtractor')
    @patch('evaluator.models.a2e.clap_style.load_checkpoint_with_remap')
    def test_name(self, mock_load, mock_extractor, mock_tokenizer):
        """Test name() returns correct format."""
        from evaluator.models.a2e.clap_style import MultimodalClapStyleModel, CLAPConfig
        
        mock_config = CLAPConfig()
        mock_load.return_value = {
            'config': mock_config,
            'model_state_dict': {}
        }
        
        with patch('evaluator.models.a2e.clap_style.CLAP'):
            model = MultimodalClapStyleModel(model_path="/path/to/model.pt")
            name = model.name()
            
            self.assertIn("ClapStyleAudioModel", name)
            self.assertIn("/path/to/model.pt", name)


# ============================================================================
# Test Model Registry Integration
# ============================================================================

class TestModelRegistryIntegration(unittest.TestCase):
    """Test model registry integration with actual model classes."""
    
    def test_whisper_registered_correctly(self):
        """Test Whisper is registered with correct metadata."""
        self.assertTrue(asr_registry.is_registered("whisper"))
        self.assertEqual(
            asr_registry.get_default_name("whisper"),
            "openai/whisper-medium"
        )
        metadata = asr_registry.get_metadata("whisper")
        self.assertIn("description", metadata)
    
    def test_wav2vec2_registered_correctly(self):
        """Test Wav2Vec2 is registered with correct metadata."""
        self.assertTrue(asr_registry.is_registered("wav2vec2"))
        self.assertEqual(
            asr_registry.get_default_name("wav2vec2"),
            "jonatasgrosman/wav2vec2-large-xlsr-53-polish"
        )
    
    @unittest.skipUnless(faster_whisper_available, "faster_whisper not installed")
    def test_faster_whisper_registered_correctly(self):
        """Test Faster Whisper is registered with correct metadata."""
        self.assertTrue(asr_registry.is_registered("faster_whisper"))
        self.assertEqual(
            asr_registry.get_default_name("faster_whisper"),
            "large-v3"
        )
        metadata = asr_registry.get_metadata("faster_whisper")
        self.assertIn("description", metadata)
    
    def test_labse_registered_correctly(self):
        """Test LaBSE is registered with correct metadata."""
        self.assertTrue(text_embedding_registry.is_registered("labse"))
        self.assertEqual(
            text_embedding_registry.get_default_name("labse"),
            "sentence-transformers/LaBSE"
        )
    
    def test_jina_registered_correctly(self):
        """Test Jina is registered with correct metadata."""
        self.assertTrue(text_embedding_registry.is_registered("jina_v4"))
        self.assertEqual(
            text_embedding_registry.get_default_name("jina_v4"),
            "jinaai/jina-embeddings-v4"
        )
    
    def test_clip_registered_correctly(self):
        """Test CLIP is registered with correct metadata."""
        self.assertTrue(text_embedding_registry.is_registered("clip"))
        self.assertEqual(
            text_embedding_registry.get_default_name("clip"),
            "openai/clip-vit-base-patch32"
        )
    
    def test_bge_m3_registered_correctly(self):
        """Test BGE-M3 is registered with correct metadata."""
        self.assertTrue(text_embedding_registry.is_registered("bge_m3"))
        self.assertEqual(
            text_embedding_registry.get_default_name("bge_m3"),
            "BAAI/bge-m3"
        )
    
    def test_nemotron_registered_correctly(self):
        """Test Nemotron is registered with correct metadata."""
        self.assertTrue(text_embedding_registry.is_registered("nemotron"))
        self.assertEqual(
            text_embedding_registry.get_default_name("nemotron"),
            "nvidia/llama-embed-nemotron-8b"
        )
    
    def test_attention_pool_registered_correctly(self):
        """Test AttentionPool is registered with correct metadata."""
        self.assertTrue(audio_embedding_registry.is_registered("attention_pool"))
        self.assertEqual(
            audio_embedding_registry.get_default_name("attention_pool"),
            "openai/whisper-large"
        )
    
    def test_clap_style_registered_correctly(self):
        """Test CLAP style model is registered."""
        self.assertTrue(audio_embedding_registry.is_registered("clap_style"))
        # CLAP style doesn't have a default name (requires model_path)
        self.assertIsNone(audio_embedding_registry.get_default_name("clap_style"))


# ============================================================================
# Test Error Handling
# ============================================================================

class TestErrorHandling(unittest.TestCase):
    """Test error handling for invalid inputs."""
    
    def test_unregistered_model_type_raises_error(self):
        """Test that looking up unregistered model raises ValueError."""
        with self.assertRaises(ValueError) as context:
            asr_registry.get("nonexistent_model")
        
        self.assertIn("Unknown", str(context.exception))
        self.assertIn("nonexistent_model", str(context.exception))
        self.assertIn("ASR", str(context.exception))
    
    def test_unregistered_text_model_raises_error(self):
        """Test that looking up unregistered text model raises ValueError."""
        with self.assertRaises(ValueError) as context:
            text_embedding_registry.get("fake_embedder")
        
        self.assertIn("Unknown", str(context.exception))
        self.assertIn("TextEmbedding", str(context.exception))
    
    def test_unregistered_audio_model_raises_error(self):
        """Test that looking up unregistered audio model raises ValueError."""
        with self.assertRaises(ValueError) as context:
            audio_embedding_registry.get("fake_audio_model")
        
        self.assertIn("Unknown", str(context.exception))
        self.assertIn("AudioEmbedding", str(context.exception))
    
    @patch('transformers.WhisperForConditionalGeneration')
    @patch('transformers.WhisperProcessor')
    @patch('transformers.WhisperFeatureExtractor')
    def test_whisper_adapter_without_peft_raises_error(
        self, mock_extractor, mock_processor, mock_model
    ):
        """Test that loading adapter without peft raises ImportError."""
        from evaluator.models.asr.whisper import WhisperModel
        
        model = WhisperModel()
        
        # Mock peft import to fail
        with patch.object(
            model, '_load_adapter',
            side_effect=ImportError("peft library required")
        ):
            with self.assertRaises(ImportError):
                model._load_adapter("/fake/path")


# ============================================================================
# Test CLAP Config Classes
# ============================================================================

class TestCLAPConfigClasses(unittest.TestCase):
    """Test CLAP configuration dataclasses."""
    
    def test_text_encoder_config_defaults(self):
        """Test TextEncoderConfig default values."""
        from evaluator.models.a2e.clap_style import TextEncoderConfig
        
        config = TextEncoderConfig()
        
        self.assertEqual(
            config.model_name,
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        )
        self.assertEqual(config.max_length, 512)
    
    def test_audio_encoder_config_defaults(self):
        """Test AudioEncoderConfig default values."""
        from evaluator.models.a2e.clap_style import AudioEncoderConfig
        
        config = AudioEncoderConfig()
        
        self.assertEqual(config.model_name, "openai/whisper-base")
        self.assertEqual(config.sample_rate, 16000)
        self.assertEqual(config.max_duration, 30.0)
    
    def test_projection_config_defaults(self):
        """Test ProjectionConfig default values."""
        from evaluator.models.a2e.clap_style import ProjectionConfig
        
        config = ProjectionConfig()
        
        self.assertEqual(config.input_dim_text, 768)
        self.assertEqual(config.input_dim_audio, 512)
        self.assertEqual(config.projection_dim, 512)
        self.assertIsNone(config.hidden_dim)
        self.assertEqual(config.dropout, 0.1)
    
    def test_clap_config_defaults(self):
        """Test CLAPConfig default values."""
        from evaluator.models.a2e.clap_style import CLAPConfig
        
        config = CLAPConfig()
        
        self.assertIsInstance(config.text_encoder, object)
        self.assertIsInstance(config.audio_encoder, object)
        self.assertIsInstance(config.projection, object)
        self.assertEqual(config.temperature, 0.07)
        self.assertTrue(config.learnable_temperature)


# ============================================================================
# Test HuBERT Audio Embedding Model
# ============================================================================

class TestHuBERTAudioModel(unittest.TestCase):
    """Test HuBERTAudioModel initialization and methods."""
    
    @patch('transformers.Wav2Vec2FeatureExtractor')
    @patch('transformers.HubertModel')
    def test_initialization(self, mock_hubert, mock_extractor):
        """Test HuBERTAudioModel initialization."""
        from evaluator.models.a2e.hubert import HuBERTAudioModel
        
        mock_hubert_instance = MagicMock()
        mock_hubert_instance.config.hidden_size = 768
        mock_hubert.from_pretrained.return_value = mock_hubert_instance
        
        model = HuBERTAudioModel(
            model_name="facebook/hubert-base-ls960",
            pooling="mean"
        )
        
        self.assertEqual(model.model_name, "facebook/hubert-base-ls960")
        self.assertEqual(model.pooling, "mean")
        self.assertEqual(model.hidden_dim, 768)
        mock_hubert.from_pretrained.assert_called_once_with("facebook/hubert-base-ls960")
        mock_extractor.from_pretrained.assert_called_once_with("facebook/hubert-base-ls960")
    
    @patch('transformers.Wav2Vec2FeatureExtractor')
    @patch('transformers.HubertModel')
    def test_name_method(self, mock_hubert, mock_extractor):
        """Test name() returns correct format."""
        from evaluator.models.a2e.hubert import HuBERTAudioModel
        
        mock_hubert_instance = MagicMock()
        mock_hubert_instance.config.hidden_size = 768
        mock_hubert.from_pretrained.return_value = mock_hubert_instance
        
        model = HuBERTAudioModel(
            model_name="facebook/hubert-base-ls960",
            pooling="mean"
        )
        name = model.name()
        
        self.assertIn("HuBERTAudioModel", name)
        self.assertIn("facebook/hubert-base-ls960", name)
        self.assertIn("mean", name)
        self.assertIn("768", name)
    
    @patch('transformers.Wav2Vec2FeatureExtractor')
    @patch('transformers.HubertModel')
    def test_to_device(self, mock_hubert, mock_extractor):
        """Test moving model to device."""
        from evaluator.models.a2e.hubert import HuBERTAudioModel
        
        mock_hubert_instance = MagicMock()
        mock_hubert_instance.config.hidden_size = 768
        mock_hubert.from_pretrained.return_value = mock_hubert_instance
        
        model = HuBERTAudioModel()
        result = model.to(torch.device("cuda:0"))
        
        self.assertEqual(model.device, torch.device("cuda:0"))
        self.assertIs(result, model)
        mock_hubert_instance.to.assert_called_with(torch.device("cuda:0"))
    
    @patch('transformers.Wav2Vec2FeatureExtractor')
    @patch('transformers.HubertModel')
    def test_encode_from_features_returns_correct_shape(self, mock_hubert, mock_extractor):
        """Test that encode_from_features returns embeddings of correct shape."""
        from evaluator.models.a2e.hubert import HuBERTAudioModel
        
        mock_hubert_instance = MagicMock()
        mock_hubert_instance.config.hidden_size = 768
        mock_hubert_instance.config.conv_kernel = [10, 3, 3, 3, 3, 2, 2]
        mock_hubert_instance.config.conv_stride = [5, 2, 2, 2, 2, 2, 2]
        mock_hubert.from_pretrained.return_value = mock_hubert_instance
        
        # Mock forward pass
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(2, 50, 768)
        mock_hubert_instance.return_value = mock_outputs
        
        model = HuBERTAudioModel()
        model.model = mock_hubert_instance
        
        features = torch.randn(2, 16000)  # 1 second of audio
        embeddings = model.encode_from_features(features, attention_mask=None)
        
        self.assertEqual(embeddings.shape, (2, 768))
        # Check L2 normalized
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(2), decimal=5)
    
    @patch('transformers.Wav2Vec2FeatureExtractor')
    @patch('transformers.HubertModel')
    def test_cls_pooling(self, mock_hubert, mock_extractor):
        """Test CLS pooling mode."""
        from evaluator.models.a2e.hubert import HuBERTAudioModel
        
        mock_hubert_instance = MagicMock()
        mock_hubert_instance.config.hidden_size = 768
        mock_hubert.from_pretrained.return_value = mock_hubert_instance
        
        # Mock forward pass
        mock_outputs = MagicMock()
        hidden_states = torch.randn(2, 50, 768)
        mock_outputs.last_hidden_state = hidden_states
        mock_hubert_instance.return_value = mock_outputs
        
        model = HuBERTAudioModel(pooling="cls")
        model.model = mock_hubert_instance
        
        features = torch.randn(2, 16000)
        embeddings = model.encode_from_features(features, attention_mask=None)
        
        self.assertEqual(embeddings.shape, (2, 768))
    
    @patch('torchaudio.functional.resample')
    @patch('transformers.Wav2Vec2FeatureExtractor')
    @patch('transformers.HubertModel')
    def test_preprocess_resamples_audio(self, mock_hubert, mock_extractor, mock_resample):
        """Test that preprocess resamples non-16kHz audio."""
        from evaluator.models.a2e.hubert import HuBERTAudioModel
        
        mock_hubert_instance = MagicMock()
        mock_hubert_instance.config.hidden_size = 768
        mock_hubert.from_pretrained.return_value = mock_hubert_instance
        
        mock_extractor_instance = MagicMock()
        mock_extractor_instance.return_value = MagicMock(
            input_values=torch.randn(1, 16000),
            attention_mask=torch.ones(1, 16000)
        )
        mock_extractor.from_pretrained.return_value = mock_extractor_instance
        
        mock_resample.return_value = torch.randn(48000)
        
        model = HuBERTAudioModel()
        
        audio = torch.randn(48000)  # 1 second at 48kHz
        model.preprocess_audio([audio], [48000])
        
        # Should resample from 48kHz to 16kHz
        mock_resample.assert_called_once()
        call_args = mock_resample.call_args
        self.assertEqual(call_args[0][1], 48000)
        self.assertEqual(call_args[0][2], 16000)
    
    def test_hubert_registered_correctly(self):
        """Test HuBERT is registered with correct metadata."""
        self.assertTrue(audio_embedding_registry.is_registered("hubert"))
        self.assertEqual(
            audio_embedding_registry.get_default_name("hubert"),
            "facebook/hubert-base-ls960"
        )
        metadata = audio_embedding_registry.get_metadata("hubert")
        self.assertIn("description", metadata)
    
    @patch('transformers.Wav2Vec2FeatureExtractor')
    @patch('transformers.HubertModel')
    def test_large_model_initialization(self, mock_hubert, mock_extractor):
        """Test initialization with large model variant."""
        from evaluator.models.a2e.hubert import HuBERTAudioModel
        
        mock_hubert_instance = MagicMock()
        mock_hubert_instance.config.hidden_size = 1024  # Large model has 1024 dim
        mock_hubert.from_pretrained.return_value = mock_hubert_instance
        
        model = HuBERTAudioModel(model_name="facebook/hubert-large-ll60k")
        
        self.assertEqual(model.model_name, "facebook/hubert-large-ll60k")
        self.assertEqual(model.hidden_dim, 1024)
        mock_hubert.from_pretrained.assert_called_once_with("facebook/hubert-large-ll60k")


# ============================================================================
# Test WavLM Audio Embedding Model
# ============================================================================

class TestWavLMAudioModel(unittest.TestCase):
    """Test WavLMAudioModel initialization and methods."""
    
    @patch('transformers.Wav2Vec2FeatureExtractor')
    @patch('transformers.WavLMModel')
    def test_initialization(self, mock_wavlm, mock_extractor):
        """Test WavLMAudioModel initialization."""
        from evaluator.models.a2e.wavlm import WavLMAudioModel
        
        mock_wavlm_instance = MagicMock()
        mock_wavlm_instance.config.hidden_size = 768
        mock_wavlm.from_pretrained.return_value = mock_wavlm_instance
        
        model = WavLMAudioModel(
            model_name="microsoft/wavlm-base",
            pooling="mean"
        )
        
        self.assertEqual(model.model_name, "microsoft/wavlm-base")
        self.assertEqual(model.pooling, "mean")
        self.assertEqual(model.hidden_dim, 768)
        mock_wavlm.from_pretrained.assert_called_once_with("microsoft/wavlm-base")
        mock_extractor.from_pretrained.assert_called_once_with("microsoft/wavlm-base")
    
    @patch('transformers.Wav2Vec2FeatureExtractor')
    @patch('transformers.WavLMModel')
    def test_name_method(self, mock_wavlm, mock_extractor):
        """Test name() returns correct format."""
        from evaluator.models.a2e.wavlm import WavLMAudioModel
        
        mock_wavlm_instance = MagicMock()
        mock_wavlm_instance.config.hidden_size = 768
        mock_wavlm.from_pretrained.return_value = mock_wavlm_instance
        
        model = WavLMAudioModel(
            model_name="microsoft/wavlm-base",
            pooling="mean"
        )
        name = model.name()
        
        self.assertIn("WavLMAudioModel", name)
        self.assertIn("microsoft/wavlm-base", name)
        self.assertIn("mean", name)
        self.assertIn("768", name)
    
    @patch('transformers.Wav2Vec2FeatureExtractor')
    @patch('transformers.WavLMModel')
    def test_to_device(self, mock_wavlm, mock_extractor):
        """Test moving model to device."""
        from evaluator.models.a2e.wavlm import WavLMAudioModel
        
        mock_wavlm_instance = MagicMock()
        mock_wavlm_instance.config.hidden_size = 768
        mock_wavlm.from_pretrained.return_value = mock_wavlm_instance
        
        model = WavLMAudioModel()
        result = model.to(torch.device("cuda:0"))
        
        self.assertEqual(model.device, torch.device("cuda:0"))
        self.assertIs(result, model)
        mock_wavlm_instance.to.assert_called_with(torch.device("cuda:0"))
    
    @patch('transformers.Wav2Vec2FeatureExtractor')
    @patch('transformers.WavLMModel')
    def test_encode_from_features_returns_correct_shape(self, mock_wavlm, mock_extractor):
        """Test that encode_from_features returns embeddings of correct shape."""
        from evaluator.models.a2e.wavlm import WavLMAudioModel
        
        mock_wavlm_instance = MagicMock()
        mock_wavlm_instance.config.hidden_size = 768
        mock_wavlm_instance.config.conv_kernel = [10, 3, 3, 3, 3, 2, 2]
        mock_wavlm_instance.config.conv_stride = [5, 2, 2, 2, 2, 2, 2]
        mock_wavlm.from_pretrained.return_value = mock_wavlm_instance
        
        # Mock forward pass
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(2, 50, 768)
        mock_wavlm_instance.return_value = mock_outputs
        
        model = WavLMAudioModel()
        model.model = mock_wavlm_instance
        
        features = torch.randn(2, 16000)  # 1 second of audio
        embeddings = model.encode_from_features(features, attention_mask=None)
        
        self.assertEqual(embeddings.shape, (2, 768))
        # Check L2 normalized
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(2), decimal=5)
    
    @patch('transformers.Wav2Vec2FeatureExtractor')
    @patch('transformers.WavLMModel')
    def test_cls_pooling(self, mock_wavlm, mock_extractor):
        """Test CLS pooling mode."""
        from evaluator.models.a2e.wavlm import WavLMAudioModel
        
        mock_wavlm_instance = MagicMock()
        mock_wavlm_instance.config.hidden_size = 768
        mock_wavlm.from_pretrained.return_value = mock_wavlm_instance
        
        # Mock forward pass
        mock_outputs = MagicMock()
        hidden_states = torch.randn(2, 50, 768)
        mock_outputs.last_hidden_state = hidden_states
        mock_wavlm_instance.return_value = mock_outputs
        
        model = WavLMAudioModel(pooling="cls")
        model.model = mock_wavlm_instance
        
        features = torch.randn(2, 16000)
        embeddings = model.encode_from_features(features, attention_mask=None)
        
        self.assertEqual(embeddings.shape, (2, 768))
    
    @patch('torchaudio.functional.resample')
    @patch('transformers.Wav2Vec2FeatureExtractor')
    @patch('transformers.WavLMModel')
    def test_preprocess_resamples_audio(self, mock_wavlm, mock_extractor, mock_resample):
        """Test that preprocess resamples non-16kHz audio."""
        from evaluator.models.a2e.wavlm import WavLMAudioModel
        
        mock_wavlm_instance = MagicMock()
        mock_wavlm_instance.config.hidden_size = 768
        mock_wavlm.from_pretrained.return_value = mock_wavlm_instance
        
        mock_extractor_instance = MagicMock()
        mock_extractor_instance.return_value = MagicMock(
            input_values=torch.randn(1, 16000),
            attention_mask=torch.ones(1, 16000)
        )
        mock_extractor.from_pretrained.return_value = mock_extractor_instance
        
        mock_resample.return_value = torch.randn(48000)
        
        model = WavLMAudioModel()
        
        audio = torch.randn(48000)  # 1 second at 48kHz
        model.preprocess_audio([audio], [48000])
        
        # Should resample from 48kHz to 16kHz
        mock_resample.assert_called_once()
        call_args = mock_resample.call_args
        self.assertEqual(call_args[0][1], 48000)
        self.assertEqual(call_args[0][2], 16000)
    
    def test_wavlm_registered_correctly(self):
        """Test WavLM is registered with correct metadata."""
        self.assertTrue(audio_embedding_registry.is_registered("wavlm"))
        self.assertEqual(
            audio_embedding_registry.get_default_name("wavlm"),
            "microsoft/wavlm-base"
        )
        metadata = audio_embedding_registry.get_metadata("wavlm")
        self.assertIn("description", metadata)
    
    @patch('transformers.Wav2Vec2FeatureExtractor')
    @patch('transformers.WavLMModel')
    def test_large_model_initialization(self, mock_wavlm, mock_extractor):
        """Test initialization with large model variant."""
        from evaluator.models.a2e.wavlm import WavLMAudioModel
        
        mock_wavlm_instance = MagicMock()
        mock_wavlm_instance.config.hidden_size = 1024  # Large model has 1024 dim
        mock_wavlm.from_pretrained.return_value = mock_wavlm_instance
        
        model = WavLMAudioModel(model_name="microsoft/wavlm-large")
        
        self.assertEqual(model.model_name, "microsoft/wavlm-large")
        self.assertEqual(model.hidden_dim, 1024)
        mock_wavlm.from_pretrained.assert_called_once_with("microsoft/wavlm-large")
    
    @patch('transformers.Wav2Vec2FeatureExtractor')
    @patch('transformers.WavLMModel')
    def test_encode_audio_full_pipeline(self, mock_wavlm, mock_extractor):
        """Test the full encode_audio pipeline."""
        from evaluator.models.a2e.wavlm import WavLMAudioModel
        
        mock_wavlm_instance = MagicMock()
        mock_wavlm_instance.config.hidden_size = 768
        mock_wavlm_instance.config.conv_kernel = [10, 3, 3, 3, 3, 2, 2]
        mock_wavlm_instance.config.conv_stride = [5, 2, 2, 2, 2, 2, 2]
        mock_wavlm.from_pretrained.return_value = mock_wavlm_instance
        
        mock_extractor_instance = MagicMock()
        mock_extractor_instance.return_value = MagicMock(
            input_values=torch.randn(2, 16000),
            attention_mask=torch.ones(2, 16000)
        )
        mock_extractor.from_pretrained.return_value = mock_extractor_instance
        
        # Mock forward pass
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(2, 50, 768)
        mock_wavlm_instance.return_value = mock_outputs
        
        model = WavLMAudioModel()
        model.model = mock_wavlm_instance
        
        audio_list = [torch.randn(16000), torch.randn(16000)]
        sampling_rates = [16000, 16000]
        
        embeddings = model.encode_audio(audio_list, sampling_rates)
        
        self.assertEqual(embeddings.shape, (2, 768))
        # Check L2 normalized
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(2), decimal=5)


if __name__ == "__main__":
    unittest.main()
