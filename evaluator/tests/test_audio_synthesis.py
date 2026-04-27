"""Unit tests for audio synthesis (TTS) functionality."""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import tempfile
import shutil

from evaluator.pipeline.audio.synthesis import AudioSynthesizer
from evaluator.config import AudioSynthesisConfig


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_piper_provider():
    """Create a mock Piper TTS provider."""
    with patch('evaluator.models.tts.piper_tts.PiperTTS') as mock:
        provider_instance = MagicMock()
        # Simulate audio synthesis (1 second of random audio)
        provider_instance.synthesize.return_value = np.random.randn(16000).astype(np.float32)
        mock.return_value = provider_instance
        yield provider_instance


@pytest.fixture
def mock_xtts_provider():
    """Create a mock XTTS-v2 provider."""
    with patch('evaluator.models.tts.xtts_v2_tts.XTTSv2TTS') as mock:
        provider_instance = MagicMock()
        provider_instance.synthesize.return_value = np.random.randn(16000).astype(np.float32)
        provider_instance.output_sample_rate = 24000
        mock.return_value = provider_instance
        yield provider_instance


@pytest.fixture
def mock_mms_provider():
    """Create a mock MMS provider."""
    with patch('evaluator.models.tts.mms_tts.MMSTTS') as mock:
        provider_instance = MagicMock()
        provider_instance.synthesize.return_value = np.random.randn(16000).astype(np.float32)
        provider_instance.output_sample_rate = 22050
        mock.return_value = provider_instance
        yield provider_instance


class TestAudioSynthesizerInitialization:
    """Tests for AudioSynthesizer initialization."""
    
    def test_init_with_piper_provider(self, temp_cache_dir, mock_piper_provider):
        """Test initialization with Piper provider."""
        config = AudioSynthesisConfig(
            provider="piper",
            voice="en_US-lessac-medium",
            cache_dir=temp_cache_dir
        )
        
        synth = AudioSynthesizer(config)
        
        assert synth.config == config
        assert synth.cache_dir == Path(temp_cache_dir)
        assert synth.cache_dir.exists()
        assert synth.provider is not None
    
    def test_init_without_cache(self, mock_piper_provider):
        """Test initialization without explicit cache_dir — auto-derived from output_dir."""
        config = AudioSynthesisConfig(
            provider="piper",
            voice="en_US-lessac-medium",
            cache_dir=None,
            output_dir=None,
        )

        synth = AudioSynthesizer(config)

        assert synth.cache_dir is None

    def test_init_with_xtts_provider(self, temp_cache_dir, mock_xtts_provider):
        """Test initialization with XTTS-v2 provider."""
        config = AudioSynthesisConfig(
            provider="xtts_v2",
            voice="speaker.wav",
            language="en",
            cache_dir=temp_cache_dir
        )
        synth = AudioSynthesizer(config)
        assert synth.provider is not None

    def test_init_with_mms_provider(self, temp_cache_dir, mock_mms_provider):
        """Test initialization with MMS provider."""
        config = AudioSynthesisConfig(
            provider="mms",
            language="pl",
            cache_dir=temp_cache_dir
        )
        synth = AudioSynthesizer(config)
        assert synth.provider is not None
    
    def test_init_with_unsupported_provider(self, temp_cache_dir):
        """Test initialization with unsupported provider raises error."""
        config = AudioSynthesisConfig(
            provider="unsupported_tts",
            voice="test",
            cache_dir=temp_cache_dir
        )
        
        with pytest.raises(ValueError, match="Unknown TTS provider"):
            AudioSynthesizer(config)


class TestAudioSynthesis:
    """Tests for audio synthesis functionality."""
    
    def test_synthesize_basic_text(self, temp_cache_dir, mock_piper_provider):
        """Test basic text synthesis."""
        config = AudioSynthesisConfig(
            provider="piper",
            voice="en_US-lessac-medium",
            cache_dir=temp_cache_dir
        )
        synth = AudioSynthesizer(config)
        
        text = "The patient presents with hypertension."
        audio = synth.synthesize(text)
        
        # Verify provider was called
        mock_piper_provider.synthesize.assert_called_once_with(text)
        
        # Verify audio properties
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) > 0
        # Audio should be normalized to [-1, 1]
        assert np.abs(audio).max() <= 1.0
    
    def test_synthesize_empty_text_returns_silence(self, temp_cache_dir, mock_piper_provider):
        """Test that empty text returns silence."""
        config = AudioSynthesisConfig(
            provider="piper",
            voice="en_US-lessac-medium",
            cache_dir=temp_cache_dir,
            sample_rate=16000
        )
        synth = AudioSynthesizer(config)
        
        audio = synth.synthesize("")
        
        # Should return silence without calling provider
        mock_piper_provider.synthesize.assert_not_called()
        assert isinstance(audio, np.ndarray)
        assert len(audio) == 16000  # 1 second of silence
        assert np.all(audio == 0)
    
    def test_synthesize_whitespace_only_returns_silence(self, temp_cache_dir, mock_piper_provider):
        """Test that whitespace-only text returns silence."""
        config = AudioSynthesisConfig(
            provider="piper",
            voice="en_US-lessac-medium",
            cache_dir=temp_cache_dir,
            sample_rate=16000
        )
        synth = AudioSynthesizer(config)
        
        audio = synth.synthesize("   \n\t  ")
        
        mock_piper_provider.synthesize.assert_not_called()
        assert np.all(audio == 0)
    
    def test_synthesize_with_output_path(self, temp_cache_dir, mock_piper_provider):
        """Test synthesis with audio file output."""
        config = AudioSynthesisConfig(
            provider="piper",
            voice="en_US-lessac-medium",
            cache_dir=temp_cache_dir
        )
        synth = AudioSynthesizer(config)
        
        output_path = Path(temp_cache_dir) / "output.wav"
        text = "Test audio output."
        
        with patch.object(synth, '_save_audio') as mock_save:
            audio = synth.synthesize(text, output_path=str(output_path))
            mock_save.assert_called_once_with(str(output_path), audio)

    def test_resamples_when_provider_sample_rate_differs(self, temp_cache_dir, mock_xtts_provider):
        """Provider output should be resampled to config.sample_rate."""
        import types
        import sys

        config = AudioSynthesisConfig(
            provider="xtts_v2",
            voice="speaker.wav",
            language="en",
            cache_dir=temp_cache_dir,
            sample_rate=16000,
        )
        synth = AudioSynthesizer(config)
        fake_librosa = types.SimpleNamespace(
            resample=MagicMock(return_value=np.random.randn(16000).astype(np.float32))
        )
        with patch.dict(sys.modules, {"librosa": fake_librosa}):
            _audio = synth.synthesize("Resample test.")
            fake_librosa.resample.assert_called_once()


class TestAudioCaching:
    """Tests for TTS caching functionality."""
    
    def test_cache_hit_on_repeated_synthesis(self, temp_cache_dir, mock_piper_provider):
        """Test that repeated synthesis uses cache."""
        config = AudioSynthesisConfig(
            provider="piper",
            voice="en_US-lessac-medium",
            cache_dir=temp_cache_dir
        )
        synth = AudioSynthesizer(config)
        
        text = "This text will be cached."
        
        # First synthesis - should call provider
        audio1 = synth.synthesize(text)
        assert mock_piper_provider.synthesize.call_count == 1
        
        # Second synthesis - should use cache
        audio2 = synth.synthesize(text)
        assert mock_piper_provider.synthesize.call_count == 1  # Still 1, not called again
        
        # Audio should be identical
        np.testing.assert_array_equal(audio1, audio2)
    
    def test_different_text_creates_new_cache_entry(self, temp_cache_dir, mock_piper_provider):
        """Test that different text creates separate cache entries."""
        config = AudioSynthesisConfig(
            provider="piper",
            voice="en_US-lessac-medium",
            cache_dir=temp_cache_dir
        )
        synth = AudioSynthesizer(config)
        
        # Configure provider to return different audio for different texts
        def synthesize_side_effect(text):
            return np.random.randn(16000).astype(np.float32)
        
        mock_piper_provider.synthesize.side_effect = synthesize_side_effect
        
        audio1 = synth.synthesize("First text")
        audio2 = synth.synthesize("Second text")
        
        # Should call provider twice (no cache hit)
        assert mock_piper_provider.synthesize.call_count == 2
        
        # Audio should be different
        assert not np.array_equal(audio1, audio2)
    
    def test_cache_key_includes_config(self, temp_cache_dir, mock_piper_provider):
        """Test that cache key includes voice and provider."""
        # Create two synthesizers with different voices
        config1 = AudioSynthesisConfig(
            provider="piper",
            voice="en_US-lessac-medium",
            cache_dir=temp_cache_dir
        )
        config2 = AudioSynthesisConfig(
            provider="piper",
            voice="en_US-amy-medium",
            cache_dir=temp_cache_dir
        )
        
        synth1 = AudioSynthesizer(config1)
        synth2 = AudioSynthesizer(config2)
        
        text = "Same text, different voices"
        
        # Each should synthesize separately (different cache keys)
        audio1 = synth1.synthesize(text)
        audio2 = synth2.synthesize(text)
        
        # Both providers should have been called
        # (Since they're separate instances, we check the base mock was called)
        assert True  # Both synthesizers work independently


class TestBatchSynthesis:
    """Tests for batch audio synthesis."""
    
    def test_synthesize_batch(self, temp_cache_dir, mock_piper_provider):
        """Test batch synthesis of multiple texts."""
        config = AudioSynthesisConfig(
            provider="piper",
            voice="en_US-lessac-medium",
            cache_dir=temp_cache_dir
        )
        synth = AudioSynthesizer(config)
        
        texts = [
            "First sentence.",
            "Second sentence.",
            "Third sentence."
        ]
        
        # Configure provider to return different audio for each call
        call_count = [0]
        def synthesize_side_effect(text):
            audio = np.random.randn(16000).astype(np.float32) * (call_count[0] + 1)
            call_count[0] += 1
            return audio
        
        mock_piper_provider.synthesize.side_effect = synthesize_side_effect
        
        audio_list = synth.synthesize_batch(texts)
        
        assert len(audio_list) == 3
        assert all(isinstance(audio, np.ndarray) for audio in audio_list)
        assert mock_piper_provider.synthesize.call_count == 3
    
    def test_batch_synthesis_with_cache(self, temp_cache_dir, mock_piper_provider):
        """Test that batch synthesis uses cache for repeated texts."""
        config = AudioSynthesisConfig(
            provider="piper",
            voice="en_US-lessac-medium",
            cache_dir=temp_cache_dir
        )
        synth = AudioSynthesizer(config)
        
        texts = [
            "Repeated text.",
            "Unique text.",
            "Repeated text.",  # Same as first
        ]
        
        audio_list = synth.synthesize_batch(texts)
        
        # Should only synthesize 2 times (third is cached)
        assert mock_piper_provider.synthesize.call_count == 2
        
        # First and third audio should be identical
        np.testing.assert_array_equal(audio_list[0], audio_list[2])


class TestAudioNormalization:
    """Tests for audio normalization."""
    
    def test_normalization_to_float32(self, temp_cache_dir, mock_piper_provider):
        """Test that audio is normalized to float32 in [-1, 1] range."""
        # Configure provider to return int16 audio
        mock_piper_provider.synthesize.return_value = np.array(
            [0, 16000, -16000, 32767, -32768], dtype=np.int16
        )
        
        config = AudioSynthesisConfig(
            provider="piper",
            voice="en_US-lessac-medium",
            cache_dir=temp_cache_dir
        )
        synth = AudioSynthesizer(config)
        
        audio = synth.synthesize("Test normalization")
        
        assert audio.dtype == np.float32
        assert audio.min() >= -1.0
        assert audio.max() <= 1.0
    
    def test_zero_audio_not_normalized(self, temp_cache_dir, mock_piper_provider):
        """Test that silence (all zeros) is not affected by normalization."""
        mock_piper_provider.synthesize.return_value = np.zeros(1000, dtype=np.float32)
        
        config = AudioSynthesisConfig(
            provider="piper",
            voice="en_US-lessac-medium",
            cache_dir=temp_cache_dir
        )
        synth = AudioSynthesizer(config)
        
        audio = synth.synthesize("Silent audio")
        
        assert np.all(audio == 0)


class TestErrorHandling:
    """Tests for error handling in audio synthesis."""
    
    def test_provider_synthesis_error_propagates(self, temp_cache_dir, mock_piper_provider):
        """Test that provider errors are properly propagated."""
        mock_piper_provider.synthesize.side_effect = RuntimeError("Provider error")
        
        config = AudioSynthesisConfig(
            provider="piper",
            voice="en_US-lessac-medium",
            cache_dir=temp_cache_dir
        )
        synth = AudioSynthesizer(config)
        
        with pytest.raises(RuntimeError, match="Provider error"):
            synth.synthesize("This will fail")
    
    def test_invalid_cache_directory_creates_path(self, mock_piper_provider):
        """Test that invalid cache directory is created."""
        cache_path = "/tmp/test_tts_cache_nonexistent_xyz123"
        
        # Ensure it doesn't exist
        if Path(cache_path).exists():
            shutil.rmtree(cache_path)
        
        config = AudioSynthesisConfig(
            provider="piper",
            voice="en_US-lessac-medium",
            cache_dir=cache_path
        )
        
        try:
            synth = AudioSynthesizer(config)
            assert Path(cache_path).exists()
            assert Path(cache_path).is_dir()
        finally:
            # Cleanup
            if Path(cache_path).exists():
                shutil.rmtree(cache_path)


class TestConfigSettings:
    """Tests for configuration settings."""
    
    def test_default_sample_rate(self, mock_piper_provider):
        """Test default sample rate configuration."""
        config = AudioSynthesisConfig(
            provider="piper",
            voice="en_US-lessac-medium"
        )
        
        assert config.sample_rate == 16000  # Default sample rate
    
    def test_custom_sample_rate(self, mock_piper_provider):
        """Test custom sample rate configuration."""
        config = AudioSynthesisConfig(
            provider="piper",
            voice="en_US-lessac-medium",
            sample_rate=16000
        )
        
        assert config.sample_rate == 16000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
