"""Piper TTS provider - fast, local, multi-language text-to-speech."""
import subprocess
import numpy as np
import tempfile
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PiperTTS:
    """Piper TTS provider - fast, local, multi-language."""
    
    def __init__(self, config):
        """
        Initialize Piper TTS provider.
        
        Args:
            config: AudioSynthesisConfig instance.
        """
        self.config = config
        self._check_installation()
        self.voice_path = self._get_voice_path()
        logger.info(f"Piper TTS initialized with voice: {config.voice}")
    
    def _check_installation(self):
        """Check if piper is installed and accessible."""
        try:
            result = subprocess.run(
                ["piper", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            logger.debug(f"Piper version: {result.stdout.strip()}")
        except FileNotFoundError:
            raise RuntimeError(
                "Piper TTS not found. Install with:\n"
                "  pip install piper-tts\n"
                "Or download from: https://github.com/rhasspy/piper/releases"
            )
        except subprocess.TimeoutExpired:
            logger.warning("Piper version check timed out")
        except Exception as e:
            logger.warning(f"Could not check Piper version: {e}")
    
    def _get_voice_path(self) -> str:
        """Get path to voice model file."""
        # Try common installation locations
        voice_dirs = [
            Path.home() / ".local/share/piper/voices",
            Path("/usr/share/piper/voices"),
            Path("/usr/local/share/piper/voices"),
            Path("./voices"),
            Path("./piper_voices"),
        ]
        
        # Try with and without .onnx extension
        voice_name = self.config.voice
        if not voice_name.endswith('.onnx'):
            voice_name += '.onnx'
        
        for voice_dir in voice_dirs:
            if not voice_dir.exists():
                continue
            
            voice_path = voice_dir / voice_name
            if voice_path.exists():
                logger.debug(f"Found voice at: {voice_path}")
                return str(voice_path)
        
        # Voice not found - provide helpful error message
        searched_paths = [str(d / voice_name) for d in voice_dirs if d.exists()]
        raise FileNotFoundError(
            f"Voice model not found: {self.config.voice}\n"
            f"Searched in:\n" + "\n".join(f"  - {p}" for p in searched_paths) + "\n"
            f"\nDownload voices from: https://github.com/rhasspy/piper/releases\n"
            f"Example voices:\n"
            f"  - en_US-lessac-medium (English, general purpose)\n"
            f"  - en_US-amy-medium (English, female)\n"
            f"  - en_GB-alba-medium (British English)\n"
        )
    
    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize audio using piper command-line tool.
        
        Args:
            text: Text to synthesize.
            
        Returns:
            audio: Float32 numpy array with audio waveform.
        """
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name
        
        try:
            # Build piper command
            cmd = [
                "piper",
                "--model", self.voice_path,
                "--output_file", output_path,
            ]
            
            # Add optional parameters
            if self.config.speed != 1.0:
                # Piper uses length_scale (inverse of speed)
                length_scale = 1.0 / self.config.speed
                cmd.extend(["--length_scale", str(length_scale)])
            
            # Run synthesis
            logger.debug(f"Running Piper: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                input=text.encode('utf-8'),
                capture_output=True,
                timeout=30,
                check=True
            )
            
            if result.stderr:
                logger.debug(f"Piper stderr: {result.stderr.decode('utf-8', errors='ignore')}")
            
            # Load synthesized audio
            audio = self._load_audio(output_path)
            
            return audio
        
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode('utf-8', errors='ignore') if e.stderr else 'N/A'
            raise RuntimeError(
                f"Piper synthesis failed (exit code {e.returncode}):\n{stderr}"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Piper synthesis timed out (>30s)")
        finally:
            # Clean up temp file
            if os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {output_path}: {e}")
    
    def _load_audio(self, path: str) -> np.ndarray:
        """Load audio file and resample if needed."""
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError(
                "soundfile is required for Piper TTS. Install with: pip install soundfile"
            )
        
        # Load audio
        audio, sr = sf.read(path)
        
        # Convert stereo to mono if needed
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # Resample if needed
        if sr != self.config.sample_rate:
            try:
                import librosa
                audio = librosa.resample(
                    audio,
                    orig_sr=sr,
                    target_sr=self.config.sample_rate
                )
                logger.debug(f"Resampled audio from {sr}Hz to {self.config.sample_rate}Hz")
            except ImportError:
                logger.warning(
                    f"librosa not installed, cannot resample from {sr}Hz to "
                    f"{self.config.sample_rate}Hz. Using original sample rate."
                )
        
        return audio.astype(np.float32)
