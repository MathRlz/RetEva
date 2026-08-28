"""Piper TTS provider - fast, local, multi-language text-to-speech."""
import subprocess
import numpy as np
import tempfile
import os
from pathlib import Path
import logging
from ..registry import register_tts_model
from .base_tts import BaseTTSModel

logger = logging.getLogger(__name__)


@register_tts_model(
    'piper',
    default_name='en_US-lessac-medium',
    capabilities=['speech_synthesis'],
    description='Piper TTS — fast, local, multi-language',
)
class PiperTTS(BaseTTSModel):
    """Piper TTS provider - fast, local, multi-language."""

    def __init__(self, config):
        """
        Initialize Piper TTS provider.

        Args:
            config: AudioSynthesisConfig instance.
        """
        super().__init__(config)
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
        # config.voice may already be a literal path (relative or absolute) to the
        # .onnx file itself, e.g. because the user ran `python3 -m piper.download_voices`
        # by hand and wants to point straight at the result.
        direct = Path(self.config.voice)
        if direct.suffix == ".onnx" and direct.is_file():
            logger.debug(f"Using voice path directly: {direct}")
            return str(direct)

        # Try with and without .onnx extension
        voice_name = self.config.voice
        if not voice_name.endswith('.onnx'):
            voice_name += '.onnx'

        # `python3 -m piper.download_voices <name>` (the current piper-tts package's own
        # download tool) writes into the CURRENT DIRECTORY by default, or into whatever
        # --data-dir/PIPER_DATA_DIR points at — neither of which is a fixed system path,
        # so bare CWD and that override must be searched, not just `./voices`-style
        # subfolders.
        voice_dirs = []
        data_dir_override = os.environ.get("PIPER_DATA_DIR")
        if data_dir_override:
            voice_dirs.append(Path(data_dir_override))
        voice_dirs.extend([
            Path.cwd(),
            Path.home() / ".local/share/piper/voices",
            Path("/usr/share/piper/voices"),
            Path("/usr/local/share/piper/voices"),
            Path("./voices"),
            Path("./piper_voices"),
        ])

        for voice_dir in voice_dirs:
            if not voice_dir.exists():
                continue

            voice_path = voice_dir / voice_name
            if voice_path.exists():
                logger.debug(f"Found voice at: {voice_path}")
                return str(voice_path)

        # Not found locally — try downloading it automatically via piper's own
        # download tool before giving up.
        downloaded = self._download_voice(voice_name)
        if downloaded:
            return downloaded

        # Voice not found - provide helpful error message
        searched_paths = [str(d / voice_name) for d in voice_dirs if d.exists()]
        raise FileNotFoundError(
            f"Voice model not found: {self.config.voice}\n"
            "Searched in:\n" + "\n".join(f"  - {p}" for p in searched_paths) + "\n"
            "\nAutomatic download also failed. Try manually:\n"
            "  python3 -m piper.download_voices " + self.config.voice + " --data-dir "
            + str(Path.home() / ".local/share/piper/voices") + "\n"
        )

    def _download_voice(self, voice_name: str) -> str | None:
        """Try `python3 -m piper.download_voices <name>` into a fixed, always-searched
        directory, so a missing voice is fetched once and found by every future run."""
        voice_id = self.config.voice
        if voice_id.endswith(".onnx"):
            voice_id = voice_id[:-len(".onnx")]
        target_dir = Path.home() / ".local/share/piper/voices"
        target_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Piper voice %r not found locally, downloading via piper.download_voices into %s ...",
            voice_id, target_dir,
        )
        try:
            result = subprocess.run(
                ["python3", "-m", "piper.download_voices", voice_id, "--data-dir", str(target_dir)],
                capture_output=True,
                text=True,
                timeout=300,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning("Automatic Piper voice download failed to run: %s", e)
            return None

        if result.returncode != 0:
            logger.warning(
                "piper.download_voices exited %s: %s",
                result.returncode, result.stderr.strip() or result.stdout.strip(),
            )
            return None

        voice_path = target_dir / voice_name
        if voice_path.exists():
            logger.info("Downloaded Piper voice to %s", voice_path)
            return str(voice_path)

        logger.warning("piper.download_voices reported success but %s is missing", voice_path)
        return None

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
            from ...utils.audio import resample_audio
            audio = resample_audio(audio, sr, self.config.sample_rate)

        return audio.astype(np.float32)
