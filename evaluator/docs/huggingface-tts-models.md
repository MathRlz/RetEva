# HuggingFace TTS Models Guide

The evaluator now supports modern HuggingFace text-to-speech models for high-quality audio synthesis.

## Available Models

### 🌟 Recommended: Parler-TTS
**Provider:** `parler_tts`  
**Quality:** Very High  
**Type:** HuggingFace

High-quality controllable TTS with natural prosody and fast inference.

**Models:**
- `parler-tts/parler-tts-mini-v1` - Fast, good quality (recommended)
- `parler-tts/parler-tts-large-v1` - Highest quality

**Setup:**
```bash
pip install parler-tts
```

**Config Example:**
```yaml
audio_synthesis:
  provider: "parler_tts"
  voice: "parler-tts/parler-tts-mini-v1"
  sample_rate: 16000
  cache_dir: ".cache/tts"
```

**Features:**
- Natural prosody
- Speaker control
- Fast inference (~100x realtime)

---

### SpeechT5 (Microsoft)
**Provider:** `speecht5`  
**Quality:** High  
**Type:** HuggingFace

Microsoft's unified speech/text transformer model.

**Models:**
- `microsoft/speecht5_tts` - Main TTS model
- `microsoft/speecht5_vc` - Voice conversion

**Setup:**
```bash
pip install transformers datasets
```

**Features:**
- Multi-speaker support
- Fast inference
- Good quality
- Low memory footprint

---

### Bark (Suno AI)
**Provider:** `bark`  
**Quality:** Very High  
**Type:** HuggingFace

Transformer-based TTS that can generate music and sound effects.

**Models:**
- `suno/bark` - Full model (~10GB)
- `suno/bark-small` - Smaller variant

**Setup:**
```bash
pip install git+https://github.com/suno-ai/bark.git
```

**Features:**
- Multi-lingual support
- Music generation
- Sound effects
- Emotional speech
- Non-verbal sounds (laughter, sighs, etc.)

**Note:** Large model, first run downloads ~10GB of weights.

---

### XTTS v2 (Coqui)
**Provider:** `xtts`  
**Quality:** Very High  
**Type:** HuggingFace

Advanced TTS with voice cloning capabilities.

**Models:**
- `coqui/XTTS-v2`

**Setup:**
```bash
pip install TTS
```

**Features:**
- Voice cloning (6-second reference audio)
- 17 languages supported
- Zero-shot voice cloning
- High naturalness

**Languages:**
English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese, Japanese, Hungarian, Korean, Hindi

---

### MMS-TTS (Meta)
**Provider:** `mms_tts`  
**Quality:** High  
**Type:** HuggingFace

Massively Multilingual Speech with 1100+ languages.

**Models:**
- `facebook/mms-tts-eng` - English
- `facebook/mms-tts-spa` - Spanish
- `facebook/mms-tts-fra` - French
- `facebook/mms-tts-deu` - German
- `facebook/mms-tts-pol` - Polish
- ...and 1100+ more!

**Setup:**
```bash
pip install transformers
```

**Features:**
- 1100+ languages
- Lightweight models
- Good coverage for low-resource languages

---

### VITS
**Provider:** `vits`  
**Quality:** High  
**Type:** HuggingFace

End-to-end TTS with multiple community models.

**Models:**
- `facebook/mms-tts`
- `kakao-enterprise/vits-ljs`

**Setup:**
```bash
pip install transformers
```

**Features:**
- Fast inference
- Good quality
- Many model variants available

---

## Local Models

### Piper TTS
**Provider:** `piper`  
**Quality:** High  
**Type:** Local

Fast, high-quality local neural TTS (100x realtime).

**Setup:**
1. Download from https://github.com/rhasspy/piper/releases
2. Download voice models to `~/.local/share/piper/voices/`

**Voice Format:** `<language>_<region>-<name>-<quality>`
- Example: `en_US-lessac-medium`, `en_GB-alan-low`

**Features:**
- Very fast (100x realtime)
- Fully offline
- Low latency
- Many voice options

---

### Coqui TTS (Local)
**Provider:** `coqui`  
**Quality:** Very High  
**Type:** Local

**Models:**
- `tts_models/en/ljspeech/tacotron2-DDC`
- `tts_models/en/ljspeech/glow-tts`

**Features:**
- Voice cloning
- Multiple model architectures
- High quality

---

## Cloud Models

### Google TTS
**Provider:** `gtts`  
**Quality:** High  
**Type:** Cloud

**Setup:**
```bash
pip install gTTS
```

**Note:** Requires internet connection.

---

## Usage in Web UI

1. Navigate to **Text-to-Speech** page
2. Use filter buttons to view models by type:
   - 🤗 **HuggingFace** - Modern transformer models
   - 💻 **Local** - Offline models
   - ☁️ **Cloud** - Internet-based services
3. Click on a provider card to select it
4. Choose a voice/model from the dropdown
5. Configure sample rate and cache directory
6. Click **Save Configuration** to persist settings

## Configuration Examples

### Medical Dataset Synthesis with Parler-TTS
```yaml
experiment_name: medical_parler_tts
audio_synthesis:
  provider: "parler_tts"
  voice: "parler-tts/parler-tts-mini-v1"
  sample_rate: 16000
  cache_dir: ".cache/medical_tts"
  speed: 1.0
  pitch: 0
  volume: 1.0

data:
  dataset_name: medical_qa_text
  # Text-only questions will be auto-synthesized
```

### Multi-lingual with MMS-TTS
```yaml
audio_synthesis:
  provider: "mms_tts"
  voice: "facebook/mms-tts-pol"  # Polish
  sample_rate: 16000
  cache_dir: ".cache/tts"
```

### Voice Cloning with XTTS
```yaml
audio_synthesis:
  provider: "xtts"
  voice: "coqui/XTTS-v2"
  sample_rate: 22050
  cache_dir: ".cache/xtts"
  # Reference audio for voice cloning can be provided
```

## Performance Comparison

| Provider | Speed | Quality | Size | Offline | Languages |
|----------|-------|---------|------|---------|-----------|
| Parler-TTS | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ~2GB | ✅ | English |
| SpeechT5 | ⚡⚡⚡ | ⭐⭐⭐⭐ | ~400MB | ✅ | English |
| Bark | ⚡⚡ | ⭐⭐⭐⭐⭐ | ~10GB | ✅ | Multi |
| XTTS v2 | ⚡⚡ | ⭐⭐⭐⭐⭐ | ~2GB | ✅ | 17 langs |
| MMS-TTS | ⚡⚡⚡ | ⭐⭐⭐⭐ | ~1GB | ✅ | 1100+ |
| Piper | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | ~100MB | ✅ | Many |
| Google TTS | ⚡⚡⚡ | ⭐⭐⭐⭐ | N/A | ❌ | 100+ |

## Recommendations

**For Medical Applications:**
- **Best Quality:** Parler-TTS mini or XTTS v2
- **Fastest:** Piper or SpeechT5
- **Multi-lingual:** MMS-TTS or XTTS v2

**For Research:**
- **Prosody Control:** Parler-TTS
- **Voice Cloning:** XTTS v2
- **Low-Resource Languages:** MMS-TTS

**For Production:**
- **Offline:** Piper (fastest) or Parler-TTS (best quality)
- **Cloud:** Google TTS (simple, reliable)

## Troubleshooting

### Model Download Issues
HuggingFace models auto-download on first use. If you encounter issues:
```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/cache

# Pre-download model
python -c "from transformers import pipeline; pipeline('text-to-speech', model='microsoft/speecht5_tts')"
```

### GPU Memory Issues
For large models like Bark:
```python
# Use CPU instead
config.audio_synthesis.device = "cpu"
```

### Cache Directory Permissions
```bash
mkdir -p .cache/tts
chmod 755 .cache/tts
```

## Next Steps

1. Test different providers on the TTS management page
2. Choose the best model for your use case
3. Configure in your experiment YAML
4. Run evaluations with auto-synthesized audio
