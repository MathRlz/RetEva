# Quick Model Testing Guide

The Local Models page now includes a **Quick Model Testing** section that allows you to test individual models without running a full evaluation.

## Features

### 1. LLM Model Testing
- Select any available LLM model (Mistral, Llama, medical models, etc.)
- Enter a custom prompt
- Get immediate response with timing information
- See token count and response time

**Example Use Cases:**
- Test if a medical LLM understands clinical terminology
- Compare response quality between general and medical models
- Verify model is loaded and responding correctly

### 2. ASR Model Testing
- Select ASR model (Whisper, Faster Whisper, Wav2Vec2)
- Upload an audio file OR record directly from microphone
- Get transcription with language detection
- See processing time

**Example Use Cases:**
- Test ASR accuracy on medical terminology
- Compare different Whisper model sizes
- Verify microphone input works correctly
- Test custom audio files before running full evaluations

### 3. TTS Provider Testing
- Select TTS provider (Piper, Google TTS, eSpeak)
- Enter text to synthesize
- Choose voice model
- Play synthesized audio immediately
- See synthesis time

**Example Use Cases:**
- Preview TTS voice quality
- Test pronunciation of medical terms
- Verify TTS provider is installed correctly
- Compare different voices for dataset synthesis

## How to Use

1. Navigate to **Local Models** page from main menu
2. Scroll to **Quick Model Testing** section
3. Click on the tab for the model type you want to test:
   - **LLM Models** - Language model inference
   - **ASR Models** - Speech recognition
   - **TTS Providers** - Speech synthesis
4. Configure settings and click the **Test** button
5. View results with timing information

## Technical Details

### LLM Testing
- Uses Ollama backend for inference
- Default max tokens: 200
- Returns response text, elapsed time, and token count
- Requires Ollama service to be running

### ASR Testing
- Supports file upload (WAV, MP3, FLAC, etc.)
- Browser recording uses `getUserMedia` API
- Audio converted to mono if stereo
- Returns transcription, language, and elapsed time

### TTS Testing
- Synthesizes audio from text input
- Returns base64-encoded WAV audio
- Audio plays directly in browser
- Shows synthesis time and audio duration

## API Endpoints

### POST /api/test/llm
```json
{
  "model": "mistral:7b-instruct",
  "prompt": "What is hypertension?"
}
```

**Response:**
```json
{
  "success": true,
  "response": "Hypertension is...",
  "elapsed": 2.3,
  "tokens": 45
}
```

### POST /api/test/asr
- Content-Type: `multipart/form-data`
- Fields: `model` (string), `audio` (file)

**Response:**
```json
{
  "success": true,
  "transcription": "The patient presents with...",
  "language": "en",
  "elapsed": 1.5
}
```

### POST /api/test/tts
```json
{
  "provider": "piper",
  "text": "The patient presents with symptoms.",
  "voice": "en_US-lessac-medium"
}
```

**Response:**
```json
{
  "success": true,
  "audio_base64": "UklGRiQAAABXQVZF...",
  "elapsed": 0.8,
  "audio_length": 3.2
}
```

## Troubleshooting

### LLM Test Fails
- **Error: "Failed to create server"**
  - Make sure Ollama is installed and running
  - Check if the selected model is pulled: `ollama list`
  - Pull model if needed: `ollama pull mistral:7b-instruct`

### ASR Test Fails
- **Error: "No audio file provided"**
  - Upload a file OR record audio using microphone
- **Microphone not working**
  - Grant browser permission to access microphone
  - Check browser console for detailed errors
  
### TTS Test Fails
- **Error: "Provider not found"**
  - Piper: Install from https://github.com/rhasspy/piper
  - Download voice models to `~/.local/share/piper/voices/`
  - See [TTS Setup Guide](tts-setup.md) for details

## Performance Tips

1. **LLM Testing**
   - First request may be slow (model loading)
   - Subsequent requests are faster
   - Stop model to free memory when done

2. **ASR Testing**
   - Smaller models (tiny, base) are faster
   - Larger models (medium, large) are more accurate
   - Use `faster-whisper` for production (2-3x faster)

3. **TTS Testing**
   - Piper is fastest (100x realtime)
   - Results are cached for repeated text
   - Different voices have different quality/speed tradeoffs

## Next Steps

After testing models individually:
1. Go to **Experiment** page to configure full evaluations
2. Use the models you tested in your evaluation config
3. Run batch evaluations with confidence in model settings
