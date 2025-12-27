# Voice Agents Examples

This folder contains comprehensive examples demonstrating all core functions and utilities from the `voice_agents` package, organized into logical categories.

## Folder Structure

```
examples/
├── text_to_speech/          # Text-to-speech examples
├── speech_to_text/          # Speech-to-text examples
├── utilities/               # Utility functions
└── workflows/               # Complete workflow examples
```

## Examples by Category

### Text-to-Speech (`text_to_speech/`)

1. **[`example_stream_tts.py`](text_to_speech/example_stream_tts.py)**
   - Unified TTS API examples with OpenAI models
   - Different voices, models, and streaming modes
   - Batch and real-time processing examples
   - Demonstrates `list_models()` function

2. **[`example_stream_tts_elevenlabs.py`](text_to_speech/example_stream_tts_elevenlabs.py)**
   - ElevenLabs TTS API examples
   - Unified `stream_tts()` and direct `stream_tts_elevenlabs()` usage
   - Voice selection and customization
   - Different output formats and settings
   - Demonstrates `list_models()` and `list_voices()` functions

3. **[`example_streaming_tts_callback.py`](text_to_speech/example_streaming_tts_callback.py)**
   - StreamingTTSCallback class usage
   - Real-time text-to-speech conversion
   - Integration with agent frameworks

4. **[`example_voice_selection.py`](text_to_speech/example_voice_selection.py)**
   - Available voices for both providers
   - Voice selection helpers using `list_voices()` function
   - Use case recommendations

### Speech-to-Text (`speech_to_text/`)

5. **[`example_speech_to_text.py`](speech_to_text/example_speech_to_text.py)**
   - OpenAI Whisper API examples
   - File-based and audio data transcription
   - Different models and response formats

6. **[`example_speech_to_text_elevenlabs_file.py`](speech_to_text/example_speech_to_text_elevenlabs_file.py)**
   - ElevenLabs Speech-to-Text from audio file
   - Non-real-time transcription

7. **[`example_speech_to_text_elevenlabs_audio.py`](speech_to_text/example_speech_to_text_elevenlabs_audio.py)**
   - ElevenLabs Speech-to-Text from audio data (numpy array)
   - Non-real-time transcription

8. **[`example_speech_to_text_elevenlabs_realtime.py`](speech_to_text/example_speech_to_text_elevenlabs_realtime.py)**
   - ElevenLabs real-time WebSocket transcription
   - Streaming audio with live transcription results

9. **[`example_speech_to_text_elevenlabs_diarization.py`](speech_to_text/example_speech_to_text_elevenlabs_diarization.py)**
   - Speaker diarization with ElevenLabs
   - Identify different speakers in conversations

10. **[`example_speech_to_text_elevenlabs_timestamps.py`](speech_to_text/example_speech_to_text_elevenlabs_timestamps.py)**
    - Word-level timestamps with ElevenLabs
    - Precise timing information for transcription

### Utilities (`utilities/`)

11. **[`example_format_text_for_speech.py`](utilities/example_format_text_for_speech.py)**
    - Text formatting for speech
    - Shows handling of abbreviations, URLs, and punctuation
    - Splits long text into speech-friendly chunks

12. **[`example_play_audio.py`](utilities/example_play_audio.py)**
    - Shows how to play audio data
    - Generates and plays simple tones
    - Demonstrates audio playback with different formats

13. **[`example_record_audio.py`](utilities/example_record_audio.py)**
    - Audio recording from microphone
    - Different durations and sample rates
    - Audio analysis and file saving

14. **[`example_get_media_type.py`](utilities/example_get_media_type.py)**
    - Media type (MIME type) utilities
    - FastAPI integration examples
    - Format validation helpers

### Complete Workflows (`workflows/`)

15. **[`example_complete_voice_agent.py`](workflows/example_complete_voice_agent.py)**
    - Full conversational voice agent
    - Combines recording, transcription, and TTS
    - Multiple agent patterns and examples

## Running Examples

### Prerequisites

1. Install the package:
   ```bash
   pip install -e .
   ```

2. Set environment variables:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export ELEVENLABS_API_KEY="your-elevenlabs-api-key"  # Optional, for ElevenLabs examples
   ```

3. Install additional dependencies (if needed):
   ```bash
   pip install soundfile  # For audio file operations
   pip install websockets  # For real-time ElevenLabs STT
   ```

### Running Individual Examples

```bash
# Text-to-Speech examples
python examples/text_to_speech/example_stream_tts.py
python examples/text_to_speech/example_stream_tts_elevenlabs.py

# Speech-to-Text examples
python examples/speech_to_text/example_speech_to_text.py
python examples/speech_to_text/example_speech_to_text_elevenlabs_file.py

# Utility examples
python examples/utilities/example_format_text_for_speech.py
python examples/utilities/example_record_audio.py

# Workflow examples
python examples/workflows/example_complete_voice_agent.py
```

### Note on Audio Examples

Some examples (like `example_record_audio.py` and speech-to-text examples) require:
- A working microphone
- Audio input permissions
- Valid audio files for file-based examples

## API Keys

### OpenAI API Key
Required for:
- `stream_tts()` - Text-to-speech
- `speech_to_text()` - Speech transcription

Get your key from: https://platform.openai.com/api-keys

### ElevenLabs API Key
Required for:
- `stream_tts_elevenlabs()` - ElevenLabs TTS
- `speech_to_text_elevenlabs()` - ElevenLabs Speech-to-Text

Get your key from: https://elevenlabs.io/app/settings/api-keys

## Common Patterns

### Basic TTS
```python
from voice_agents import stream_tts
stream_tts(["Hello, world!"], model="openai/tts-1", voice="alloy")
```

### Record and Transcribe
```python
from voice_agents import record_audio, speech_to_text
audio = record_audio(duration=5.0)
text = speech_to_text(audio_data=audio, sample_rate=16000)
```

### Complete Voice Loop
```python
from voice_agents import record_audio, speech_to_text, stream_tts, format_text_for_speech

# Record
audio = record_audio(duration=5.0)

# Transcribe
text = speech_to_text(audio_data=audio, sample_rate=16000)

# Respond
response = f"You said: {text}"
chunks = format_text_for_speech(response)
stream_tts(chunks, model="openai/tts-1", voice="alloy")
```

### List Available Models and Voices
```python
from voice_agents import list_models, list_voices

# List all available models
models = list_models()
for model in models:
    print(f"{model['model']} ({model['provider']})")

# List all available voices
voices = list_voices()
for voice in voices:
    if voice['provider'] == 'openai':
        print(f"{voice['voice']} ({voice['provider']})")
    else:
        print(f"{voice['voice']} ({voice['provider']}) - {voice['description']}")
```

## Troubleshooting

### Audio Playback Issues
- Ensure your system audio is working
- Check that `sounddevice` is properly installed
- On Linux, you may need: `sudo apt-get install portaudio19-dev`

### API Key Errors
- Verify your API keys are set correctly: `echo $OPENAI_API_KEY`
- Check that keys are valid and have sufficient credits
- Ensure keys don't have extra spaces or newlines

### Microphone Issues
- Check microphone permissions in system settings
- Verify microphone is connected and working
- Test with system audio recording tools first

### WebSocket Issues (Real-time STT)
- Ensure `websockets` library is installed: `pip install websockets`
- Check network connectivity
- Verify API key has access to real-time features

## Additional Resources

- Main package documentation: See `README.md` in the root directory
- API reference: See docstrings in `voice_agents/main.py`
- Issues: Report problems on the project's issue tracker
