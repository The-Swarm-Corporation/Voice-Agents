# Voice Agents Examples

This folder contains comprehensive examples demonstrating all core functions and utilities from the `voice_agents` package.

## Examples Table

| #   | Example File                                                          | Description                                          | Category          |
| --- | --------------------------------------------------------------------- | ---------------------------------------------------- | ----------------- |
| 1   | [`example_format_text_for_speech.py`](example_format_text_for_speech.py) | Text formatting for speech with abbreviation handling | Core Functions     |
| 2   | [`example_play_audio.py`](example_play_audio.py)                     | Audio playback and tone generation                   | Core Functions     |
| 3   | [`example_stream_tts.py`](example_stream_tts.py)                     | Unified TTS with OpenAI models, `list_models()`     | Core Functions     |
| 4   | [`example_stream_tts_elevenlabs.py`](example_stream_tts_elevenlabs.py) | ElevenLabs TTS, unified and direct functions         | Core Functions     |
| 5   | [`example_speech_to_text.py`](example_speech_to_text.py)              | OpenAI Whisper transcription                        | Core Functions     |
| 6   | [`example_record_audio.py`](example_record_audio.py)                 | Microphone audio recording                          | Core Functions     |
| 7   | [`example_streaming_tts_callback.py`](example_streaming_tts_callback.py) | StreamingTTSCallback for real-time TTS               | Utilities          |
| 8   | [`example_get_media_type.py`](example_get_media_type.py)            | Media type (MIME) utilities for FastAPI             | Utilities          |
| 9   | [`example_voice_selection.py`](example_voice_selection.py)            | Voice selection with `list_voices()`                 | Utilities          |
| 10  | [`example_complete_voice_agent.py`](example_complete_voice_agent.py)  | Complete voice agent workflows                      | Complete Workflows |

## Examples Overview

### Core Functions

1. **[`example_format_text_for_speech.py`](example_format_text_for_speech.py)**
   - Demonstrates text formatting for speech
   - Shows handling of abbreviations, URLs, and punctuation
   - Splits long text into speech-friendly chunks

2. **[`example_play_audio.py`](example_play_audio.py)**
   - Shows how to play audio data
   - Generates and plays simple tones
   - Demonstrates audio playback with different formats

3. **[`example_stream_tts.py`](example_stream_tts.py)**
   - Unified TTS API examples with OpenAI models
   - Different voices, models, and streaming modes
   - Batch and real-time processing examples
   - Demonstrates `list_models()` function

4. **[`example_stream_tts_elevenlabs.py`](example_stream_tts_elevenlabs.py)**
   - ElevenLabs TTS API examples
   - Unified `stream_tts()` and direct `stream_tts_elevenlabs()` usage
   - Voice selection and customization
   - Different output formats and settings
   - Demonstrates `list_models()` and `list_voices()` functions

5. **[`example_speech_to_text.py`](example_speech_to_text.py)**
   - OpenAI Whisper API examples
   - File-based and audio data transcription
   - Different models and response formats

6. **[`example_record_audio.py`](example_record_audio.py)**
   - Audio recording from microphone
   - Different durations and sample rates
   - Audio analysis and file saving

### Utilities

7. **[`example_streaming_tts_callback.py`](example_streaming_tts_callback.py)**
   - StreamingTTSCallback class usage
   - Real-time text-to-speech conversion
   - Integration with agent frameworks

8. **[`example_get_media_type.py`](example_get_media_type.py)**
   - Media type (MIME type) utilities
   - FastAPI integration examples
   - Format validation helpers

9. **[`example_voice_selection.py`](example_voice_selection.py)**
   - Available voices for both providers
   - Voice selection helpers using `list_voices()` function
   - Use case recommendations

### Complete Workflows

10. **[`example_complete_voice_agent.py`](example_complete_voice_agent.py)**
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
   ```

### Running Individual Examples

```bash
# Run a specific example
python examples/example_format_text_for_speech.py
python examples/example_stream_tts.py
python examples/example_speech_to_text.py
# ... etc
```

### Note on Audio Examples

Some examples (like `example_record_audio.py` and `example_speech_to_text.py`) require:
- A working microphone
- Audio input permissions
- Some examples are commented out to prevent accidental execution

Uncomment the relevant sections in the code to run them.

## Example Categories

### Text-to-Speech (TTS)
- [`example_stream_tts.py`](example_stream_tts.py) - Unified TTS with OpenAI models
- [`example_stream_tts_elevenlabs.py`](example_stream_tts_elevenlabs.py) - ElevenLabs TTS
- [`example_streaming_tts_callback.py`](example_streaming_tts_callback.py) - Streaming callbacks

### Speech-to-Text (STT)
- [`example_speech_to_text.py`](example_speech_to_text.py) - Whisper transcription
- [`example_record_audio.py`](example_record_audio.py) - Audio recording

### Utilities
- [`example_format_text_for_speech.py`](example_format_text_for_speech.py) - Text formatting
- [`example_play_audio.py`](example_play_audio.py) - Audio playback
- [`example_get_media_type.py`](example_get_media_type.py) - Media type helpers
- [`example_voice_selection.py`](example_voice_selection.py) - Voice selection

### Complete Workflows
- [`example_complete_voice_agent.py`](example_complete_voice_agent.py) - Full agent examples

## API Keys

### OpenAI API Key
Required for:
- `stream_tts()` - Text-to-speech
- `speech_to_text()` - Speech transcription

Get your key from: https://platform.openai.com/api-keys

### ElevenLabs API Key
Required for:
- `stream_tts_elevenlabs()` - ElevenLabs TTS

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

## Additional Resources

- Main package documentation: See `README.md` in the root directory
- API reference: See docstrings in `voice_agents/main.py`
- Issues: Report problems on the project's issue tracker

