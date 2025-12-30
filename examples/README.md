# Voice Agents Examples

This folder contains comprehensive examples demonstrating all core functions and utilities from the `voice_agents` package, organized into logical categories.

## Folder Structure

```
examples/
├── text_to_speech/          # Text-to-speech examples
├── speech_to_text/          # Speech-to-text examples
├── utilities/               # Utility functions
├── agents/                  # Agent integration examples
└── workflows/               # Complete workflow examples
```

## Complete Examples Table

| Category | Example File | Description | Provider | Link |
|----------|-------------|-------------|----------|------|
| **Text-to-Speech** | `example_stream_tts_elevenlabs.py` | ElevenLabs TTS with unified API | ElevenLabs | [Link](text_to_speech/example_stream_tts_elevenlabs.py) |
| **Text-to-Speech** | `example_stream_tts_groq.py` | Groq TTS with Orpheus models | Groq | [Link](text_to_speech/example_stream_tts_groq.py) |
| **Text-to-Speech** | `example_streaming_tts_callback.py` | Streaming TTS callback class | Multi-provider | [Link](text_to_speech/example_streaming_tts_callback.py) |
| **Text-to-Speech** | `example_stt.py` | Simple TTS with OpenAI | OpenAI | [Link](text_to_speech/example_stt.py) |
| **Text-to-Speech** | `example_voice_selection.py` | Voice selection examples | Multi-provider | [Link](text_to_speech/example_voice_selection.py) |
| **Text-to-Speech** | `streaming_callback_example.py` | Streaming callback pattern | Multi-provider | [Link](text_to_speech/streaming_callback_example.py) |
| **Speech-to-Text** | `example_speech_to_text_elevenlabs_file.py` | Transcribe audio from file | ElevenLabs | [Link](speech_to_text/example_speech_to_text_elevenlabs_file.py) |
| **Speech-to-Text** | `example_speech_to_text_elevenlabs_audio.py` | Transcribe audio from numpy array | ElevenLabs | [Link](speech_to_text/example_speech_to_text_elevenlabs_audio.py) |
| **Speech-to-Text** | `example_speech_to_text_elevenlabs_realtime.py` | Real-time transcription via WebSocket | ElevenLabs | [Link](speech_to_text/example_speech_to_text_elevenlabs_realtime.py) |
| **Speech-to-Text** | `example_speech_to_text_elevenlabs_diarization.py` | Speaker diarization | ElevenLabs | [Link](speech_to_text/example_speech_to_text_elevenlabs_diarization.py) |
| **Speech-to-Text** | `example_speech_to_text_elevenlabs_timestamps.py` | Word-level timestamps | ElevenLabs | [Link](speech_to_text/example_speech_to_text_elevenlabs_timestamps.py) |
| **Speech-to-Text** | `example_speech_to_text_groq.py` | Fast Whisper transcription | Groq | [Link](speech_to_text/example_speech_to_text_groq.py) |
| **Utilities** | `example_record_audio.py` | Audio recording from microphone | - | [Link](utilities/example_record_audio.py) |
| **Utilities** | `example_play_audio.py` | Audio playback | - | [Link](utilities/example_play_audio.py) |
| **Utilities** | `example_format_text_for_speech.py` | Text formatting for speech | - | [Link](utilities/example_format_text_for_speech.py) |
| **Utilities** | `example_get_media_type.py` | Media type utilities | - | [Link](utilities/example_get_media_type.py) |
| **Utilities** | `example.py` | General utilities example | - | [Link](utilities/example.py) |
| **Agents** | `speech_to_text_agent.py` | Agent with speech-to-text input | Swarms | [Link](agents/speech_to_text_agent.py) |
| **Workflows** | `example_complete_voice_agent.py` | Complete conversational voice agent | Multi-provider | [Link](workflows/example_complete_voice_agent.py) |
| **Root Examples** | `example_speech_to_text_elevenlabs_audio.py` | ElevenLabs STT from audio data | ElevenLabs | [Link](example_speech_to_text_elevenlabs_audio.py) |
| **Root Examples** | `example_speech_to_text_elevenlabs_file.py` | ElevenLabs STT from file | ElevenLabs | [Link](example_speech_to_text_elevenlabs_file.py) |

## Examples by Category

### Text-to-Speech (`text_to_speech/`)

See the [Text-to-Speech README](text_to_speech/README.md) for detailed documentation.

- **[`example_stream_tts_elevenlabs.py`](text_to_speech/example_stream_tts_elevenlabs.py)** - ElevenLabs TTS with unified API
- **[`example_stream_tts_groq.py`](text_to_speech/example_stream_tts_groq.py)** - Groq TTS with Orpheus models
- **[`example_streaming_tts_callback.py`](text_to_speech/example_streaming_tts_callback.py)** - Streaming TTS callback class
- **[`example_stt.py`](text_to_speech/example_stt.py)** - Simple TTS with OpenAI
- **[`example_voice_selection.py`](text_to_speech/example_voice_selection.py)** - Voice selection examples
- **[`streaming_callback_example.py`](text_to_speech/streaming_callback_example.py)** - Streaming callback pattern

### Speech-to-Text (`speech_to_text/`)

See the [Speech-to-Text README](speech_to_text/README.md) for detailed documentation.

- **[`example_speech_to_text_elevenlabs_file.py`](speech_to_text/example_speech_to_text_elevenlabs_file.py)** - Transcribe audio from file
- **[`example_speech_to_text_elevenlabs_audio.py`](speech_to_text/example_speech_to_text_elevenlabs_audio.py)** - Transcribe audio from numpy array
- **[`example_speech_to_text_elevenlabs_realtime.py`](speech_to_text/example_speech_to_text_elevenlabs_realtime.py)** - Real-time transcription via WebSocket
- **[`example_speech_to_text_elevenlabs_diarization.py`](speech_to_text/example_speech_to_text_elevenlabs_diarization.py)** - Speaker diarization
- **[`example_speech_to_text_elevenlabs_timestamps.py`](speech_to_text/example_speech_to_text_elevenlabs_timestamps.py)** - Word-level timestamps
- **[`example_speech_to_text_groq.py`](speech_to_text/example_speech_to_text_groq.py)** - Fast Whisper transcription

### Utilities (`utilities/`)

See the [Utilities README](utilities/README.md) for detailed documentation.

- **[`example_record_audio.py`](utilities/example_record_audio.py)** - Audio recording from microphone
- **[`example_play_audio.py`](utilities/example_play_audio.py)** - Audio playback
- **[`example_format_text_for_speech.py`](utilities/example_format_text_for_speech.py)** - Text formatting for speech
- **[`example_get_media_type.py`](utilities/example_get_media_type.py)** - Media type utilities
- **[`example.py`](utilities/example.py)** - General utilities example

### Agents (`agents/`)

See the [Agents README](agents/README.md) for detailed documentation.

- **[`speech_to_text_agent.py`](agents/speech_to_text_agent.py)** - Agent with speech-to-text input

### Workflows (`workflows/`)

See the [Workflows README](workflows/README.md) for detailed documentation.

- **[`example_complete_voice_agent.py`](workflows/example_complete_voice_agent.py)** - Complete conversational voice agent

## Running Examples

### Prerequisites

1. Install the package:
   ```bash
   pip install -e .
   ```

2. Set environment variables:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export ELEVENLABS_API_KEY="your-elevenlabs-api-key"  # Optional
   export GROQ_API_KEY="your-groq-api-key"              # Optional
   ```

3. Install additional dependencies (if needed):
   ```bash
   pip install soundfile  # For audio file operations
   pip install websockets  # For real-time ElevenLabs STT
   ```

### Running Individual Examples

```bash
# Text-to-Speech examples
python examples/text_to_speech/example_stream_tts_elevenlabs.py
python examples/text_to_speech/example_stream_tts_groq.py
python examples/text_to_speech/example_streaming_tts_callback.py
python examples/text_to_speech/example_stt.py
python examples/text_to_speech/example_voice_selection.py
python examples/text_to_speech/streaming_callback_example.py

# Speech-to-Text examples
python examples/speech_to_text/example_speech_to_text_elevenlabs_file.py
python examples/speech_to_text/example_speech_to_text_elevenlabs_audio.py
python examples/speech_to_text/example_speech_to_text_elevenlabs_realtime.py
python examples/speech_to_text/example_speech_to_text_elevenlabs_diarization.py
python examples/speech_to_text/example_speech_to_text_elevenlabs_timestamps.py
python examples/speech_to_text/example_speech_to_text_groq.py

# Utility examples
python examples/utilities/example_record_audio.py
python examples/utilities/example_play_audio.py
python examples/utilities/example_format_text_for_speech.py
python examples/utilities/example_get_media_type.py
python examples/utilities/example.py

# Agent examples
python examples/agents/speech_to_text_agent.py

# Workflow examples
python examples/workflows/example_complete_voice_agent.py

# Root examples
python examples/example_speech_to_text_elevenlabs_audio.py
python examples/example_speech_to_text_elevenlabs_file.py
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

### Groq API Key
Required for:
- `stream_tts_groq()` - Groq TTS
- `speech_to_text_groq()` - Groq STT/Translation

Get your key from: https://console.groq.com/keys

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
