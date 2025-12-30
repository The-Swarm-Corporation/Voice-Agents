# Text-to-Speech Examples

This folder contains examples demonstrating text-to-speech (TTS) functionality using various providers including OpenAI, ElevenLabs, and Groq.

## Examples

| Example | Description | Provider | Features |
|---------|-------------|----------|----------|
| [`example_stream_tts_elevenlabs.py`](example_stream_tts_elevenlabs.py) | ElevenLabs TTS with unified API | ElevenLabs | Voice selection, streaming, multiple formats |
| [`example_stream_tts_groq.py`](example_stream_tts_groq.py) | Groq TTS with Orpheus models | Groq | Fast TTS, vocal directions, long text chunking |
| [`example_streaming_tts_callback.py`](example_streaming_tts_callback.py) | Streaming TTS callback class | Multi-provider | Real-time TTS, agent integration |
| [`example_stt.py`](example_stt.py) | Simple TTS with OpenAI | OpenAI | Basic TTS, default voice and model |
| [`example_voice_selection.py`](example_voice_selection.py) | Voice selection examples | Multi-provider | Available voices for all providers |
| [`streaming_callback_example.py`](streaming_callback_example.py) | Streaming callback pattern | Multi-provider | Advanced streaming patterns |

## Quick Start

### Prerequisites

1. Install required dependencies:
   ```bash
   pip install voice_agents
   ```

2. Set API keys:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export ELEVENLABS_API_KEY="your-elevenlabs-api-key"  # Optional
   export GROQ_API_KEY="your-groq-api-key"              # Optional
   ```

### Running Examples

```bash
# ElevenLabs TTS
python examples/text_to_speech/example_stream_tts_elevenlabs.py

# Groq TTS
python examples/text_to_speech/example_stream_tts_groq.py

# Streaming callback
python examples/text_to_speech/example_streaming_tts_callback.py

# Simple TTS
python examples/text_to_speech/example_stt.py

# Voice selection
python examples/text_to_speech/example_voice_selection.py

# Streaming callback example
python examples/text_to_speech/streaming_callback_example.py
```

## Example Details

### ElevenLabs TTS

Unified TTS API with ElevenLabs:

```python
from voice_agents import stream_tts

stream_tts(
    ["Hello! This is a test of ElevenLabs text-to-speech API using the unified function."],
    model="elevenlabs/eleven_multilingual_v2",
    voice="rachel",
)
```

### Groq TTS

Fast TTS with Groq's Orpheus models:

```python
from voice_agents import stream_tts_groq

stream_tts_groq(
    text_chunks=["Hello! This is a test of Groq's fast text to speech using the Orpheus model."],
    voice="austin",
    model="canopylabs/orpheus-v1-english",
    stream_mode=True,
)
```

### Simple TTS (OpenAI)

Basic TTS with default settings:

```python
from voice_agents import stream_tts

stream_tts(
    text_chunks=["Hello! This is a test of OpenAI's text-to-speech API."],
    model="openai/tts-1",
    voice="alloy",
    stream_mode=False,
    response_format="pcm",
)
```

### Voice Selection

List and select available voices:

```python
from voice_agents import stream_tts, VOICES

stream_tts(
    ["This demonstrates voice selection."],
    model="openai/tts-1",
    voice=VOICES[0],
)
```

### Streaming Callback

Use StreamingTTSCallback for real-time TTS:

```python
from voice_agents import StreamingTTSCallback

# See example_streaming_tts_callback.py for full implementation
```

## API Keys

### OpenAI API Key
Required for OpenAI TTS examples. Get your key from: https://platform.openai.com/api-keys

### ElevenLabs API Key
Required for ElevenLabs TTS examples. Get your key from: https://elevenlabs.io/app/settings/api-keys

### Groq API Key
Required for Groq TTS examples. Get your key from: https://console.groq.com/keys

## Available Models

### OpenAI
- `openai/tts-1` - Standard TTS model
- `openai/tts-1-hd` - High-definition TTS model

### ElevenLabs
- `elevenlabs/eleven_multilingual_v2` - Multilingual model
- `elevenlabs/eleven_turbo_v2` - Fast model

### Groq
- `canopylabs/orpheus-v1-english` - English Orpheus model

## Available Voices

Use `list_voices()` to see all available voices:

```python
from voice_agents import list_voices

voices = list_voices()
for voice in voices:
    print(f"{voice['voice']} ({voice['provider']})")
```

## Common Use Cases

### Basic TTS
Convert text to speech with default settings.

### Streaming TTS
Stream audio as it's generated for real-time playback.

### Long Text Chunking
Automatically split long text into speech-friendly chunks.

### Voice Customization
Select from various voices across providers.

### Multi-language Support
Use multilingual models for different languages.

## Troubleshooting

### Audio Playback Issues
- Ensure your system audio is working
- Check that `sounddevice` is properly installed
- On Linux, you may need: `sudo apt-get install portaudio19-dev`

### API Key Errors
- Verify your API keys are set correctly: `echo $OPENAI_API_KEY`
- Check that keys are valid and have sufficient credits
- Ensure keys don't have extra spaces or newlines

### Streaming Issues
- Check network connectivity
- Verify API key has access to streaming features
- Ensure sufficient API rate limits

### Voice Not Found
- Use `list_voices()` to see available voices
- Check provider-specific voice names
- Verify voice is available for selected model

