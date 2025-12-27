<div align="center">

# 🗣️ Voice-Agents

*Enterprise-Grade Voice Agent Framework*

<br>

🏠 [Swarms Website](https://swarms.ai) • 📚 [Documentation](https://docs.swarms.world) • 📦 [Examples](#swarms-integration)

<br>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![GitHub stars](https://img.shields.io/github/stars/The-Swarm-Corporation/Voice-Agents?style=social&logo=github)](https://github.com/The-Swarm-Corporation/Voice-Agents)

</div>

---

## Overview

**Voice-Agents** is a production-ready Python library for building enterprise-grade voice-enabled agentic applications. Built by [Swarms Corporation](https://swarms.ai), it provides seamless integration with multiple TTS/STT providers including OpenAI, ElevenLabs, and Groq, with real-time streaming capabilities optimized for agent-based architectures.

Voice-Agents delivers the infrastructure required to build conversational agentic assistants, voice-enabled agents, and real-time audio processing systems, enabling rapid deployment from prototype to production.

### Built by Swarms Corporation

Voice-Agents is part of the [Swarms](https://github.com/kyegomez/swarms) ecosystem—the enterprise-grade, production-ready multi-agent orchestration framework. Learn more at [swarms.ai](https://swarms.ai) and [docs.swarms.world](https://docs.swarms.world).

---

## Features

### Core Capabilities

| Feature                          | Description                                                        |
|-----------------------------------|--------------------------------------------------------------------|
| **Multi-Provider TTS Support**    | Seamlessly switch between OpenAI, ElevenLabs, and Groq             |
| **Real-Time Streaming**           | Low-latency audio streaming for live agent interactions            |
| **Speech-to-Text**                | High-accuracy transcription using OpenAI Whisper                   |
| **Audio Processing**              | Built-in utilities for recording, playback, and format conversion  |
| **Production-Ready**              | Enterprise-grade error handling, authentication, and logging       |

### Advanced Features

| Feature                  | Description                                                        |
|--------------------------|--------------------------------------------------------------------|
| **Streaming Callbacks**  | Real-time TTS callbacks for agent streaming outputs                |
| **Multiple Audio Formats** | Support for PCM, MP3, Opus, AAC, FLAC, and more                  |
| **Voice Customization**  | 10+ OpenAI voices and 30+ ElevenLabs voices                        |
| **Sentence Detection**   | Intelligent text formatting for natural speech pauses              |
| **FastAPI Integration**  | Generator-based streaming for web applications                     |
| **Type Safety**          | Full type hints and Literal types for better IDE support           |

---

## Installation

### Basic Installation

```bash
pip install voice-agents
```

### Development Installation

```bash
git clone https://github.com/The-Swarm-Corporation/Voice-Agents.git
cd Voice-Agents
pip install -e .
```

### Requirements

- Python 3.10+
- API keys for your chosen providers:
  - OpenAI API key (for TTS and Whisper STT)
  - ElevenLabs API key (optional, for ElevenLabs TTS)

---

## Quick Start

### Environment Setup

Create a `.env` file or set environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ELEVENLABS_API_KEY="your-elevenlabs-api-key"  # Optional
export GROQ_API_KEY="your-groq-api-key"              # Optional
```

### Basic Text-to-Speech

```python
from voice_agents import stream_tts, format_text_for_speech

# Format text for natural speech
text = "Hello! This is a voice agent speaking. How can I help you today?"
chunks = format_text_for_speech(text)

# Convert to speech and play
stream_tts(chunks, model="openai/tts-1", voice="alloy")
```

### Speech-to-Text

```python
from voice_agents import speech_to_text, record_audio

# Record audio from microphone
audio = record_audio(duration=5.0, sample_rate=16000)

# Transcribe to text
transcription = speech_to_text(audio_data=audio, sample_rate=16000)
print(f"Transcribed: {transcription}")
```

---

## Core Functions

### Text-to-Speech (OpenAI)

```python
from voice_agents import stream_tts, format_text_for_speech, VOICES

# Available voices: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
text_chunks = format_text_for_speech("Your text here")

# Basic usage - plays audio
stream_tts(text_chunks, model="openai/tts-1", voice="nova")

# Streaming mode for real-time processing
stream_tts(
    text_chunks,
    model="openai/tts-1",
    voice="alloy",
    stream_mode=True,  # Process chunks as they arrive
    response_format="pcm"
)

# For FastAPI/web streaming
from fastapi.responses import StreamingResponse

def audio_endpoint():
    generator = stream_tts(
        text_chunks,
        voice="alloy",
        return_generator=True
    )
    return StreamingResponse(generator, media_type="audio/pcm")
```

### Text-to-Speech (ElevenLabs)

```python
from voice_agents import stream_tts_elevenlabs, ELEVENLABS_VOICE_NAMES

# Available voices: rachel, domi, bella, antoni, elli, josh, and 25+ more
print(f"Available voices: {ELEVENLABS_VOICE_NAMES}")

# Basic usage
stream_tts_elevenlabs(
    text_chunks,
    voice_id="rachel",  # Use friendly name or voice ID
    model_id="eleven_multilingual_v2",
    stability=0.5,
    similarity_boost=0.75
)

# High-quality streaming for web
generator = stream_tts_elevenlabs(
    text_chunks,
    voice_id="domi",
    output_format="mp3_44100_128",  # Recommended for web
    return_generator=True
)
```

### Speech-to-Text

```python
from voice_agents import speech_to_text, record_audio
import numpy as np

# From audio file
transcription = speech_to_text(
    audio_file_path="recording.wav",
    model="whisper-1",
    language="en",  # Optional: auto-detect if None
    response_format="text"
)

# From numpy array (recorded audio)
audio = record_audio(duration=5.0, sample_rate=16000)
transcription = speech_to_text(
    audio_data=audio,
    sample_rate=16000,
    prompt="This is a technical conversation about AI"  # Optional context
)

# Get structured output
result = speech_to_text(
    audio_file_path="meeting.mp3",
    response_format="verbose_json"  # Returns detailed metadata
)
```

### Audio Recording

```python
from voice_agents import record_audio

# Record 5 seconds of audio
audio = record_audio(duration=5.0, sample_rate=16000, channels=1)

# Use with speech-to-text
from voice_agents import speech_to_text
text = speech_to_text(audio_data=audio, sample_rate=16000)
```

### Streaming TTS Callback for Agents

```python
from voice_agents import StreamingTTSCallback

# Create callback for real-time agent responses
tts_callback = StreamingTTSCallback(
    voice="alloy",
    model="openai/tts-1",
    min_sentence_length=10  # Minimum chars before speaking
)

# Use with any streaming text generator
def agent_stream():
    for chunk in some_agent.generate():
        tts_callback(chunk)  # Automatically speaks complete sentences
    tts_callback.flush()  # Speak any remaining text
```

### Audio Format Utilities

```python
from voice_agents import get_media_type_for_format

# Get MIME type for FastAPI
media_type = get_media_type_for_format("mp3_44100_128")
# Returns: "audio/mpeg"

media_type = get_media_type_for_format("pcm_44100")
# Returns: "audio/pcm"
```

---

## Swarms Integration

Voice-Agents is designed to work seamlessly with [Swarms](https://github.com/kyegomez/swarms), the enterprise-grade multi-agent orchestration framework.

### Complete Example: Voice-Enabled Trading Agent

```python
from swarms import Agent
from voice_agents import StreamingTTSCallback, format_text_for_speech

# Initialize the Swarms agent
agent = Agent(
    agent_name="Quantitative-Trading-Agent",
    agent_description="Advanced quantitative trading and algorithmic analysis agent",
    model_name="gpt-4",
    dynamic_temperature_enabled=True,
    max_loops=1,
    dynamic_context_window=True,
    top_p=None,
    streaming_on=True,
    interactive=False,
)

# Create the streaming TTS callback
tts_callback = StreamingTTSCallback(voice="alloy", model="openai/tts-1")

# Run the agent with streaming TTS callback
out = agent.run(
    task="What are the top five best energy stocks across nuclear, solar, gas, and other energy sources?",
    streaming_callback=tts_callback,
)

# Flush any remaining text in the buffer
tts_callback.flush()

print(out)
```

---

## Examples

The `examples/` directory contains comprehensive examples demonstrating all features of Voice-Agents, organized into logical categories. See the [Examples README](examples/README.md) for detailed documentation.

### Examples by Category

#### Text-to-Speech (`examples/text_to_speech/`)

| Example File | Description |
|-------------|-------------|
| [`example_stream_tts.py`](examples/text_to_speech/example_stream_tts.py) | Unified TTS with OpenAI models, `list_models()` |
| [`example_stream_tts_elevenlabs.py`](examples/text_to_speech/example_stream_tts_elevenlabs.py) | ElevenLabs TTS, unified and direct functions |
| [`example_streaming_tts_callback.py`](examples/text_to_speech/example_streaming_tts_callback.py) | StreamingTTSCallback for real-time TTS |
| [`example_voice_selection.py`](examples/text_to_speech/example_voice_selection.py) | Voice selection with `list_voices()` |

#### Speech-to-Text (`examples/speech_to_text/`)

| Example File | Description |
|-------------|-------------|
| [`example_speech_to_text.py`](examples/speech_to_text/example_speech_to_text.py) | OpenAI Whisper transcription |
| [`example_speech_to_text_elevenlabs_file.py`](examples/speech_to_text/example_speech_to_text_elevenlabs_file.py) | ElevenLabs STT from audio file |
| [`example_speech_to_text_elevenlabs_audio.py`](examples/speech_to_text/example_speech_to_text_elevenlabs_audio.py) | ElevenLabs STT from audio data |

#### Utilities (`examples/utilities/`)

| Example File | Description |
|-------------|-------------|
| [`example_format_text_for_speech.py`](examples/utilities/example_format_text_for_speech.py) | Text formatting for speech with abbreviation handling |
| [`example_play_audio.py`](examples/utilities/example_play_audio.py) | Audio playback and tone generation |
| [`example_record_audio.py`](examples/utilities/example_record_audio.py) | Microphone audio recording |
| [`example_get_media_type.py`](examples/utilities/example_get_media_type.py) | Media type (MIME) utilities for FastAPI |

#### Workflows (`examples/workflows/`)

| Example File | Description |
|-------------|-------------|
| [`example_complete_voice_agent.py`](examples/workflows/example_complete_voice_agent.py) | Complete voice agent workflows |

### Running Examples

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

For more details, see the [Examples README](examples/README.md).

---

## API Reference

### Constants

- `SAMPLE_RATE`: Default sample rate (24000 Hz)
- `VOICES`: List of available OpenAI voices
- `ELEVENLABS_VOICES`: Dictionary mapping friendly names to ElevenLabs voice IDs
- `ELEVENLABS_VOICE_NAMES`: List of available ElevenLabs voice names
- `OPENAI_TTS_MODELS`: List of available OpenAI TTS models
- `ELEVENLABS_TTS_MODELS`: List of available ElevenLabs TTS models
- `VoiceType`: Type alias for OpenAI voice options

### Functions

#### `format_text_for_speech(text: str) -> List[str]`
Intelligently formats text into speech-friendly chunks by detecting sentence boundaries, handling abbreviations, and preserving natural pauses.

#### `stream_tts(text_chunks, model, voice, stream_mode, response_format, return_generator)`
Unified TTS function supporting both OpenAI and ElevenLabs. Model format: `"provider/model_name"` (e.g., `"openai/tts-1"`, `"elevenlabs/eleven_multilingual_v2"`). Returns generator for web streaming or plays audio directly.

#### `list_models() -> List[dict]`
List all available TTS models with their providers. Returns list of dictionaries with `model`, `provider`, and `model_name` keys.

#### `list_voices() -> List[dict]`
List all available voices from all providers. Returns list of dictionaries with `voice`, `provider`, `voice_id`, and `description` keys.

#### `stream_tts_openai(text_chunks, voice, model, stream_mode, response_format, return_generator)`
OpenAI TTS with streaming support. Returns generator for web streaming or plays audio directly.

#### `stream_tts_elevenlabs(text_chunks, voice_id, model_id, stability, similarity_boost, output_format, return_generator)`
ElevenLabs TTS with advanced voice control and multiple output formats.

#### `speech_to_text(audio_file_path, audio_data, sample_rate, model, language, prompt, response_format)`
OpenAI Whisper transcription with support for files or numpy arrays.

#### `speech_to_text_elevenlabs(audio_file_path, audio_data, sample_rate, realtime, model_id, ...)`
ElevenLabs Speech-to-Text with support for both real-time (WebSocket) and non-real-time (file upload) modes. Supports speaker diarization, timestamps, and language detection.

#### `record_audio(duration, sample_rate, channels) -> np.ndarray`
Record audio from default microphone. Returns numpy array.

#### `play_audio(audio_data: np.ndarray)`
Play audio data using sounddevice.

#### `get_media_type_for_format(output_format: str) -> str`
Get MIME type for audio format (useful for FastAPI).

### Classes

#### `StreamingTTSCallback`
Real-time TTS callback for agent streaming outputs. Automatically detects complete sentences and converts them to speech.

**Methods:**
- `__call__(chunk: str)`: Process streaming text chunk
- `flush()`: Speak any remaining buffered text

---

## Use Cases

### Conversational AI Assistants
Build voice-enabled chatbots and virtual assistants with natural, real-time speech synthesis.

### Agent Narration
Provide audio feedback for long-running agent tasks, making agent behavior transparent and engaging.

### Voice-Enabled Analytics
Create voice interfaces for data analysis, trading systems, and business intelligence tools.

### Real-Time Transcription
Transcribe meetings, interviews, and conversations with high accuracy using Whisper.

### Multi-Modal Applications
Combine voice input/output with visual interfaces for rich, interactive experiences.

---

## Configuration

### Environment Variables

```bash
# Required for OpenAI TTS and STT
OPENAI_API_KEY=your-key-here

# Required for ElevenLabs TTS
ELEVENLABS_API_KEY=your-key-here
```

### Voice Selection

**OpenAI Voices:**
- `alloy`, `ash`, `ballad`, `coral`, `echo`, `fable`, `nova`, `onyx`, `sage`, `shimmer`

**ElevenLabs Voices:**
- Professional: `rachel`, `nicole`, `grace`
- Expressive: `domi`, `elli`, `bella`
- Deep: `antoni`, `josh`, `clyde`
- And 20+ more (see `ELEVENLABS_VOICE_NAMES`)

---

## Contributing

Voice-Agents is built by the community, for the community. We welcome contributions!

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Commit your changes**: `git commit -m 'Add amazing feature'`
5. **Push to the branch**: `git push origin feature/amazing-feature`
6. **Open a Pull Request**

### Development Setup

```bash
git clone https://github.com/The-Swarm-Corporation/Voice-Agents.git
cd Voice-Agents
pip install -e ".[dev]"
pre-commit install
```

### Code Standards

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include docstrings for all public APIs
- Write tests for new features

---

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Built by [Swarms Corporation](https://swarms.ai)
- Part of the [Swarms](https://github.com/kyegomez/swarms) ecosystem
- Powered by OpenAI, ElevenLabs, and Groq APIs

---

## Support & Community

- **Documentation**: [GitHub Repository](https://github.com/The-Swarm-Corporation/Voice-Agents)
- **Swarms Documentation**: [docs.swarms.world](https://docs.swarms.world)
- **Swarms Community**: [Discord](https://discord.gg/EamjgSaEQf)
- **Issues**: [GitHub Issues](https://github.com/The-Swarm-Corporation/Voice-Agents/issues)

---

<div align="center">

**Made by [Swarms Corporation](https://swarms.ai)**

[Website](https://swarms.ai) • [Documentation](https://docs.swarms.world) • [GitHub](https://github.com/kyegomez/swarms)

</div>
