# Speech-to-Text Examples

This folder contains examples demonstrating speech-to-text (STT) functionality using various providers including ElevenLabs and Groq.

## Examples

| Example | Description | Provider | Features |
|---------|-------------|----------|----------|
| [`example_speech_to_text_elevenlabs_file.py`](example_speech_to_text_elevenlabs_file.py) | Transcribe audio from a file | ElevenLabs | File-based, non-real-time transcription |
| [`example_speech_to_text_elevenlabs_audio.py`](example_speech_to_text_elevenlabs_audio.py) | Transcribe audio from numpy array | ElevenLabs | Audio data transcription, non-real-time |
| [`example_speech_to_text_elevenlabs_realtime.py`](example_speech_to_text_elevenlabs_realtime.py) | Real-time transcription via WebSocket | ElevenLabs | Streaming audio, live transcription results |
| [`example_speech_to_text_elevenlabs_diarization.py`](example_speech_to_text_elevenlabs_diarization.py) | Speaker diarization | ElevenLabs | Identify different speakers in conversations |
| [`example_speech_to_text_elevenlabs_timestamps.py`](example_speech_to_text_elevenlabs_timestamps.py) | Word-level timestamps | ElevenLabs | Precise timing information for transcription |
| [`example_speech_to_text_groq.py`](example_speech_to_text_groq.py) | Fast Whisper transcription | Groq | Fast transcription, word-level timestamps, verbose JSON |

## Quick Start

### Prerequisites

1. Install required dependencies:
   ```bash
   pip install voice_agents
   pip install websockets  # For real-time ElevenLabs STT
   ```

2. Set API keys:
   ```bash
   export ELEVENLABS_API_KEY="your-elevenlabs-api-key"
   export GROQ_API_KEY="your-groq-api-key"
   ```

### Running Examples

```bash
# File-based transcription (ElevenLabs)
python examples/speech_to_text/example_speech_to_text_elevenlabs_file.py

# Audio data transcription (ElevenLabs)
python examples/speech_to_text/example_speech_to_text_elevenlabs_audio.py

# Real-time transcription (ElevenLabs)
python examples/speech_to_text/example_speech_to_text_elevenlabs_realtime.py

# Speaker diarization (ElevenLabs)
python examples/speech_to_text/example_speech_to_text_elevenlabs_diarization.py

# Word-level timestamps (ElevenLabs)
python examples/speech_to_text/example_speech_to_text_elevenlabs_timestamps.py

# Groq transcription
python examples/speech_to_text/example_speech_to_text_groq.py
```

## Example Details

### File-based Transcription (ElevenLabs)

Transcribes audio from a file path:

```python
from voice_agents import speech_to_text_elevenlabs

text = speech_to_text_elevenlabs(
    audio_file_path="path/to/your/audio.wav",
    model_id="scribe_v1",
    realtime=False,
)
```

### Audio Data Transcription (ElevenLabs)

Transcribes audio from numpy array:

```python
from voice_agents import record_audio, speech_to_text_elevenlabs

audio_data = record_audio(duration=5.0, sample_rate=16000)

text = speech_to_text_elevenlabs(
    audio_data=audio_data,
    sample_rate=16000,
    model_id="scribe_v1",
    realtime=False,
)
```

### Real-time Transcription (ElevenLabs)

Streaming transcription with live results:

```python
from voice_agents import record_audio, speech_to_text_elevenlabs

audio_data = record_audio(duration=5.0, sample_rate=16000)

for message in speech_to_text_elevenlabs(
    audio_data=audio_data,
    sample_rate=16000,
    realtime=True,
    audio_format="pcm_16000",
    commit_strategy="vad",
):
    message_type = message.get("message_type")
    if message_type == "committed_transcript":
        text = message.get("text")
        print(text)
```

### Speaker Diarization (ElevenLabs)

Identify different speakers in conversations:

```python
from voice_agents import speech_to_text_elevenlabs

text = speech_to_text_elevenlabs(
    audio_file_path="path/to/conversation.wav",
    model_id="scribe_v1",
    diarize=True,
    num_speakers=2,
    timestamps_granularity="word",
    realtime=False,
)
```

### Word-level Timestamps (ElevenLabs)

Get precise timing information:

```python
from voice_agents import speech_to_text_elevenlabs

text = speech_to_text_elevenlabs(
    audio_file_path="path/to/your/audio.wav",
    model_id="scribe_v1",
    timestamps_granularity="word",
    realtime=False,
)
```

### Groq Transcription

Fast transcription with Groq's Whisper models:

```python
from voice_agents import speech_to_text_groq, record_audio

audio_data = record_audio(duration=5.0)

text = speech_to_text_groq(
    audio_data=audio_data,
    model="whisper-large-v3-turbo",
    response_format="text",
)
```

## API Keys

### ElevenLabs API Key
Required for all ElevenLabs examples. Get your key from: https://elevenlabs.io/app/settings/api-keys

### Groq API Key
Required for Groq examples. Get your key from: https://console.groq.com/keys

## Supported Audio Formats

- **ElevenLabs**: WAV, MP3, M4A, FLAC, and more
- **Groq**: WAV, MP3, and other common formats

## Troubleshooting

### Real-time Transcription Issues
- Ensure `websockets` library is installed: `pip install websockets`
- Check network connectivity
- Verify API key has access to real-time features

### Audio Format Issues
- Ensure audio files are in supported formats
- Check sample rate compatibility (typically 16000 Hz for real-time)
- Verify audio file is not corrupted

### API Key Errors
- Verify your API keys are set correctly: `echo $ELEVENLABS_API_KEY`
- Check that keys are valid and have sufficient credits
- Ensure keys don't have extra spaces or newlines

