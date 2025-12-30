# Utilities Examples

This folder contains examples demonstrating utility functions for audio processing, text formatting, and media type handling.

## Examples

| Example | Description | Functionality |
|---------|-------------|---------------|
| [`example_record_audio.py`](example_record_audio.py) | Audio recording from microphone | Record audio with different durations and sample rates |
| [`example_play_audio.py`](example_play_audio.py) | Audio playback | Play audio data with different formats |
| [`example_format_text_for_speech.py`](example_format_text_for_speech.py) | Text formatting for speech | Format text with abbreviations, URLs, and punctuation |
| [`example_get_media_type.py`](example_get_media_type.py) | Media type utilities | Get MIME types and format validation |
| [`example.py`](example.py) | General utilities example | Basic utility function usage |

## Quick Start

### Prerequisites

1. Install required dependencies:
   ```bash
   pip install voice_agents
   pip install soundfile  # For audio file operations
   ```

2. For audio recording/playback:
   - Ensure microphone is connected and working
   - Check microphone permissions in system settings
   - On Linux, you may need: `sudo apt-get install portaudio19-dev`

### Running Examples

```bash
# Record audio
python examples/utilities/example_record_audio.py

# Play audio
python examples/utilities/example_play_audio.py

# Format text for speech
python examples/utilities/example_format_text_for_speech.py

# Get media type
python examples/utilities/example_get_media_type.py

# General utilities
python examples/utilities/example.py
```

## Example Details

### Record Audio

Record audio from microphone:

```python
from voice_agents import record_audio

# Record 5 seconds of audio
audio = record_audio(duration=5.0, sample_rate=16000)
```

**Parameters:**
- `duration`: Recording duration in seconds (default: 5.0)
- `sample_rate`: Sample rate in Hz (default: 16000)

**Returns:**
- NumPy array of audio data

### Play Audio

Play audio data:

```python
from voice_agents import play_audio

# Play audio data
play_audio(audio_data, sample_rate=16000)
```

**Parameters:**
- `audio_data`: NumPy array of audio data
- `sample_rate`: Sample rate in Hz (default: 16000)

### Format Text for Speech

Format text to be speech-friendly:

```python
from voice_agents import format_text_for_speech

text = "Hello world! This is a test. How are you today?"
chunks = format_text_for_speech(text)
```

**Features:**
- Handles abbreviations (e.g., "Dr.", "Mr.", "etc.")
- Processes URLs appropriately
- Splits long text into chunks
- Handles punctuation for natural speech

**Returns:**
- List of text chunks ready for TTS

### Get Media Type

Get MIME type for media files:

```python
from voice_agents import get_media_type

# Get media type from file path
media_type = get_media_type("audio.wav")
# Returns: "audio/wav"

# Get media type from file extension
media_type = get_media_type("audio.mp3")
# Returns: "audio/mpeg"
```

**Use Cases:**
- FastAPI integration
- Format validation
- Content-type headers

## Common Patterns

### Record and Process

```python
from voice_agents import record_audio, speech_to_text

# Record audio
audio = record_audio(duration=5.0, sample_rate=16000)

# Transcribe
text = speech_to_text(audio_data=audio, sample_rate=16000)
```

### Format and Speak

```python
from voice_agents import format_text_for_speech, stream_tts

# Long text
long_text = "This is a very long text that needs to be formatted..."

# Format for speech
chunks = format_text_for_speech(long_text)

# Speak
stream_tts(chunks, model="openai/tts-1", voice="alloy")
```

### Audio Pipeline

```python
from voice_agents import record_audio, play_audio, speech_to_text, stream_tts, format_text_for_speech

# Record
audio = record_audio(duration=5.0)

# Transcribe
text = speech_to_text(audio_data=audio, sample_rate=16000)

# Format response
response = f"You said: {text}"
chunks = format_text_for_speech(response)

# Speak response
stream_tts(chunks, model="openai/tts-1", voice="alloy")
```

## Troubleshooting

### Microphone Issues
- Check microphone permissions in system settings
- Verify microphone is connected and working
- Test with system audio recording tools first
- On macOS, check System Preferences > Security & Privacy > Microphone

### Audio Playback Issues
- Ensure your system audio is working
- Check that `sounddevice` is properly installed
- On Linux, you may need: `sudo apt-get install portaudio19-dev`
- Verify audio data is in correct format (numpy array)

### Text Formatting Issues
- Ensure text is a string
- Check for encoding issues
- Verify text is not empty

### Media Type Issues
- Check file extension is recognized
- Verify file exists if using file path
- Ensure extension matches actual file type

## System Requirements

### macOS
- No additional setup required
- Microphone permissions may be needed

### Linux
- Install portaudio: `sudo apt-get install portaudio19-dev`
- May need ALSA or PulseAudio configured

### Windows
- No additional setup required
- Microphone permissions may be needed

