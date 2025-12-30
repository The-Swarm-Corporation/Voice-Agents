# Workflows Examples

This folder contains complete workflow examples that demonstrate end-to-end voice agent functionality, combining multiple voice_agents functions into practical use cases.

## Examples

| Example | Description | Components |
|---------|-------------|------------|
| [`example_complete_voice_agent.py`](example_complete_voice_agent.py) | Complete conversational voice agent | Recording, transcription, TTS, text formatting |

## Quick Start

### Prerequisites

1. Install required dependencies:
   ```bash
   pip install voice_agents
   ```

2. Set API keys:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

### Running Examples

```bash
# Complete voice agent
python examples/workflows/example_complete_voice_agent.py
```

## Example Details

### Complete Voice Agent

A full conversational voice agent that:
1. Records audio from microphone
2. Transcribes speech to text
3. Processes the text (simple echo in this example)
4. Formats response for speech
5. Converts response to speech

```python
from voice_agents import (
    record_audio,
    speech_to_text,
    stream_tts,
    format_text_for_speech,
)

# Record audio
audio = record_audio(duration=5.0, sample_rate=16000)

# Transcribe
user_text = speech_to_text(audio_data=audio, sample_rate=16000)

# Process (simple echo in this example)
response = f"I heard you say: {user_text}."

# Format for speech
chunks = format_text_for_speech(response)

# Speak response
stream_tts(chunks, model="openai/tts-1", voice="alloy")
```

## Workflow Patterns

### Basic Voice Loop

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

### Voice Loop with Agent

```python
from swarms import Agent
from voice_agents import record_audio, speech_to_text, stream_tts, format_text_for_speech

agent = Agent(...)

# Record
audio = record_audio(duration=5.0)
text = speech_to_text(audio_data=audio, sample_rate=16000)

# Process with agent
result = agent.run(task=text)

# Respond
chunks = format_text_for_speech(result)
stream_tts(chunks, model="openai/tts-1", voice="alloy")
```

### Continuous Conversation Loop

```python
from voice_agents import record_audio, speech_to_text, stream_tts, format_text_for_speech

while True:
    # Record
    audio = record_audio(duration=5.0)
    text = speech_to_text(audio_data=audio, sample_rate=16000)
    
    # Check for exit
    if text.lower() in ["exit", "quit", "stop"]:
        break
    
    # Process and respond
    response = f"You said: {text}"
    chunks = format_text_for_speech(response)
    stream_tts(chunks, model="openai/tts-1", voice="alloy")
```

## Use Cases

### Voice Assistant
Create a voice assistant that listens and responds to user input.

### Voice-Enabled Chatbot
Integrate voice I/O with chatbot functionality.

### Voice Command System
Process voice commands and execute actions.

### Voice Note Taker
Record and transcribe voice notes.

### Interactive Voice Application
Build interactive applications with voice interface.

## Best Practices

### Error Handling
Always include error handling for:
- Audio recording failures
- Transcription errors
- TTS generation issues
- Network connectivity problems

### Audio Quality
- Use appropriate sample rates (16000 Hz recommended)
- Ensure good microphone quality
- Minimize background noise

### Response Formatting
- Always format text before TTS
- Split long responses into chunks
- Handle special characters appropriately

### Performance
- Use streaming TTS for better responsiveness
- Consider async operations for better UX
- Cache frequently used responses

## Troubleshooting

### Complete Workflow Issues
- Check all API keys are set
- Verify microphone is working
- Ensure audio playback is functional
- Test each component individually

### Audio Recording Issues
- Check microphone permissions
- Verify microphone is connected
- Test with system recording tools

### Transcription Issues
- Ensure audio quality is good
- Check API key is valid
- Verify network connectivity

### TTS Issues
- Check API key is valid
- Verify text formatting is correct
- Ensure audio playback is working

