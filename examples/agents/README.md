# Agents Examples

This folder contains examples demonstrating integration of voice agents with agent frameworks like Swarms.

## Examples

| Example | Description | Framework | Features |
|---------|-------------|-----------|----------|
| [`speech_to_text_agent.py`](speech_to_text_agent.py) | Agent with speech-to-text input | Swarms | Voice input to agent, transcription integration |

## Quick Start

### Prerequisites

1. Install required dependencies:
   ```bash
   pip install voice_agents
   pip install swarms  # For agent framework
   ```

2. Set API keys:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export GROQ_API_KEY="your-groq-api-key"
   ```

### Running Examples

```bash
# Speech-to-text agent
python examples/agents/speech_to_text_agent.py
```

## Example Details

### Speech-to-Text Agent

Integrate voice input with an agent:

```python
from swarms import Agent
from voice_agents import record_audio, speech_to_text_groq

# Create agent
agent = Agent(
    agent_name="Quantitative-Trading-Agent",
    agent_description="Advanced quantitative trading and algorithmic analysis agent",
    model_name="gpt-4.1",
    dynamic_temperature_enabled=True,
    max_loops=1,
    dynamic_context_window=True,
    top_p=None,
    streaming_on=True,
)

# Record audio
audio_data = record_audio(duration=5.0, sample_rate=16000)

# Transcribe
transcribed_text = speech_to_text_groq(
    audio_data=audio_data,
    sample_rate=16000,
    model="whisper-large-v3-turbo",
    response_format="text",
)

# Run agent with transcribed text
out = agent.run(task=transcribed_text)
```

## Integration Patterns

### Voice Input → Agent → Text Output

```python
from swarms import Agent
from voice_agents import record_audio, speech_to_text

# Setup
agent = Agent(...)
audio = record_audio(duration=5.0)
text = speech_to_text(audio_data=audio, sample_rate=16000)

# Process
result = agent.run(task=text)
print(result)
```

### Voice Input → Agent → Voice Output

```python
from swarms import Agent
from voice_agents import record_audio, speech_to_text, stream_tts, format_text_for_speech

# Setup
agent = Agent(...)

# Input
audio = record_audio(duration=5.0)
text = speech_to_text(audio_data=audio, sample_rate=16000)

# Process
result = agent.run(task=text)

# Output
chunks = format_text_for_speech(result)
stream_tts(chunks, model="openai/tts-1", voice="alloy")
```

## Supported Agent Frameworks

### Swarms
- Full integration support
- Streaming responses
- Dynamic configuration

### Other Frameworks
- Can be adapted to work with any agent framework
- Use voice_agents functions for I/O
- Integrate agent.run() calls

## Troubleshooting

### Agent Framework Issues
- Ensure agent framework is properly installed
- Check API keys for agent framework
- Verify agent configuration is correct

### Transcription Issues
- Check microphone permissions
- Verify audio recording is working
- Ensure transcription API keys are set

### Integration Issues
- Verify audio data format matches expected input
- Check sample rate compatibility
- Ensure text format matches agent expectations

