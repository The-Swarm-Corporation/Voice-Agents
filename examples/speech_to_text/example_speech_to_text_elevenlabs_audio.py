"""
Example: speech_to_text_elevenlabs (Audio Data, Non-real-time)

Demonstrates how to use ElevenLabs Speech-to-Text API to convert speech to text
from recorded audio data (numpy array).
"""

from voice_agents import record_audio, speech_to_text_elevenlabs

# Record audio
print("Recording 5 seconds of audio...")
audio_data = record_audio(duration=5.0, sample_rate=16000)

# Transcribe audio data
print("Transcribing...")
text = speech_to_text_elevenlabs(
    audio_data=audio_data,
    sample_rate=16000,
    model_id="scribe_v1",
    realtime=False,
)

print(f"Transcribed text: {text}")
