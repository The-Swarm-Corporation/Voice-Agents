"""
Example: speech_to_text_elevenlabs (File-based, Non-real-time)

Demonstrates how to use ElevenLabs Speech-to-Text API to convert speech to text
from an audio/video file.
"""

from voice_agents import speech_to_text_elevenlabs

# Transcribe audio from file
text = speech_to_text_elevenlabs(
    audio_file_path="path/to/your/audio.wav",
    model_id="scribe_v1",
    realtime=False
)

print(f"Transcribed text: {text}")
