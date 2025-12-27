"""
Example: speech_to_text_elevenlabs (Word-level Timestamps)

Demonstrates how to use ElevenLabs Speech-to-Text API with word-level timestamps
for precise timing information.
"""

from voice_agents import speech_to_text_elevenlabs

# Transcribe with word-level timestamps
text = speech_to_text_elevenlabs(
    audio_file_path="path/to/your/audio.wav",
    model_id="scribe_v1",
    timestamps_granularity="word",
    realtime=False
)

print(f"Transcribed text with timestamps: {text}")

