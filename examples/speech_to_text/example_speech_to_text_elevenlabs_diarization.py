"""
Example: speech_to_text_elevenlabs (Speaker Diarization)

Demonstrates how to use ElevenLabs Speech-to-Text API with speaker diarization
to identify different speakers in a conversation.
"""

from voice_agents import speech_to_text_elevenlabs

# Transcribe with speaker diarization
text = speech_to_text_elevenlabs(
    audio_file_path="path/to/conversation.wav",
    model_id="scribe_v1",
    diarize=True,
    num_speakers=2,  # Expected number of speakers
    timestamps_granularity="word",
    realtime=False,
)

print(f"Transcribed text with speaker labels: {text}")
