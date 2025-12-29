"""
Example: speech_to_text_elevenlabs (Real-time WebSocket)

Demonstrates how to use ElevenLabs Speech-to-Text API for real-time streaming
transcription via WebSocket.
"""

from voice_agents import record_audio, speech_to_text_elevenlabs

# Record audio
print("Recording 5 seconds of audio...")
audio_data = record_audio(duration=5.0, sample_rate=16000)

# Real-time transcription
print("Transcribing in real-time mode...")
for message in speech_to_text_elevenlabs(
    audio_data=audio_data,
    sample_rate=16000,
    realtime=True,
    audio_format="pcm_16000",
    commit_strategy="vad",
):
    message_type = message.get("message_type")

    if message_type == "session_started":
        print(f"Session started: {message.get('session_id')}")
    elif message_type == "partial_transcript":
        print(f"Partial: {message.get('text')}")
    elif message_type == "committed_transcript":
        print(f"Committed: {message.get('text')}")
    elif message_type == "committed_transcript_with_timestamps":
        print(f"Committed with timestamps: {message.get('text')}")
        words = message.get("words", [])
        if words:
            print(
                f"Words: {[(w.get('text'), w.get('start'), w.get('end')) for w in words[:5]]}"
            )
    elif message_type in ["error", "auth_error", "quota_exceeded"]:
        print(f"Error: {message.get('error')}")
