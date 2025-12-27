"""
Example: speech_to_text

Demonstrates how to use OpenAI's Whisper API to convert speech to text.
Shows both file-based and audio data transcription.
"""

from voice_agents import speech_to_text, record_audio
import numpy as np
import sounddevice as sd

# Example 1: Transcribe from audio file
print("Example 1: Transcribe from audio file")
print("Note: This requires an audio file. Uncomment and provide a file path.")
# text = speech_to_text(audio_file_path="path/to/your/audio.wav")
# print(f"Transcribed text: {text}")
print("Example 1 skipped (no audio file provided)\n")

# Example 2: Record and transcribe audio
print("Example 2: Record and transcribe audio")
print("This will record 3 seconds of audio from your microphone...")
print("Make sure your microphone is connected and working.")
print()

# Uncomment to actually record and transcribe:
# audio_data = record_audio(duration=3.0, sample_rate=16000)
# print("Recording complete. Transcribing...")
# text = speech_to_text(audio_data=audio_data, sample_rate=16000)
# print(f"Transcribed text: {text}")
print("Example 2 skipped (commented out to avoid accidental recording)\n")

# Example 3: Transcribe with different models
print("Example 3: Transcribe with different models")
print("Note: This requires an audio file.")
# text_whisper1 = speech_to_text(
#     audio_file_path="audio.wav",
#     model="whisper-1"
# )
# print(f"Whisper-1 result: {text_whisper1}")
print("Example 3 skipped (no audio file provided)\n")

# Example 4: Transcribe with language specification
print("Example 4: Transcribe with language specification")
print("Note: This requires an audio file.")
# text_english = speech_to_text(
#     audio_file_path="audio.wav",
#     language="en"
# )
# text_spanish = speech_to_text(
#     audio_file_path="audio.wav",
#     language="es"
# )
# print(f"English transcription: {text_english}")
# print(f"Spanish transcription: {text_spanish}")
print("Example 4 skipped (no audio file provided)\n")

# Example 5: Transcribe with prompt (for context)
print("Example 5: Transcribe with prompt for context")
print("Note: This requires an audio file.")
# text_with_prompt = speech_to_text(
#     audio_file_path="audio.wav",
#     prompt="This is a technical discussion about artificial intelligence."
# )
# print(f"Transcription with prompt: {text_with_prompt}")
print("Example 5 skipped (no audio file provided)\n")

# Example 6: Different response formats
print("Example 6: Different response formats")
print("Note: This requires an audio file.")
# text_format = speech_to_text(
#     audio_file_path="audio.wav",
#     response_format="text"
# )
# json_format = speech_to_text(
#     audio_file_path="audio.wav",
#     response_format="json"
# )
# srt_format = speech_to_text(
#     audio_file_path="audio.wav",
#     response_format="srt"
# )
# print(f"Text format: {text_format}")
# print(f"JSON format: {json_format}")
# print(f"SRT format (first 200 chars): {srt_format[:200]}")
print("Example 6 skipped (no audio file provided)\n")

# Example 7: Complete workflow - Record, transcribe, and speak back
print("Example 7: Complete workflow")
print("This demonstrates a full voice interaction loop:")
print("1. Record audio from microphone")
print("2. Transcribe to text")
print("3. Convert text back to speech")
print()
print("Uncomment to run:")
print("""
from voice_agents import record_audio, speech_to_text, stream_tts

# Step 1: Record
print("Recording...")
audio = record_audio(duration=5.0, sample_rate=16000)

# Step 2: Transcribe
print("Transcribing...")
text = speech_to_text(audio_data=audio, sample_rate=16000)
print(f"You said: {text}")

# Step 3: Speak back
print("Speaking back...")
stream_tts([f"You said: {text}"], voice="alloy")
""")

