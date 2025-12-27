"""
Example: stream_tts_elevenlabs

Demonstrates how to use ElevenLabs TTS API to convert text to speech.
Shows different voices, formats, and streaming modes.
"""

from voice_agents import stream_tts_elevenlabs, format_text_for_speech, ELEVENLABS_VOICE_NAMES

# Example 1: Simple TTS with friendly voice name
print("Example 1: Simple TTS with friendly voice name (rachel)")
text1 = "Hello! This is a test of ElevenLabs text-to-speech API."
stream_tts_elevenlabs([text1], voice_id="rachel")
print("Playback complete!\n")

# Example 2: Using different ElevenLabs voices
print("Example 2: Trying different ElevenLabs voices")
text2 = "This is a demonstration of different voice options."
voices_to_try = ["rachel", "domi", "antoni", "bella"]
for voice in voices_to_try:
    print(f"Playing with voice: {voice}")
    stream_tts_elevenlabs([text2], voice_id=voice)
    print()

# Example 3: Using formatted text chunks
print("Example 3: Using formatted text chunks")
long_text = """
Welcome to ElevenLabs text-to-speech! 
This API provides high-quality, natural-sounding voices.
You can customize stability and similarity settings.
"""
chunks = format_text_for_speech(long_text)
print(f"Text split into {len(chunks)} chunks")
print("Playing formatted chunks...")
stream_tts_elevenlabs(chunks, voice_id="rachel")
print("Playback complete!\n")

# Example 4: Custom voice settings
print("Example 4: Custom voice settings")
text4 = "This voice has custom stability and similarity boost settings."
print("Playing with default settings...")
stream_tts_elevenlabs([text4], voice_id="rachel")
print("Playing with high stability (0.8) and similarity (0.9)...")
stream_tts_elevenlabs(
    [text4], 
    voice_id="rachel",
    stability=0.8,
    similarity_boost=0.9
)
print("Settings comparison complete!\n")

# Example 5: Stream mode
print("Example 5: Stream mode (real-time processing)")
text_chunks = [
    "First chunk of text.",
    "Second chunk of text.",
    "Third chunk of text."
]
print("Processing chunks in stream mode...")
stream_tts_elevenlabs(text_chunks, voice_id="rachel", stream_mode=True)
print("Stream mode playback complete!\n")

# Example 6: Different output formats
print("Example 6: Different output formats")
text6 = "This demonstrates different audio formats."
print("Note: PCM formats are supported for playback.")
print("MP3 and Opus formats require return_generator=True for API use.")
print("Playing with PCM 44100 format...")
stream_tts_elevenlabs([text6], voice_id="rachel", output_format="pcm_44100")
print("Format example complete!\n")

# Example 7: List available voices
print("Example 7: Available ElevenLabs voices")
print(f"Total voices available: {len(ELEVENLABS_VOICE_NAMES)}")
print("First 10 voices:")
for i, voice in enumerate(ELEVENLABS_VOICE_NAMES[:10], 1):
    print(f"  {i}. {voice}")
print("...")

# Example 8: Generator mode (for API streaming)
print("\nExample 8: Generator mode (for FastAPI/API usage)")
print("Note: This example shows how to get a generator, but doesn't play audio.")
text8 = "This would be streamed in an API response."

# Get generator (for use with FastAPI StreamingResponse)
# generator = stream_tts_elevenlabs(
#     [text8], 
#     voice_id="rachel",
#     output_format="mp3_44100_128",
#     return_generator=True
# )
# 
# # In a FastAPI endpoint, you would use:
# # from fastapi.responses import StreamingResponse
# # from voice_agents import get_media_type_for_format
# # media_type = get_media_type_for_format("mp3_44100_128")
# # return StreamingResponse(generator, media_type=media_type)
print("Generator mode example (commented out - requires FastAPI setup)")

