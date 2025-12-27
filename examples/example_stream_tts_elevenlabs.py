"""
Example: Unified stream_tts with ElevenLabs

Demonstrates how to use the unified stream_tts function with ElevenLabs models.
Shows different voices, formats, and streaming modes.
Also shows how to use the direct stream_tts_elevenlabs function.
"""

from voice_agents import (
    stream_tts,
    stream_tts_elevenlabs,
    format_text_for_speech,
    list_models,
    list_voices,
    ELEVENLABS_VOICE_NAMES
)

# Example 0: List available models and voices
print("Example 0: Available Models and Voices")
models = list_models()
elevenlabs_models = [m for m in models if m['provider'] == 'elevenlabs']
print("Available ElevenLabs models:")
for model in elevenlabs_models:
    print(f"  {model['model']}")
print()

voices = list_voices()
elevenlabs_voices = [v for v in voices if v['provider'] == 'elevenlabs']
print(f"Available ElevenLabs voices: {len(elevenlabs_voices)}")
print("First 5 voices:")
for voice in elevenlabs_voices[:5]:
    print(f"  {voice['voice']} - {voice['description']}")
print()

# Example 1: Simple TTS with unified stream_tts function
print("Example 1: Using unified stream_tts with ElevenLabs")
text1 = "Hello! This is a test of ElevenLabs text-to-speech API using the unified function."
stream_tts([text1], model="elevenlabs/eleven_multilingual_v2", voice="rachel")
print("Playback complete!\n")

# Example 1b: Using direct stream_tts_elevenlabs function
print("Example 1b: Using direct stream_tts_elevenlabs function")
text1b = "This is using the direct ElevenLabs function."
stream_tts_elevenlabs([text1b], voice_id="rachel")
print("Playback complete!\n")

# Example 2: Using different ElevenLabs voices with unified function
print("Example 2: Trying different ElevenLabs voices with unified stream_tts")
text2 = "This is a demonstration of different voice options."
voices_to_try = ["rachel", "domi", "antoni", "bella"]
for voice in voices_to_try:
    print(f"Playing with voice: {voice}")
    stream_tts([text2], model="elevenlabs/eleven_multilingual_v2", voice=voice)
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
print("Playing formatted chunks with unified function...")
stream_tts(chunks, model="elevenlabs/eleven_multilingual_v2", voice="rachel")
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
print("Processing chunks in stream mode with unified function...")
stream_tts(text_chunks, model="elevenlabs/eleven_multilingual_v2", voice="rachel", stream_mode=True)
print("Stream mode playback complete!\n")

# Example 6: Different output formats
print("Example 6: Different output formats")
text6 = "This demonstrates different audio formats."
print("Note: PCM formats are supported for playback.")
print("MP3 and Opus formats require return_generator=True for API use.")
print("Playing with PCM 44100 format...")
stream_tts_elevenlabs([text6], voice_id="rachel", output_format="pcm_44100")
print("Format example complete!\n")

# Example 7: List available voices using list_voices()
print("Example 7: Available ElevenLabs voices using list_voices()")
elevenlabs_voices = [v for v in list_voices() if v['provider'] == 'elevenlabs']
print(f"Total voices available: {len(elevenlabs_voices)}")
print("First 10 voices with descriptions:")
for i, voice in enumerate(elevenlabs_voices[:10], 1):
    desc = voice['description'] or "No description"
    print(f"  {i}. {voice['voice']:15} - {desc}")
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

