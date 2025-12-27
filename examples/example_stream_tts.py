"""
Example: stream_tts (OpenAI TTS)

Demonstrates how to use OpenAI's TTS API to convert text to speech.
Shows different modes: batch processing, streaming, and generator mode.
"""

from voice_agents import stream_tts, format_text_for_speech, VOICES

# Example 1: Simple TTS with default voice
print("Example 1: Simple TTS with default voice (alloy)")
text1 = "Hello! This is a test of OpenAI's text-to-speech API."
stream_tts([text1], voice="alloy")
print("Playback complete!\n")

# Example 2: TTS with different voices
print("Example 2: Trying different voices")
text2 = "This is the nova voice. It sounds different from alloy."
for voice in ["alloy", "nova", "shimmer"]:
    print(f"Playing with voice: {voice}")
    stream_tts([text2], voice=voice)
    print()

# Example 3: Using formatted text chunks
print("Example 3: Using formatted text chunks")
long_text = """
Welcome to the Voice Agents library! 
This library provides powerful tools for text-to-speech conversion.
You can use it to build voice-enabled applications and agents.
"""
chunks = format_text_for_speech(long_text)
print(f"Text split into {len(chunks)} chunks:")
for i, chunk in enumerate(chunks, 1):
    print(f"  {i}. {chunk}")
print("\nPlaying formatted chunks...")
stream_tts(chunks, voice="alloy")
print("Playback complete!\n")

# Example 4: Stream mode (process chunks as they arrive)
print("Example 4: Stream mode (real-time processing)")
text_chunks = [
    "First chunk of text.",
    "Second chunk of text.",
    "Third chunk of text."
]
print("Processing chunks in stream mode...")
stream_tts(text_chunks, voice="alloy", stream_mode=True)
print("Stream mode playback complete!\n")

# Example 5: Different models
print("Example 5: Using different TTS models")
text5 = "This is using the tts-1 model."
print("Using tts-1 model...")
stream_tts([text5], voice="alloy", model="tts-1")
print("Using tts-1-hd model (higher quality)...")
stream_tts([text5], voice="alloy", model="tts-1-hd")
print("Model comparison complete!\n")

# Example 6: Generator mode (for API streaming)
print("Example 6: Generator mode (for FastAPI/API usage)")
print("Note: This example shows how to get a generator, but doesn't play audio.")
text6 = "This would be streamed in an API response."

# Get generator (for use with FastAPI StreamingResponse)
# generator = stream_tts([text6], voice="alloy", return_generator=True)
# 
# # In a FastAPI endpoint, you would use:
# # from fastapi.responses import StreamingResponse
# # return StreamingResponse(generator, media_type="audio/pcm")
print("Generator mode example (commented out - requires FastAPI setup)")

