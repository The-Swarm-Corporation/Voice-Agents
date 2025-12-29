from voice_agents.main import stream_tts_openai

# Example: Simple TTS with default voice and model
text1 = "Hello! This is a test of OpenAI's text-to-speech API."

print("Starting TTS generation...")
try:
    stream_tts_openai(
        text_chunks=[text1],
        model="tts-1",  # Note: stream_tts_openai expects "tts-1", not "openai/tts-1"
        voice="alloy",
        stream_mode=False,  # Changed to False - processes all text at once, more reliable
        response_format="pcm",  # Changed to PCM - works without pydub/simpleaudio dependencies
    )
    print("TTS generation completed!")
except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
