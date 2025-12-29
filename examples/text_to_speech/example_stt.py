from voice_agents.main import stream_tts

# Example: Simple TTS with default voice and model
text1 = "Hello! This is a test of OpenAI's text-to-speech API."

print("Starting TTS generation...")
try:
    stream_tts(
        text_chunks=[text1],
        model="openai/tts-1",
        voice="alloy",
        stream_mode=False,  # Set to False for more reliable playback
        response_format="pcm",
        verbose=True,
    )
    print("TTS generation completed!")
except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
