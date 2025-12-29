from voice_agents import StreamingTTSCallback

callback = StreamingTTSCallback(
    voice="alloy", model="openai/tts-1", formatting=False
)

callback("Hello, world! My name is Kye Gomez.")

callback.flush()
