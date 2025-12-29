from voice_agents import StreamingTTSCallback

callback = StreamingTTSCallback(voice="alloy", model="openai/tts-1")

streaming_text = [
    "Hello! ",
    "This is a ",
    "streaming text ",
    "example. ",
    "The callback will ",
    "detect complete sentences ",
    "and convert them to speech.",
]

for chunk in streaming_text:
    callback(chunk)

callback.flush()
