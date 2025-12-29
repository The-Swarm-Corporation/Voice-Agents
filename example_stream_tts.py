from voice_agents import stream_tts

stream_tts(
    ["Hello! This is a test of OpenAI's text-to-speech API."],
    model="openai/tts-1",
    voice="alloy",
)
