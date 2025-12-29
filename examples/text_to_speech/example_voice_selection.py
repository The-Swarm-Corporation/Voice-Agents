from voice_agents import stream_tts, VOICES

stream_tts(
    ["This demonstrates voice selection."],
    model="openai/tts-1",
    voice=VOICES[0],
)
