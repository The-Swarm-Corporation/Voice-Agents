from voice_agents import stream_tts_groq

stream_tts_groq(
    text_chunks=[
        "Hello! This is a test of Groq's fast text to speech using the Orpheus model."
    ],
    voice="austin",
    model="canopylabs/orpheus-v1-english",
    stream_mode=True,
)
