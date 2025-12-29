from voice_agents import stream_tts

stream_tts(
    [
        "Hello! This is a test of ElevenLabs text-to-speech API using the unified function."
    ],
    model="elevenlabs/eleven_multilingual_v2",
    voice="rachel",
)
