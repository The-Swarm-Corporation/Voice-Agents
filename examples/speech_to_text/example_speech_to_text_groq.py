from voice_agents import speech_to_text_groq, record_audio

audio_data = record_audio(duration=5.0)

text = speech_to_text_groq(
    audio_data=audio_data,
    model="whisper-large-v3-turbo",
    response_format="text",
)
