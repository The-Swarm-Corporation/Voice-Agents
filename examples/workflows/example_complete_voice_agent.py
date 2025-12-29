from voice_agents import (
    record_audio,
    speech_to_text,
    stream_tts,
    format_text_for_speech,
)

audio = record_audio(duration=5.0, sample_rate=16000)

user_text = speech_to_text(audio_data=audio, sample_rate=16000)

response = f"I heard you say: {user_text}."

chunks = format_text_for_speech(response)
stream_tts(chunks, model="openai/tts-1", voice="alloy")
