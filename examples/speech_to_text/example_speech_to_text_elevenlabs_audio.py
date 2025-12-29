from voice_agents import record_audio, speech_to_text_elevenlabs

audio_data = record_audio(duration=5.0, sample_rate=16000)

text = speech_to_text_elevenlabs(
    audio_data=audio_data,
    sample_rate=16000,
    model_id="scribe_v1",
    realtime=False,
)
