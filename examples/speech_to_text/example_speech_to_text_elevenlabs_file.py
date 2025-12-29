from voice_agents import speech_to_text_elevenlabs

text = speech_to_text_elevenlabs(
    audio_file_path="path/to/your/audio.wav",
    model_id="scribe_v1",
    realtime=False,
)
