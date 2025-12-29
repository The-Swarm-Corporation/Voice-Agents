from voice_agents import speech_to_text_elevenlabs

text = speech_to_text_elevenlabs(
    audio_file_path="path/to/conversation.wav",
    model_id="scribe_v1",
    diarize=True,
    num_speakers=2,
    timestamps_granularity="word",
    realtime=False,
)
