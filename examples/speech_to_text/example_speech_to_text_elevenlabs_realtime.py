from voice_agents import record_audio, speech_to_text_elevenlabs

audio_data = record_audio(duration=5.0, sample_rate=16000)

for message in speech_to_text_elevenlabs(
    audio_data=audio_data,
    sample_rate=16000,
    realtime=True,
    audio_format="pcm_16000",
    commit_strategy="vad",
):
    message_type = message.get("message_type")
    if message_type == "committed_transcript":
        text = message.get("text")
