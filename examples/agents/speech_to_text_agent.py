from swarms import Agent
from voice_agents import record_audio, speech_to_text_groq

agent = Agent(
    agent_name="Quantitative-Trading-Agent",
    agent_description="Advanced quantitative trading and algorithmic analysis agent",
    model_name="gpt-4.1",
    dynamic_temperature_enabled=True,
    max_loops=1,
    dynamic_context_window=True,
    top_p=None,
    streaming_on=True,
)

audio_data = record_audio(duration=5.0, sample_rate=16000)

transcribed_text = speech_to_text_groq(
    audio_data=audio_data,
    sample_rate=16000,
    model="whisper-large-v3-turbo",
    response_format="text",
)

out = agent.run(task=transcribed_text)
