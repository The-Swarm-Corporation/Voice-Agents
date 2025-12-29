import numpy as np
from voice_agents import play_audio, SAMPLE_RATE

duration = 2.0
frequency = 440

t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
wave = np.sin(2 * np.pi * frequency * t)
audio_data = (wave * 32767).astype(np.int16)

play_audio(audio_data)
