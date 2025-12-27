"""
Example: play_audio

Demonstrates how to play audio data using the play_audio function.
This example generates a simple sine wave tone and plays it.
"""

import numpy as np
from voice_agents import play_audio, SAMPLE_RATE

# Generate a simple sine wave tone (440 Hz for 2 seconds)
duration = 2.0  # seconds
frequency = 440  # Hz (A4 note)

# Generate time array
t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)

# Generate sine wave
wave = np.sin(2 * np.pi * frequency * t)

# Convert to int16 format (required by play_audio)
# Scale to int16 range [-32768, 32767]
audio_data = (wave * 32767).astype(np.int16)

print(f"Playing {frequency} Hz tone for {duration} seconds...")
print(f"Sample rate: {SAMPLE_RATE} Hz")
play_audio(audio_data)
print("Playback complete!")

# Example 2: Play a chord (multiple frequencies)
print("\nPlaying a chord (A, C#, E)...")
duration = 2.0
frequencies = [440, 554.37, 659.25]  # A, C#, E

t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
chord = np.zeros_like(t)

for freq in frequencies:
    chord += np.sin(2 * np.pi * freq * t)

# Normalize to prevent clipping
chord = chord / len(frequencies)

# Convert to int16
audio_data = (chord * 32767).astype(np.int16)

play_audio(audio_data)
print("Chord playback complete!")

