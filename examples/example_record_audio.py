"""
Example: record_audio

Demonstrates how to record audio from the default microphone.
Shows different recording durations and sample rates.
"""

from voice_agents import record_audio
import numpy as np

# Example 1: Basic recording (default 5 seconds)
print("Example 1: Basic recording (5 seconds)")
print("Recording audio...")
audio1 = record_audio()
print(f"Recorded audio shape: {audio1.shape}")
print(f"Audio dtype: {audio1.dtype}")
print(f"Audio duration: {len(audio1) / 16000:.2f} seconds")
print("Recording complete!\n")

# Example 2: Custom duration
print("Example 2: Custom duration (3 seconds)")
print("Recording for 3 seconds...")
audio2 = record_audio(duration=3.0)
print(f"Recorded {len(audio2) / 16000:.2f} seconds of audio")
print("Recording complete!\n")

# Example 3: Different sample rates
print("Example 3: Different sample rates")
print("Recording at 16kHz (standard for speech recognition)...")
audio_16k = record_audio(duration=2.0, sample_rate=16000)
print(f"16kHz recording: {len(audio_16k)} samples")
print("Recording at 24kHz (higher quality)...")
audio_24k = record_audio(duration=2.0, sample_rate=24000)
print(f"24kHz recording: {len(audio_24k)} samples")
print("Recording at 44.1kHz (CD quality)...")
audio_44k = record_audio(duration=2.0, sample_rate=44100)
print(f"44.1kHz recording: {len(audio_44k)} samples")
print("Sample rate comparison complete!\n")

# Example 4: Save recorded audio to file
print("Example 4: Save recorded audio to file")
print("Recording 2 seconds of audio...")
audio4 = record_audio(duration=2.0)

# Save to WAV file
try:
    import soundfile as sf

    output_file = "recorded_audio.wav"
    # Convert int16 to float32 for soundfile
    audio_float = audio4.astype(np.float32) / 32768.0
    sf.write(output_file, audio_float, 16000)
    print(f"Audio saved to: {output_file}")
except ImportError:
    print(
        "soundfile library not available. Install with: pip install soundfile"
    )
print()

# Example 5: Record and analyze audio
print("Example 5: Record and analyze audio")
print("Recording 3 seconds...")
audio5 = record_audio(duration=3.0)

# Basic audio analysis
max_amplitude = np.max(np.abs(audio5))
rms = np.sqrt(np.mean(audio5.astype(np.float32) ** 2))
print(f"Maximum amplitude: {max_amplitude}")
print(f"RMS (root mean square): {rms:.2f}")
print(
    f"Audio level: {'Loud' if rms > 1000 else 'Quiet' if rms < 100 else 'Normal'}"
)
print()

# Example 6: Record multiple segments
print("Example 6: Record multiple segments")
print("This will record 3 separate 2-second segments...")
segments = []
for i in range(3):
    print(f"Recording segment {i+1}/3...")
    segment = record_audio(duration=2.0)
    segments.append(segment)
    print(f"Segment {i+1} recorded: {len(segment)} samples")

# Concatenate segments
if segments:
    combined = np.concatenate(segments)
    print(
        f"Combined audio: {len(combined)} samples ({len(combined)/16000:.2f} seconds)"
    )
print("Multiple segment recording complete!\n")

# Example 7: Record with different channels
print("Example 7: Record with different channels")
print("Recording mono audio (1 channel)...")
audio_mono = record_audio(duration=2.0, channels=1)
print(f"Mono audio shape: {audio_mono.shape}")
print("Note: Stereo recording (channels=2) is also supported")
print()

# Example 8: Use recorded audio with speech_to_text
print("Example 8: Use recorded audio with speech_to_text")
print("This demonstrates recording and then transcribing:")
print(
    """
from voice_agents import record_audio, speech_to_text

# Record audio
print("Recording...")
audio = record_audio(duration=5.0, sample_rate=16000)

# Transcribe
print("Transcribing...")
text = speech_to_text(audio_data=audio, sample_rate=16000)
print(f"Transcribed text: {text}")
"""
)
print(
    "Example 8 skipped (commented out to avoid accidental recording)"
)
