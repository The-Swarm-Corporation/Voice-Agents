"""
Example: Voice Selection and Constants

Demonstrates the available voices and voice selection options
for both OpenAI TTS and ElevenLabs TTS.
"""

from voice_agents import (
    VOICES,
    ELEVENLABS_VOICES,
    ELEVENLABS_VOICE_NAMES,
    VoiceType,
    SAMPLE_RATE
)

# Example 1: OpenAI TTS Voices
print("Example 1: Available OpenAI TTS Voices")
print(f"Total OpenAI voices: {len(VOICES)}")
print("Available voices:")
for i, voice in enumerate(VOICES, 1):
    print(f"  {i:2}. {voice}")
print()

# Example 2: ElevenLabs Voices
print("Example 2: Available ElevenLabs Voices")
print(f"Total ElevenLabs voices: {len(ELEVENLABS_VOICES)}")
print("First 10 voices (friendly names):")
for i, voice_name in enumerate(ELEVENLABS_VOICE_NAMES[:10], 1):
    voice_id = ELEVENLABS_VOICES[voice_name]
    print(f"  {i:2}. {voice_name:15} (ID: {voice_id})")
print(f"  ... and {len(ELEVENLABS_VOICES) - 10} more")
print()

# Example 3: Voice Type Checking
print("Example 3: Voice Type Validation")
print("VoiceType is a type alias for OpenAI voice literals.")
print("Valid OpenAI voice types include:")
for voice in VOICES[:5]:
    print(f"  - {voice}")
print("  ...")
print()

# Example 4: Sample Rate Constant
print("Example 4: Sample Rate Constant")
print(f"Default sample rate: {SAMPLE_RATE} Hz")
print(f"This is used for audio playback and processing.")
print()

# Example 5: Voice Selection Helper
print("Example 5: Voice Selection Helper Function")
def get_voice_info(provider: str = "openai"):
    """Get information about available voices."""
    if provider.lower() == "openai":
        return {
            "provider": "OpenAI",
            "voices": VOICES,
            "count": len(VOICES),
            "example": "alloy"
        }
    elif provider.lower() == "elevenlabs":
        return {
            "provider": "ElevenLabs",
            "voices": ELEVENLABS_VOICE_NAMES,
            "count": len(ELEVENLABS_VOICE_NAMES),
            "example": "rachel"
        }
    else:
        return None

openai_info = get_voice_info("openai")
elevenlabs_info = get_voice_info("elevenlabs")

print(f"{openai_info['provider']} Voices:")
print(f"  Count: {openai_info['count']}")
print(f"  Example: {openai_info['example']}")
print(f"  All voices: {', '.join(openai_info['voices'][:5])}...")
print()

print(f"{elevenlabs_info['provider']} Voices:")
print(f"  Count: {elevenlabs_info['count']}")
print(f"  Example: {elevenlabs_info['example']}")
print(f"  Sample voices: {', '.join(elevenlabs_info['voices'][:5])}...")
print()

# Example 6: Voice Comparison
print("Example 6: Voice Comparison")
print("Comparing voice options between providers:")
print()
print("OpenAI TTS:")
print("  - 10 pre-defined voices")
print("  - Consistent quality")
print("  - Fast generation")
print("  - Good for general use")
print()
print("ElevenLabs TTS:")
print("  - 30+ voices available")
print("  - More natural sounding")
print("  - Customizable (stability, similarity)")
print("  - Better for expressive content")
print()

# Example 7: Voice Selection by Use Case
print("Example 7: Voice Selection by Use Case")
use_cases = {
    "Professional/Business": {
        "openai": ["nova", "onyx"],
        "elevenlabs": ["rachel", "nicole", "grace"]
    },
    "Casual/Friendly": {
        "openai": ["alloy", "shimmer"],
        "elevenlabs": ["bella", "domi", "elli"]
    },
    "Deep/Masculine": {
        "openai": ["onyx", "echo"],
        "elevenlabs": ["antoni", "josh", "adam"]
    },
    "Soft/Feminine": {
        "openai": ["shimmer", "nova"],
        "elevenlabs": ["bella", "glinda", "mimi"]
    }
}

for use_case, voices in use_cases.items():
    print(f"{use_case}:")
    print(f"  OpenAI: {', '.join(voices['openai'])}")
    print(f"  ElevenLabs: {', '.join(voices['elevenlabs'])}")
    print()

# Example 8: Using Voice IDs Directly
print("Example 8: Using Voice IDs Directly")
print("For ElevenLabs, you can use either:")
print("  1. Friendly names: 'rachel', 'domi', etc.")
print("  2. Voice IDs directly: '21m00Tcm4TlvDq8ikWAM'")
print()
print("Example:")
print("  stream_tts_elevenlabs(['Hello'], voice_id='rachel')  # Friendly name")
print("  stream_tts_elevenlabs(['Hello'], voice_id='21m00Tcm4TlvDq8ikWAM')  # Direct ID")
print()
print("For OpenAI, use the voice names directly:")
print("  stream_tts(['Hello'], voice='alloy')")
print("  stream_tts(['Hello'], voice='nova')")

