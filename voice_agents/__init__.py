"""
Voice Agents - A comprehensive library for text-to-speech and speech-to-text operations.

This package provides unified interfaces for multiple TTS and STT providers including
OpenAI, ElevenLabs, and Groq.
"""

# Import constants and types from models_and_voices
from voice_agents.models_and_voices import (
    # Constants
    ELEVENLABS_TTS_MODELS,
    ELEVENLABS_VOICES,
    ELEVENLABS_VOICE_NAMES,
    GROQ_ORPHEUS_ARABIC_VOICES,
    GROQ_ORPHEUS_ENGLISH_VOICES,
    GROQ_STT_MODELS,
    GROQ_TTS_MODELS,
    OPENAI_TTS_MODELS,
    VOICES,
    # Type aliases
    VoiceType,
)

# Import utilities from utils
from voice_agents.utils import (
    # Constants
    SAMPLE_RATE,
    # Functions
    format_text_for_speech,
    get_media_type_for_format,
    play_audio,
    record_audio,
)

# Import TTS functions and classes from main
from voice_agents.main import (
    # Functions
    list_models,
    list_voices,
    stream_tts,
    stream_tts_elevenlabs,
    stream_tts_groq,
    stream_tts_openai,
    # Classes
    StreamingTTSCallback,
)

# Import STT functions from speech_to_text
from voice_agents.speech_to_text import (
    speech_to_text,
    speech_to_text_elevenlabs,
    speech_to_text_groq,
)

__all__ = [
    # Constants from models_and_voices
    "ELEVENLABS_TTS_MODELS",
    "ELEVENLABS_VOICES",
    "ELEVENLABS_VOICE_NAMES",
    "GROQ_ORPHEUS_ARABIC_VOICES",
    "GROQ_ORPHEUS_ENGLISH_VOICES",
    "GROQ_STT_MODELS",
    "GROQ_TTS_MODELS",
    "OPENAI_TTS_MODELS",
    "VOICES",
    # Type aliases
    "VoiceType",
    # Constants from utils
    "SAMPLE_RATE",
    # Functions from utils
    "format_text_for_speech",
    "get_media_type_for_format",
    "play_audio",
    "record_audio",
    # Functions from main (TTS)
    "list_models",
    "list_voices",
    "stream_tts",
    "stream_tts_elevenlabs",
    "stream_tts_groq",
    "stream_tts_openai",
    # Classes from main
    "StreamingTTSCallback",
    # Functions from speech_to_text (STT)
    "speech_to_text",
    "speech_to_text_elevenlabs",
    "speech_to_text_groq",
]

