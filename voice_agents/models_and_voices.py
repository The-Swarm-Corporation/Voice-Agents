from typing import List, Literal

# Available OpenAI TTS voices
VOICES: List[
    Literal[
        "alloy",
        "ash",
        "ballad",
        "coral",
        "echo",
        "fable",
        "nova",
        "onyx",
        "sage",
        "shimmer",
    ]
] = [
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
]

VoiceType = Literal[
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
]

# Eleven Labs voice IDs mapping (friendly names to voice IDs)
# Note: These are common pre-made voices. You can also use your own custom voice IDs.
ELEVENLABS_VOICES: dict[str, str] = {
    "rachel": "21m00Tcm4TlvDq8ikWAM",  # Professional female voice
    "domi": "AZnzlk1XvdvUeBnXmlld",  # Confident female voice
    "bella": "EXAVITQu4vr4xnSDxMaL",  # Soft female voice
    "antoni": "ErXwobaYiN019PkySvjV",  # Deep male voice
    "elli": "MF3mGyEYCl7XYWbV9V6O",  # Expressive female voice
    "josh": "TxGEqnHWrfWFTfGW9XjX",  # Deep male voice
    "arnold": "VR6AewLTigWG4xSOukaG",  # British male voice
    "adam": "pNInz6obpgDQGcFmaJgB",  # American male voice
    "sam": "yoZ06aMxZJJ28mfd3POQ",  # American male voice
    "nicole": "piTKgcLEGmPE4e6mEKli",  # Professional female voice
    "glinda": "z9fAnlkpzviPz146aGWa",  # Warm female voice
    "giovanni": "zcAOhNBS3c14rBihAFp1",  # Italian male voice
    "mimi": "zrHiDhphv9ZnVXBqCLjz",  # Playful female voice
    "freya": "jsCqWAovK2LkecY7zXl4",  # British female voice
    "shimmer": "onwK4e9ZLuTAKqWW03F9",  # Soft female voice
    "grace": "oWAxZDx7w5VEj9dCyTzz",  # Professional female voice
    "daniel": "onwK4e9ZLuTAKqWW03F9",  # British male voice
    "lily": "pFZP5JQG7iQjIQuC4Bku",  # Young female voice
    "dorothy": "ThT5KcBeYPX3keUQqHPh",  # Mature female voice
    "charlie": "IKne3meq5aSn9XLyUdCD",  # American male voice
    "fin": "xrExE9yKIg1WjnnlVkGX",  # Irish male voice
    "sarah": "EXAVITQu4vr4xnSDxMaL",  # Professional female voice
    "michelle": "flq6f7yk4E4fJM5XTYeZ",  # Warm female voice
    "ryan": "wViXBPUzp2ZZixB1xQuM",  # American male voice
    "paul": "5Q0t7uMcjvnagumLfvZi",  # British male voice
    "drew": "29vD33N1CtxCmqQRPOHJ",  # American male voice
    "clyde": "2EiwWnXFnvU5JabPnv8n",  # Deep male voice
    "dave": "CYw3kZ02Hs0563khs1Fj",  # American male voice
}

# List of available Eleven Labs voice names (for easy reference)
ELEVENLABS_VOICE_NAMES: List[str] = list(ELEVENLABS_VOICES.keys())

# Available TTS models by provider
OPENAI_TTS_MODELS: List[str] = [
    "tts-1",
    "tts-1-hd",
]

ELEVENLABS_TTS_MODELS: List[str] = [
    "eleven_multilingual_v2",
    "eleven_turbo_v2",
    "eleven_monolingual_v1",
]

# Groq TTS models
GROQ_TTS_MODELS: List[str] = [
    "canopylabs/orpheus-v1-english",
    "canopylabs/orpheus-arabic-saudi",
]

# Groq STT models
GROQ_STT_MODELS: List[str] = [
    "whisper-large-v3-turbo",
    "whisper-large-v3",
]

# Groq Orpheus English voices
GROQ_ORPHEUS_ENGLISH_VOICES: List[str] = [
    "austin",
    "hannah",
    "troy",
]

# Groq Orpheus Arabic voices
GROQ_ORPHEUS_ARABIC_VOICES: List[str] = [
    "salma",
    "omar",
]


