"""
- Add groq voice agents
- Add 11even labs voice agents
- Groq voice agents
"""

import os
import re
from typing import Generator, Iterable, List, Literal, Optional, Union

import httpx
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

load_dotenv()

SAMPLE_RATE = 24000

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


def format_text_for_speech(text: str) -> List[str]:
    """
    Format a long string into a list of speech-friendly chunks by splitting on
    sentence boundaries and other natural speech pauses.

    Splits on:
    - Periods (.)
    - Exclamation marks (!)
    - Question marks (?)
    - Newlines (\n)
    - Semicolons (;)
    - Colons followed by space (: )

    Handles edge cases:
    - Abbreviations (e.g., "Dr.", "Mr.", "U.S.A.")
    - Decimal numbers (e.g., "3.14")
    - URLs and email addresses
    - Multiple consecutive punctuation marks

    Args:
        text: Long string of text to format

    Returns:
        List of formatted text chunks, stripped of whitespace and filtered
        to remove empty strings
    """
    if not text or not text.strip():
        return []

    # Common abbreviations that shouldn't split sentences
    abbreviations = [
        r"\bDr\.",
        r"\bMr\.",
        r"\bMrs\.",
        r"\bMs\.",
        r"\bProf\.",
        r"\bSr\.",
        r"\bJr\.",
        r"\bInc\.",
        r"\bLtd\.",
        r"\bCorp\.",
        r"\bvs\.",
        r"\betc\.",
        r"\be\.g\.",
        r"\bi\.e\.",
        r"\bU\.S\.A\.",
        r"\bU\.K\.",
        r"\bA\.I\.",
        r"\bPh\.D\.",
        r"\bM\.D\.",
        r"\bB\.A\.",
        r"\bM\.A\.",
        r"\bB\.S\.",
        r"\bM\.S\.",
    ]

    # Split on sentence boundaries, but be smart about it
    # Split on: . ! ? followed by space or end of string
    # Also split on: newlines, semicolons, colons (when followed by space)

    # First, protect abbreviations by temporarily replacing them
    protected_text = text
    abbrev_map = {}
    for i, abbrev in enumerate(abbreviations):
        placeholder = f"__ABBREV_{i}__"
        protected_text = re.sub(abbrev, placeholder, protected_text)
        abbrev_map[placeholder] = abbrev.replace("\\b", "").replace(
            "\\.", "."
        )

    # Split on sentence boundaries
    # Pattern: sentence ending (. ! ?) followed by whitespace or end of string
    # Also split on newlines, semicolons, and colons (when followed by space)
    split_pattern = (
        r"(?<=[.!?])\s+|(?<=[.!?])$|\n+|(?<=;)\s+|(?<=:\s)"
    )

    chunks = re.split(split_pattern, protected_text)

    # Restore abbreviations and clean up chunks
    result = []
    for chunk in chunks:
        if not chunk or not chunk.strip():
            continue

        # Restore abbreviations
        restored_chunk = chunk
        for placeholder, abbrev in abbrev_map.items():
            restored_chunk = restored_chunk.replace(
                placeholder, abbrev
            )

        # Strip whitespace and add to result if not empty
        cleaned = restored_chunk.strip()
        if cleaned:
            result.append(cleaned)

    # If no splits occurred, return the original text as a single chunk
    if not result:
        return [text.strip()] if text.strip() else []

    return result


def play_audio(audio_data: np.ndarray) -> None:
    """
    Play audio data using sounddevice.

    Args:
        audio_data: Audio data as numpy array of int16 samples
    """
    if len(audio_data) > 0:
        # Convert int16 to float32 and normalize to [-1, 1] range
        # int16 range is [-32768, 32767]
        audio_float = audio_data.astype(np.float32) / 32768.0
        sd.play(audio_float, SAMPLE_RATE)
        sd.wait()


def stream_tts_openai(
    text_chunks: Union[List[str], Iterable[str]],
    voice: VoiceType = "alloy",
    model: str = "tts-1",
    stream_mode: bool = False,
    response_format: str = "pcm",
    return_generator: bool = False,
) -> Union[None, Generator[bytes, None, None]]:
    """
    Stream text-to-speech using OpenAI TTS API, processing chunks and playing the resulting audio stream.

    Args:
        text_chunks (Union[List[str], Iterable[str]]): A list or iterable of text strings (already formatted/split)
            to convert to speech. If stream_mode is True, chunks are processed as they arrive.
        voice (VoiceType): Which voice to use for the TTS synthesis. Default is "alloy".
        model (str): The model to use for TTS. Default is "tts-1".
        stream_mode (bool): If True, process chunks as they arrive in real-time. If False, join all chunks
            and process as a single request. Default is False.
        response_format (str): Audio format to request from OpenAI. Options: "pcm", "mp3", "opus", "aac", "flac".
            Default is "pcm" (16-bit PCM at 24kHz). Note: When return_generator is False and format is not "pcm",
            audio will be streamed as bytes but may not play correctly.
        return_generator (bool): If True, returns a generator that yields audio chunks as bytes (for FastAPI streaming).
            If False, plays audio to system output. Default is False.

    Returns:
        Union[None, Generator[bytes, None, None]]:
            - None if return_generator is False (plays audio)
            - Generator[bytes, None, None] if return_generator is True (yields audio chunks)

    Details:
        - This function uses the OpenAI TTS API's streaming capabilities via httpx.
        - When stream_mode is False, all `text_chunks` are joined into a single string for synthesis.
        - When stream_mode is True, each chunk is processed individually as it arrives.
        - When return_generator is False, audio is streamed, buffered, and played using the `play_audio` helper.
        - When return_generator is True, audio chunks are yielded as bytes for use with FastAPI StreamingResponse.
        - Handles incomplete PCM audio samples by only processing complete 16-bit samples.
        - Useful for real-time output, agent system narration, or API streaming.

    Example:
        >>> # Play audio locally
        >>> stream_tts(["Hello world"], voice="alloy")
        >>>
        >>> # Get generator for FastAPI
        >>> from fastapi.responses import StreamingResponse
        >>> generator = stream_tts(["Hello world"], voice="alloy", return_generator=True)
        >>> return StreamingResponse(generator, media_type="audio/pcm")
    """
    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None or not api_key.strip():
        raise ValueError(
            "OpenAI API key not provided. Set OPENAI_API_KEY environment variable.\n"
            "You can get your API key from: https://platform.openai.com/api-keys"
        )

    # Strip any whitespace from the API key
    api_key = api_key.strip()

    # OpenAI TTS API endpoint
    url = "https://api.openai.com/v1/audio/speech"

    # Headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # If stream_mode is False, process all chunks at once (backward compatible)
    if not stream_mode:
        # Convert iterable to list if needed
        if isinstance(text_chunks, (list, tuple)):
            chunks_list = list(text_chunks)
        else:
            chunks_list = list(text_chunks)

        # Join all text chunks into a single string
        text = " ".join(chunks_list)

        # Payload
        payload = {
            "model": model,
            "voice": voice,
            "input": text,
            "response_format": response_format,
        }

        # If return_generator is True, yield chunks directly
        if return_generator:
            # Make streaming request to OpenAI TTS API
            try:
                with httpx.stream(
                    "POST",
                    url,
                    headers=headers,
                    json=payload,
                    timeout=30.0,
                ) as response:
                    # Check for authentication errors
                    if response.status_code == 401:
                        error_text = (
                            "No additional error details available"
                        )
                        try:
                            error_bytes = b""
                            for chunk in response.iter_bytes():
                                error_bytes += chunk
                            if error_bytes:
                                error_text = error_bytes.decode(
                                    "utf-8", errors="ignore"
                                )
                        except Exception as e:
                            error_text = f"Could not read error response: {str(e)}"

                        raise ValueError(
                            f"Authentication failed (401). Please check your OPENAI_API_KEY.\n"
                            f"The API key may be invalid, expired, or not set correctly.\n"
                            f"Error details: {error_text}\n"
                            f"Get your API key from: https://platform.openai.com/api-keys"
                        )

                    response.raise_for_status()

                    # Stream audio chunks and yield them
                    for audio_chunk in response.iter_bytes():
                        if audio_chunk:
                            yield audio_chunk
            except httpx.HTTPStatusError as e:
                # Re-raise ValueError if we already converted it
                if isinstance(e, ValueError):
                    raise
                # Otherwise, provide a generic error message
                raise ValueError(
                    f"HTTP error {e.response.status_code}: {e.response.text}\n"
                    f"URL: {e.request.url}"
                ) from e
            return

        # Buffer to handle incomplete chunks (int16 = 2 bytes per sample)
        buffer = bytearray()

        # Make streaming request to OpenAI TTS API
        try:
            with httpx.stream(
                "POST",
                url,
                headers=headers,
                json=payload,
                timeout=30.0,
            ) as response:
                # Check for authentication errors
                if response.status_code == 401:
                    error_text = (
                        "No additional error details available"
                    )
                    try:
                        error_bytes = b""
                        for chunk in response.iter_bytes():
                            error_bytes += chunk
                        if error_bytes:
                            error_text = error_bytes.decode(
                                "utf-8", errors="ignore"
                            )
                    except Exception as e:
                        error_text = (
                            f"Could not read error response: {str(e)}"
                        )

                    raise ValueError(
                        f"Authentication failed (401). Please check your OPENAI_API_KEY.\n"
                        f"The API key may be invalid, expired, or not set correctly.\n"
                        f"Error details: {error_text}\n"
                        f"Get your API key from: https://platform.openai.com/api-keys"
                    )

                response.raise_for_status()

                # Stream audio chunks
                for audio_chunk in response.iter_bytes():
                    if audio_chunk:
                        buffer.extend(audio_chunk)

                # Process all buffered data at once (only for PCM format)
                if response_format == "pcm" and len(buffer) >= 2:
                    # Ensure we have complete samples (multiples of 2 bytes)
                    complete_samples_size = (len(buffer) // 2) * 2
                    complete_buffer = bytes(
                        buffer[:complete_samples_size]
                    )
                    audio = np.frombuffer(
                        complete_buffer, dtype=np.int16
                    )
                    play_audio(audio)
                elif response_format != "pcm":
                    # For non-PCM formats, we can't play directly
                    # User should use return_generator=True for these formats
                    pass
        except httpx.HTTPStatusError as e:
            # Re-raise ValueError if we already converted it
            if isinstance(e, ValueError):
                raise
            # Otherwise, provide a generic error message
            raise ValueError(
                f"HTTP error {e.response.status_code}: {e.response.text}\n"
                f"URL: {e.request.url}"
            ) from e
    else:
        # Stream mode: process each chunk as it arrives
        for chunk in text_chunks:
            if not chunk or not chunk.strip():
                continue

            # Payload for this chunk
            payload = {
                "model": model,
                "voice": voice,
                "input": chunk.strip(),
                "response_format": response_format,
            }

            # If return_generator is True, yield chunks directly
            if return_generator:
                # Make streaming request to OpenAI TTS API for this chunk
                try:
                    with httpx.stream(
                        "POST",
                        url,
                        headers=headers,
                        json=payload,
                        timeout=30.0,
                    ) as response:
                        # Check for authentication errors
                        if response.status_code == 401:
                            error_text = "No additional error details available"
                            try:
                                error_bytes = b""
                                for (
                                    audio_chunk
                                ) in response.iter_bytes():
                                    error_bytes += audio_chunk
                                if error_bytes:
                                    error_text = error_bytes.decode(
                                        "utf-8", errors="ignore"
                                    )
                            except Exception as e:
                                error_text = f"Could not read error response: {str(e)}"

                            raise ValueError(
                                f"Authentication failed (401). Please check your OPENAI_API_KEY.\n"
                                f"The API key may be invalid, expired, or not set correctly.\n"
                                f"Error details: {error_text}\n"
                                f"Get your API key from: https://platform.openai.com/api-keys"
                            )

                        response.raise_for_status()

                        # Stream audio chunks for this text chunk and yield them
                        for audio_chunk in response.iter_bytes():
                            if audio_chunk:
                                yield audio_chunk
                except httpx.HTTPStatusError as e:
                    # Re-raise ValueError if we already converted it
                    if isinstance(e, ValueError):
                        raise
                    # Otherwise, provide a generic error message
                    raise ValueError(
                        f"HTTP error {e.response.status_code}: {e.response.text}\n"
                        f"URL: {e.request.url}"
                    ) from e
                continue

            # Buffer to handle incomplete chunks (int16 = 2 bytes per sample)
            buffer = bytearray()

            # Make streaming request to OpenAI TTS API for this chunk
            try:
                with httpx.stream(
                    "POST",
                    url,
                    headers=headers,
                    json=payload,
                    timeout=30.0,
                ) as response:
                    # Check for authentication errors
                    if response.status_code == 401:
                        error_text = (
                            "No additional error details available"
                        )
                        try:
                            error_bytes = b""
                            for audio_chunk in response.iter_bytes():
                                error_bytes += audio_chunk
                            if error_bytes:
                                error_text = error_bytes.decode(
                                    "utf-8", errors="ignore"
                                )
                        except Exception as e:
                            error_text = f"Could not read error response: {str(e)}"

                        raise ValueError(
                            f"Authentication failed (401). Please check your OPENAI_API_KEY.\n"
                            f"The API key may be invalid, expired, or not set correctly.\n"
                            f"Error details: {error_text}\n"
                            f"Get your API key from: https://platform.openai.com/api-keys"
                        )

                    response.raise_for_status()

                    # Stream audio chunks for this text chunk
                    for audio_chunk in response.iter_bytes():
                        if audio_chunk:
                            buffer.extend(audio_chunk)

                    # Process and play audio for this chunk immediately (only for PCM format)
                    if response_format == "pcm" and len(buffer) >= 2:
                        # Ensure we have complete samples (multiples of 2 bytes)
                        complete_samples_size = (len(buffer) // 2) * 2
                        complete_buffer = bytes(
                            buffer[:complete_samples_size]
                        )
                        audio = np.frombuffer(
                            complete_buffer, dtype=np.int16
                        )
                        play_audio(audio)
                    elif response_format != "pcm":
                        # For non-PCM formats, we can't play directly
                        # User should use return_generator=True for these formats
                        pass
            except httpx.HTTPStatusError as e:
                # Re-raise ValueError if we already converted it
                if isinstance(e, ValueError):
                    raise
                # Otherwise, provide a generic error message
                raise ValueError(
                    f"HTTP error {e.response.status_code}: {e.response.text}\n"
                    f"URL: {e.request.url}"
                ) from e


def list_models() -> List[dict[str, str]]:
    """
    List all available TTS models with their providers.

    Returns:
        List[dict[str, str]]: A list of dictionaries, each containing:
            - "model": The full model identifier (e.g., "openai/tts-1")
            - "provider": The provider name (e.g., "openai", "elevenlabs")
            - "model_name": The model name without provider prefix (e.g., "tts-1")

    Example:
        >>> models = list_models()
        >>> for model in models:
        ...     print(f"{model['model']} ({model['provider']})")
        openai/tts-1 (openai)
        openai/tts-1-hd (openai)
        elevenlabs/eleven_multilingual_v2 (elevenlabs)
        ...
    """
    models = []

    # Add OpenAI models
    for model_name in OPENAI_TTS_MODELS:
        models.append(
            {
                "model": f"openai/{model_name}",
                "provider": "openai",
                "model_name": model_name,
            }
        )

    # Add ElevenLabs models
    for model_name in ELEVENLABS_TTS_MODELS:
        models.append(
            {
                "model": f"elevenlabs/{model_name}",
                "provider": "elevenlabs",
                "model_name": model_name,
            }
        )

    # Add Groq TTS models
    for model_name in GROQ_TTS_MODELS:
        models.append(
            {
                "model": f"groq/{model_name}",
                "provider": "groq",
                "model_name": model_name,
            }
        )

    return models


def list_voices() -> List[dict[str, Union[str, None]]]:
    """
    List all available TTS voices with their providers.

    Returns:
        List[dict[str, Union[str, None]]]: A list of dictionaries, each containing:
            - "voice": The voice identifier (e.g., "alloy", "rachel")
            - "provider": The provider name (e.g., "openai", "elevenlabs")
            - "voice_id": The voice ID (for ElevenLabs) or None (for OpenAI)
            - "description": Optional description of the voice (for ElevenLabs)

    Example:
        >>> voices = list_voices()
        >>> for voice in voices:
        ...     print(f"{voice['voice']} ({voice['provider']})")
        alloy (openai)
        nova (openai)
        rachel (elevenlabs)
        ...
    """
    voices = []

    # Add OpenAI voices
    for voice_name in VOICES:
        voices.append(
            {
                "voice": voice_name,
                "provider": "openai",
                "voice_id": None,
                "description": None,
            }
        )

    # Add ElevenLabs voices
    # Extract descriptions from comments if available
    voice_descriptions = {
        "rachel": "Professional female voice",
        "domi": "Confident female voice",
        "bella": "Soft female voice",
        "antoni": "Deep male voice",
        "elli": "Expressive female voice",
        "josh": "Deep male voice",
        "arnold": "British male voice",
        "adam": "American male voice",
        "sam": "American male voice",
        "nicole": "Professional female voice",
        "glinda": "Warm female voice",
        "giovanni": "Italian male voice",
        "mimi": "Playful female voice",
        "freya": "British female voice",
        "shimmer": "Soft female voice",
        "grace": "Professional female voice",
        "daniel": "British male voice",
        "lily": "Young female voice",
        "dorothy": "Mature female voice",
        "charlie": "American male voice",
        "fin": "Irish male voice",
        "sarah": "Professional female voice",
        "michelle": "Warm female voice",
        "ryan": "American male voice",
        "paul": "British male voice",
        "drew": "American male voice",
        "clyde": "Deep male voice",
        "dave": "American male voice",
    }

    for voice_name, voice_id in ELEVENLABS_VOICES.items():
        voices.append(
            {
                "voice": voice_name,
                "provider": "elevenlabs",
                "voice_id": voice_id,
                "description": voice_descriptions.get(voice_name),
            }
        )

    # Add Groq Orpheus English voices
    groq_english_descriptions = {
        "austin": "Male English voice",
        "hannah": "Female English voice",
        "troy": "Male English voice",
    }
    for voice_name in GROQ_ORPHEUS_ENGLISH_VOICES:
        voices.append(
            {
                "voice": voice_name,
                "provider": "groq",
                "voice_id": None,
                "description": groq_english_descriptions.get(voice_name),
            }
        )

    # Add Groq Orpheus Arabic voices
    groq_arabic_descriptions = {
        "salma": "Female Arabic (Saudi) voice",
        "omar": "Male Arabic (Saudi) voice",
    }
    for voice_name in GROQ_ORPHEUS_ARABIC_VOICES:
        voices.append(
            {
                "voice": voice_name,
                "provider": "groq",
                "voice_id": None,
                "description": groq_arabic_descriptions.get(voice_name),
            }
        )

    return voices


def stream_tts(
    text_chunks: Union[List[str], Iterable[str]],
    model: str = "openai/tts-1",
    voice: Optional[str] = None,
    stream_mode: bool = False,
    return_generator: bool = False,
    # OpenAI-specific parameters
    response_format: Optional[str] = None,
    # ElevenLabs-specific parameters
    voice_id: Optional[str] = None,
    stability: float = 0.5,
    similarity_boost: float = 0.75,
    output_format: Optional[str] = None,
    optimize_streaming_latency: Optional[int] = None,
    enable_logging: bool = True,
) -> Union[None, Generator[bytes, None, None]]:
    """
    Unified text-to-speech streaming function that supports both OpenAI and ElevenLabs providers.

    This function automatically detects the provider based on the model name and routes to the
    appropriate backend, similar to how LiteLLM works.

    Args:
        text_chunks (Union[List[str], Iterable[str]]): A list or iterable of text strings to convert to speech.
        model (str): The model name to use in format "provider/model_name". Determines the provider:
            - OpenAI models: "openai/tts-1", "openai/tts-1-hd" (default: "openai/tts-1")
            - ElevenLabs models: "elevenlabs/eleven_multilingual_v2", "elevenlabs/eleven_turbo_v2", etc.
            - Groq models: "groq/canopylabs/orpheus-v1-english", "groq/canopylabs/orpheus-arabic-saudi"
            - For backward compatibility, also accepts "tts-1", "tts-1-hd", "eleven_multilingual_v2", etc.
        voice (Optional[str]): Voice identifier. For OpenAI, use voice names like "alloy", "nova", etc.
            For ElevenLabs, use friendly names like "rachel", "domi", etc. or voice IDs.
            For Groq English: "austin", "hannah", "troy". For Groq Arabic: "salma", "omar".
            If not provided, defaults to "alloy" for OpenAI or requires voice for Groq/ElevenLabs.
        stream_mode (bool): If True, process chunks as they arrive in real-time. Default is False.
        return_generator (bool): If True, returns a generator that yields audio chunks as bytes.
            If False, plays audio to system output. Default is False.
        response_format (Optional[str]): OpenAI-specific audio format. Options: "pcm", "mp3", "opus", "aac", "flac".
            Default is "pcm" for OpenAI. Ignored for ElevenLabs.
        voice_id (Optional[str]): ElevenLabs-specific voice ID. If provided, overrides voice parameter for ElevenLabs.
            Ignored for OpenAI.
        stability (float): ElevenLabs-specific stability setting (0.0 to 1.0). Default is 0.5. Ignored for OpenAI.
        similarity_boost (float): ElevenLabs-specific similarity boost (0.0 to 1.0). Default is 0.75. Ignored for OpenAI.
        output_format (Optional[str]): ElevenLabs-specific output format. Options include "pcm_44100", "mp3_44100_128", etc.
            Default is "pcm_44100" for ElevenLabs. Ignored for OpenAI.
        optimize_streaming_latency (Optional[int]): ElevenLabs-specific latency optimization (0-4). Ignored for OpenAI.
        enable_logging (bool): ElevenLabs-specific logging setting. Default is True. Ignored for ElevenLabs.

    Returns:
        Union[None, Generator[bytes, None, None]]:
            - None if return_generator is False (plays audio)
            - Generator[bytes, None, None] if return_generator is True (yields audio chunks)

    Example:
        >>> # Using OpenAI with new format
        >>> stream_tts(["Hello world"], model="openai/tts-1", voice="alloy")
        >>>
        >>> # Using ElevenLabs with new format
        >>> stream_tts(["Hello world"], model="elevenlabs/eleven_multilingual_v2", voice="rachel")
        >>>
        >>> # Using Groq with new format
        >>> stream_tts(["Hello world"], model="groq/canopylabs/orpheus-v1-english", voice="austin")
        >>>
        >>> # Backward compatible (old format still works)
        >>> stream_tts(["Hello world"], model="tts-1", voice="alloy")
        >>>
        >>> # Get generator for FastAPI
        >>> generator = stream_tts(
        ...     ["Hello world"],
        ...     model="openai/tts-1",
        ...     voice="alloy",
        ...     return_generator=True
        ... )
    """
    # Parse model name to extract provider and model
    provider = None
    model_name = model

    # Check if model is in provider/model_name format
    if "/" in model:
        parts = model.split("/", 1)
        if len(parts) == 2:
            provider = parts[0].lower()
            model_name = parts[1]

    # If no provider prefix, try to infer from model name (backward compatibility)
    if provider is None:
        model_lower = model_name.lower()

        # Check if it's an OpenAI model
        if model_lower.startswith("tts-1"):
            provider = "openai"
        # Check if it's an ElevenLabs model
        elif model_lower.startswith("eleven_"):
            provider = "elevenlabs"
        # Check if it's a Groq model
        elif model_lower.startswith("canopylabs/") or model_lower.startswith("whisper-"):
            provider = "groq"
        else:
            # Default to OpenAI for backward compatibility
            provider = "openai"

    # Route to appropriate provider
    if provider == "openai":
        # Use OpenAI
        if voice is None:
            voice = "alloy"  # Default OpenAI voice

        # Set default response_format for OpenAI if not provided
        if response_format is None:
            response_format = "pcm"

        return stream_tts_openai(
            text_chunks=text_chunks,
            voice=voice,  # type: ignore
            model=model_name,
            stream_mode=stream_mode,
            response_format=response_format,
            return_generator=return_generator,
        )

    elif provider == "elevenlabs":
        # Use ElevenLabs
        # Determine voice_id: use voice_id parameter if provided, otherwise use voice parameter
        if voice_id is None:
            if voice is None:
                raise ValueError(
                    "Either 'voice' or 'voice_id' must be provided for ElevenLabs models. "
                    "Use a friendly name like 'rachel' or a voice ID."
                )
            voice_id = voice
        else:
            # voice_id was explicitly provided, use it
            pass

        # Set default output_format for ElevenLabs if not provided
        if output_format is None:
            output_format = "pcm_44100"

        return stream_tts_elevenlabs(
            text_chunks=text_chunks,
            voice_id=voice_id,
            model_id=model_name,
            stability=stability,
            similarity_boost=similarity_boost,
            output_format=output_format,
            optimize_streaming_latency=optimize_streaming_latency,
            enable_logging=enable_logging,
            stream_mode=stream_mode,
            return_generator=return_generator,
        )

    elif provider == "groq":
        # Use Groq
        if voice is None:
            raise ValueError(
                "Voice must be provided for Groq models. "
                "For English model: 'austin', 'hannah', or 'troy'. "
                "For Arabic model: 'salma' or 'omar'."
            )

        # Set default response_format for Groq if not provided
        if response_format is None:
            response_format = "wav"

        return stream_tts_groq(
            text_chunks=text_chunks,
            voice=voice,
            model=model_name,
            stream_mode=stream_mode,
            response_format=response_format,
            return_generator=return_generator,
        )

    else:
        raise ValueError(
            f"Unknown provider: {provider}. Supported providers are 'openai', 'elevenlabs', and 'groq'. "
            f"Use format 'provider/model_name' (e.g., 'openai/tts-1', 'elevenlabs/eleven_multilingual_v2', or 'groq/canopylabs/orpheus-v1-english')."
        )


def stream_tts_elevenlabs(
    text_chunks: Union[List[str], Iterable[str]],
    voice_id: str,
    model_id: str = "eleven_multilingual_v2",
    stability: float = 0.5,
    similarity_boost: float = 0.75,
    output_format: str = "pcm_44100",
    optimize_streaming_latency: Optional[int] = None,
    enable_logging: bool = True,
    stream_mode: bool = False,
    return_generator: bool = False,
) -> Union[None, Generator[bytes, None, None]]:
    """
    Stream text-to-speech using Eleven Labs TTS API, processing chunks and playing the resulting audio stream.

    Args:
        text_chunks (Union[List[str], Iterable[str]]): A list or iterable of text strings (already formatted/split)
            to convert to speech. If stream_mode is True, chunks are processed as they arrive.
        voice_id (str): The Eleven Labs voice ID or friendly name (e.g., "rachel", "domi") to use for TTS synthesis.
        model_id (str): The model ID to use. Default is "eleven_multilingual_v2".
        stability (float): Stability setting for voice (0.0 to 1.0). Default is 0.5.
        similarity_boost (float): Similarity boost setting (0.0 to 1.0). Default is 0.75.
        output_format (str): Output audio format. Options: "mp3_22050_32", "mp3_24000_48", "mp3_44100_32",
            "mp3_44100_64", "mp3_44100_96", "mp3_44100_128", "mp3_44100_192", "pcm_8000", "pcm_16000",
            "pcm_22050", "pcm_24000", "pcm_32000", "pcm_44100", "pcm_48000", "ulaw_8000", "alaw_8000",
            "opus_48000_32", "opus_48000_64", "opus_48000_96", "opus_48000_128", "opus_48000_192".
            Default is "pcm_44100" for compatibility with play_audio. When return_generator is True,
            "mp3_44100_128" is recommended for web streaming.
        optimize_streaming_latency (Optional[int]): Latency optimization (0-4). Default is None.
        enable_logging (bool): Enable logging for the request. Default is True.
        stream_mode (bool): If True, process chunks as they arrive in real-time. If False, join all chunks
            and process as a single request. Default is False.
        return_generator (bool): If True, returns a generator that yields audio chunks as bytes (for FastAPI streaming).
            If False, plays audio to system output. Default is False.

    Returns:
        Union[None, Generator[bytes, None, None]]:
            - None if return_generator is False (plays audio)
            - Generator[bytes, None, None] if return_generator is True (yields audio chunks)

    Details:
        - This function uses the Eleven Labs TTS API streaming endpoint via httpx.
        - When stream_mode is False, all `text_chunks` are joined into a single string for synthesis.
        - When stream_mode is True, each chunk is processed individually as it arrives.
        - When return_generator is False, audio is streamed, buffered, and played using the `play_audio` helper.
        - When return_generator is True, audio chunks are yielded as bytes for use with FastAPI StreamingResponse.
        - For PCM formats, handles audio data as int16 samples.
        - For MP3/Opus formats, when return_generator is True, chunks are yielded directly without decoding.
        - Useful for real-time output, agent system narration, or API streaming.

    Example:
        >>> # Play audio locally
        >>> stream_tts_elevenlabs(["Hello world"], voice_id="rachel")
        >>>
        >>> # Get generator for FastAPI
        >>> from fastapi.responses import StreamingResponse
        >>> generator = stream_tts_elevenlabs(
        ...     ["Hello world"],
        ...     voice_id="rachel",
        ...     output_format="mp3_44100_128",
        ...     return_generator=True
        ... )
        >>> return StreamingResponse(generator, media_type="audio/mpeg")
    """
    # Get API key from parameter or environment variable
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if api_key is None or not api_key.strip():
        raise ValueError(
            "Eleven Labs API key not provided. Set ELEVENLABS_API_KEY environment variable.\n"
            "You can get your API key from: https://elevenlabs.io/app/settings/api-keys"
        )

    # Strip any whitespace from the API key
    api_key = api_key.strip()

    # Check if voice_id is a friendly name and look it up in ELEVENLABS_VOICES
    # If it's not found, assume it's already a voice ID
    actual_voice_id = ELEVENLABS_VOICES.get(
        voice_id.lower(), voice_id
    )

    # Determine sample rate from output format
    sample_rate_map = {
        "pcm_8000": 8000,
        "pcm_16000": 16000,
        "pcm_22050": 22050,
        "pcm_24000": 24000,
        "pcm_32000": 32000,
        "pcm_44100": 44100,
        "pcm_48000": 48000,
        "ulaw_8000": 8000,
        "alaw_8000": 8000,
        "mp3_22050_32": 22050,
        "mp3_24000_48": 24000,
        "mp3_44100_32": 44100,
        "mp3_44100_64": 44100,
        "mp3_44100_96": 44100,
        "mp3_44100_128": 44100,
        "mp3_44100_192": 44100,
        "opus_48000_32": 48000,
        "opus_48000_64": 48000,
        "opus_48000_96": 48000,
        "opus_48000_128": 48000,
        "opus_48000_192": 48000,
    }

    # Extract sample rate from format or use default
    if output_format.startswith("pcm_"):
        sample_rate = sample_rate_map.get(output_format, 44100)
    elif output_format.startswith(
        "ulaw_"
    ) or output_format.startswith("alaw_"):
        sample_rate = sample_rate_map.get(output_format, 8000)
    elif output_format.startswith("mp3_"):
        # For MP3 formats, we'd need to decode first (not implemented)
        raise ValueError(
            f"MP3 format '{output_format}' not yet supported. Please use PCM format (e.g., 'pcm_44100')."
        )
    elif output_format.startswith("opus_"):
        # For Opus formats, we'd need to decode first (not implemented)
        raise ValueError(
            f"Opus format '{output_format}' not yet supported. Please use PCM format (e.g., 'pcm_44100')."
        )
    else:
        sample_rate = 44100  # Default fallback

    # Helper function to process and play audio
    def process_audio_buffer(
        buffer: bytearray, sample_rate: int
    ) -> None:
        """Process audio buffer and play it."""
        if len(buffer) > 0:
            if output_format.startswith("pcm_"):
                # For PCM format, convert bytes to numpy array
                # PCM is 16-bit signed integers (2 bytes per sample)
                if len(buffer) >= 2:
                    complete_samples_size = (len(buffer) // 2) * 2
                    complete_buffer = bytes(
                        buffer[:complete_samples_size]
                    )
                    audio = np.frombuffer(
                        complete_buffer, dtype=np.int16
                    )

                    # Play audio with the appropriate sample rate
                    if len(audio) > 0:
                        audio_float = (
                            audio.astype(np.float32) / 32768.0
                        )
                        sd.play(audio_float, sample_rate)
                        sd.wait()
            elif output_format.startswith(
                "ulaw_"
            ) or output_format.startswith("alaw_"):
                # For μ-law and A-law formats, we need to decode them
                # These are 8-bit per sample formats
                try:
                    import audioop

                    if output_format.startswith("ulaw_"):
                        decoded = audioop.ulaw2lin(bytes(buffer), 2)
                    else:  # alaw
                        decoded = audioop.alaw2lin(bytes(buffer), 2)
                    audio = np.frombuffer(decoded, dtype=np.int16)
                    if len(audio) > 0:
                        audio_float = (
                            audio.astype(np.float32) / 32768.0
                        )
                        sd.play(audio_float, sample_rate)
                        sd.wait()
                except ImportError:
                    raise ValueError(
                        f"Format '{output_format}' requires the 'audioop' module for decoding. "
                        "Please use PCM format instead (e.g., 'pcm_44100')."
                    )
            else:
                raise ValueError(
                    f"Format '{output_format}' is not yet supported for playback. "
                    "Please use PCM format (e.g., 'pcm_44100')."
                )

    # Build URL with query parameters
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{actual_voice_id}/stream"

    # Build query parameters
    params = {
        "output_format": output_format,
        "enable_logging": str(enable_logging).lower(),
    }

    if optimize_streaming_latency is not None:
        params["optimize_streaming_latency"] = str(
            optimize_streaming_latency
        )

    # Headers matching the Eleven Labs API specification
    # Note: Accept header is optional for streaming endpoint, but can help with content negotiation
    headers = {
        "xi-api-key": api_key,  # Already stripped above
        "Content-Type": "application/json",
    }

    # Optionally add Accept header for better content negotiation
    # For streaming, the API will return the format specified in output_format query param
    if output_format.startswith("pcm_"):
        headers["Accept"] = "audio/pcm"
    elif output_format.startswith("mp3_"):
        headers["Accept"] = "audio/mpeg"
    elif output_format.startswith("opus_"):
        headers["Accept"] = "audio/opus"
    # For ulaw/alaw, we can omit Accept or use audio/basic, but it's optional

    # If stream_mode is False, process all chunks at once (backward compatible)
    if not stream_mode:
        # Convert iterable to list if needed
        if isinstance(text_chunks, (list, tuple)):
            chunks_list = list(text_chunks)
        else:
            chunks_list = list(text_chunks)

        # Join all text chunks into a single string
        text = " ".join(chunks_list)

        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
            },
        }

        # Make streaming request to Eleven Labs API
        try:
            with httpx.stream(
                "POST",
                url,
                headers=headers,
                params=params,
                json=payload,
                timeout=30.0,
            ) as response:
                # Check for authentication errors first (before reading response)
                if response.status_code == 401:
                    # Try to read error response for more details
                    error_text = (
                        "No additional error details available"
                    )
                    try:
                        # Read the error response body
                        error_bytes = b""
                        for chunk in response.iter_bytes():
                            error_bytes += chunk
                        if error_bytes:
                            error_text = error_bytes.decode(
                                "utf-8", errors="ignore"
                            )
                    except Exception as e:
                        error_text = (
                            f"Could not read error response: {str(e)}"
                        )

                    # Debug information
                    debug_info = (
                        f"Request URL: {url}\n"
                        f"Voice ID used: {actual_voice_id}\n"
                        f"Output format: {output_format}\n"
                        f"Model ID: {model_id}\n"
                        f"Headers sent: {dict((k, v if k != 'xi-api-key' else '***REDACTED***') for k, v in headers.items())}"
                    )

                    raise ValueError(
                        f"Authentication failed (401). Please check your ELEVENLABS_API_KEY.\n"
                        f"The API key may be invalid, expired, or not set correctly.\n"
                        f"Error details: {error_text}\n"
                        f"Debug info:\n{debug_info}\n"
                        f"Get your API key from: https://elevenlabs.io/app/settings/api-keys"
                    )
                elif response.status_code == 404:
                    raise ValueError(
                        f"Voice ID '{actual_voice_id}' not found. Please check if the voice ID is correct.\n"
                        f"If you used a friendly name like '{voice_id}', verify it exists in ELEVENLABS_VOICES."
                    )

                response.raise_for_status()

                # If return_generator is True, yield chunks directly
                if return_generator:
                    # Stream audio chunks and yield them
                    for chunk in response.iter_bytes():
                        if chunk:
                            yield chunk
                    return

                # Buffer to accumulate audio data
                buffer = bytearray()

                # Stream audio chunks
                for chunk in response.iter_bytes():
                    if chunk:
                        buffer.extend(chunk)

                # Process buffered audio data
                process_audio_buffer(buffer, sample_rate)
        except httpx.HTTPStatusError as e:
            # Re-raise ValueError if we already converted it
            if isinstance(e, ValueError):
                raise
            # Otherwise, provide a generic error message
            raise ValueError(
                f"HTTP error {e.response.status_code}: {e.response.text}\n"
                f"URL: {e.request.url}"
            ) from e
    else:
        # Stream mode: process each chunk as it arrives
        for chunk in text_chunks:
            if not chunk or not chunk.strip():
                continue

            payload = {
                "text": chunk.strip(),
                "model_id": model_id,
                "voice_settings": {
                    "stability": stability,
                    "similarity_boost": similarity_boost,
                },
            }

            # Make streaming request to Eleven Labs API for this chunk
            try:
                with httpx.stream(
                    "POST",
                    url,
                    headers=headers,
                    params=params,
                    json=payload,
                    timeout=30.0,
                ) as response:
                    # Check for authentication errors first (before reading response)
                    if response.status_code == 401:
                        # Try to read error response for more details
                        error_text = (
                            "No additional error details available"
                        )
                        try:
                            # Read the error response body
                            error_bytes = b""
                            for audio_chunk in response.iter_bytes():
                                error_bytes += audio_chunk
                            if error_bytes:
                                error_text = error_bytes.decode(
                                    "utf-8", errors="ignore"
                                )
                        except Exception as e:
                            error_text = f"Could not read error response: {str(e)}"

                        # Debug information
                        debug_info = (
                            f"Request URL: {url}\n"
                            f"Voice ID used: {actual_voice_id}\n"
                            f"Output format: {output_format}\n"
                            f"Model ID: {model_id}\n"
                            f"Headers sent: {dict((k, v if k != 'xi-api-key' else '***REDACTED***') for k, v in headers.items())}"
                        )

                        raise ValueError(
                            f"Authentication failed (401). Please check your ELEVENLABS_API_KEY.\n"
                            f"The API key may be invalid, expired, or not set correctly.\n"
                            f"Error details: {error_text}\n"
                            f"Debug info:\n{debug_info}\n"
                            f"Get your API key from: https://elevenlabs.io/app/settings/api-keys"
                        )
                    elif response.status_code == 404:
                        raise ValueError(
                            f"Voice ID '{actual_voice_id}' not found. Please check if the voice ID is correct.\n"
                            f"If you used a friendly name like '{voice_id}', verify it exists in ELEVENLABS_VOICES."
                        )

                    response.raise_for_status()

                    # If return_generator is True, yield chunks directly
                    if return_generator:
                        # Stream audio chunks for this text chunk and yield them
                        for audio_chunk in response.iter_bytes():
                            if audio_chunk:
                                yield audio_chunk
                        continue

                    # Buffer to accumulate audio data for this chunk
                    buffer = bytearray()

                    # Stream audio chunks for this text chunk
                    for audio_chunk in response.iter_bytes():
                        if audio_chunk:
                            buffer.extend(audio_chunk)

                    # Process and play audio for this chunk immediately
                    process_audio_buffer(buffer, sample_rate)
            except httpx.HTTPStatusError as e:
                # Re-raise ValueError if we already converted it
                if isinstance(e, ValueError):
                    raise
                # Otherwise, provide a generic error message
                raise ValueError(
                    f"HTTP error {e.response.status_code}: {e.response.text}\n"
                    f"URL: {e.request.url}"
                ) from e


def stream_tts_groq(
    text_chunks: Union[List[str], Iterable[str]],
    voice: str,
    model: str = "canopylabs/orpheus-v1-english",
    stream_mode: bool = False,
    response_format: str = "wav",
    return_generator: bool = False,
) -> Union[None, Generator[bytes, None, None]]:
    """
    Stream text-to-speech using Groq's fast TTS API, processing chunks and playing the resulting audio stream.

    Args:
        text_chunks (Union[List[str], Iterable[str]]): A list or iterable of text strings (already formatted/split)
            to convert to speech. If stream_mode is True, chunks are processed as they arrive.
        voice (str): The voice to use for TTS synthesis.
            For English model (canopylabs/orpheus-v1-english): "austin", "hannah", "troy"
            For Arabic model (canopylabs/orpheus-arabic-saudi): "salma", "omar"
        model (str): The model to use for TTS.
            Options: "canopylabs/orpheus-v1-english", "canopylabs/orpheus-arabic-saudi"
            Default is "canopylabs/orpheus-v1-english".
        stream_mode (bool): If True, process chunks as they arrive in real-time. If False, join all chunks
            and process as a single request. Default is False.
        response_format (str): Audio format to request from Groq. Options: "wav", "mp3", "opus", "aac", "flac".
            Default is "wav". Note: When return_generator is False and format is not "wav",
            audio will be streamed as bytes but may not play correctly.
        return_generator (bool): If True, returns a generator that yields audio chunks as bytes (for FastAPI streaming).
            If False, plays audio to system output. Default is False.

    Returns:
        Union[None, Generator[bytes, None, None]]:
            - None if return_generator is False (plays audio)
            - Generator[bytes, None, None] if return_generator is True (yields audio chunks)

    Details:
        - This function uses the Groq TTS API's streaming capabilities via httpx.
        - When stream_mode is False, all `text_chunks` are joined into a single string for synthesis.
        - When stream_mode is True, each chunk is processed individually as it arrives.
        - When return_generator is False, audio is streamed, buffered, and played using the `play_audio` helper.
        - When return_generator is True, audio chunks are yielded as bytes for use with FastAPI StreamingResponse.
        - Supports vocal directions in text (e.g., "[cheerful] Hello world").
        - Useful for real-time output, agent system narration, or API streaming.

    Example:
        >>> # Play audio locally
        >>> stream_tts_groq(["Hello world"], voice="austin")
        >>>
        >>> # With vocal directions
        >>> stream_tts_groq(
        ...     ["Welcome to Orpheus. [cheerful] This is an example."],
        ...     voice="hannah",
        ...     model="canopylabs/orpheus-v1-english"
        ... )
        >>>
        >>> # Get generator for FastAPI
        >>> from fastapi.responses import StreamingResponse
        >>> generator = stream_tts_groq(
        ...     ["Hello world"],
        ...     voice="austin",
        ...     return_generator=True
        ... )
        >>> return StreamingResponse(generator, media_type="audio/wav")
    """
    # Get API key from environment variable
    api_key = os.getenv("GROQ_API_KEY")
    if api_key is None or not api_key.strip():
        raise ValueError(
            "Groq API key not provided. Set GROQ_API_KEY environment variable.\n"
            "You can get your API key from: https://console.groq.com/keys"
        )

    # Strip any whitespace from the API key
    api_key = api_key.strip()

    # Validate model
    if model not in GROQ_TTS_MODELS:
        raise ValueError(
            f"Invalid model '{model}'. Supported models: {', '.join(GROQ_TTS_MODELS)}"
        )

    # Validate voice based on model
    if model == "canopylabs/orpheus-v1-english":
        if voice not in GROQ_ORPHEUS_ENGLISH_VOICES:
            raise ValueError(
                f"Invalid voice '{voice}' for English model. "
                f"Supported voices: {', '.join(GROQ_ORPHEUS_ENGLISH_VOICES)}"
            )
    elif model == "canopylabs/orpheus-arabic-saudi":
        if voice not in GROQ_ORPHEUS_ARABIC_VOICES:
            raise ValueError(
                f"Invalid voice '{voice}' for Arabic model. "
                f"Supported voices: {', '.join(GROQ_ORPHEUS_ARABIC_VOICES)}"
            )

    # Groq TTS API endpoint
    url = "https://api.groq.com/openai/v1/audio/speech"

    # Headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # If stream_mode is False, process all chunks at once (backward compatible)
    if not stream_mode:
        # Convert iterable to list if needed
        if isinstance(text_chunks, (list, tuple)):
            chunks_list = list(text_chunks)
        else:
            chunks_list = list(text_chunks)

        # Join all text chunks into a single string
        text = " ".join(chunks_list)

        # Payload
        payload = {
            "model": model,
            "voice": voice,
            "input": text,
            "response_format": response_format,
        }

        # If return_generator is True, yield chunks directly
        if return_generator:
            # Make streaming request to Groq TTS API
            try:
                with httpx.stream(
                    "POST",
                    url,
                    headers=headers,
                    json=payload,
                    timeout=30.0,
                ) as response:
                    # Check for authentication errors
                    if response.status_code == 401:
                        error_text = (
                            "No additional error details available"
                        )
                        try:
                            error_bytes = b""
                            for chunk in response.iter_bytes():
                                error_bytes += chunk
                            if error_bytes:
                                error_text = error_bytes.decode(
                                    "utf-8", errors="ignore"
                                )
                        except Exception as e:
                            error_text = f"Could not read error response: {str(e)}"

                        raise ValueError(
                            f"Authentication failed (401). Please check your GROQ_API_KEY.\n"
                            f"The API key may be invalid, expired, or not set correctly.\n"
                            f"Error details: {error_text}\n"
                            f"Get your API key from: https://console.groq.com/keys"
                        )

                    response.raise_for_status()

                    # Stream audio chunks and yield them
                    for audio_chunk in response.iter_bytes():
                        if audio_chunk:
                            yield audio_chunk
            except httpx.HTTPStatusError as e:
                # Re-raise ValueError if we already converted it
                if isinstance(e, ValueError):
                    raise
                # Otherwise, provide a generic error message
                raise ValueError(
                    f"HTTP error {e.response.status_code}: {e.response.text}\n"
                    f"URL: {e.request.url}"
                ) from e
            return

        # Buffer to accumulate audio data
        buffer = bytearray()

        # Make streaming request to Groq TTS API
        try:
            with httpx.stream(
                "POST",
                url,
                headers=headers,
                json=payload,
                timeout=30.0,
            ) as response:
                # Check for authentication errors
                if response.status_code == 401:
                    error_text = (
                        "No additional error details available"
                    )
                    try:
                        error_bytes = b""
                        for chunk in response.iter_bytes():
                            error_bytes += chunk
                        if error_bytes:
                            error_text = error_bytes.decode(
                                "utf-8", errors="ignore"
                            )
                    except Exception as e:
                        error_text = (
                            f"Could not read error response: {str(e)}"
                        )

                    raise ValueError(
                        f"Authentication failed (401). Please check your GROQ_API_KEY.\n"
                        f"The API key may be invalid, expired, or not set correctly.\n"
                        f"Error details: {error_text}\n"
                        f"Get your API key from: https://console.groq.com/keys"
                    )

                response.raise_for_status()

                # Stream audio chunks
                for audio_chunk in response.iter_bytes():
                    if audio_chunk:
                        buffer.extend(audio_chunk)

                # Process and play audio (for WAV format)
                if response_format == "wav" and len(buffer) > 0:
                    try:
                        import wave
                        import io

                        # Read WAV file from buffer
                        wav_io = io.BytesIO(bytes(buffer))
                        with wave.open(wav_io, "rb") as wav_file:
                            # Get audio parameters
                            frames = wav_file.getnframes()
                            sample_rate = wav_file.getframerate()
                            channels = wav_file.getnchannels()
                            sample_width = wav_file.getsampwidth()

                            # Read audio data
                            audio_bytes = wav_file.readframes(frames)

                            # Convert to numpy array
                            if sample_width == 2:  # 16-bit
                                audio = np.frombuffer(
                                    audio_bytes, dtype=np.int16
                                )
                            elif sample_width == 4:  # 32-bit
                                audio = np.frombuffer(
                                    audio_bytes, dtype=np.int32
                                )
                            else:
                                raise ValueError(
                                    f"Unsupported sample width: {sample_width}"
                                )

                            # Handle stereo audio (convert to mono)
                            if channels > 1:
                                audio = audio.reshape(-1, channels)
                                audio = audio[:, 0]  # Take first channel

                            # Play audio
                            if len(audio) > 0:
                                audio_float = (
                                    audio.astype(np.float32) / 32768.0
                                )
                                sd.play(audio_float, sample_rate)
                                sd.wait()
                    except ImportError:
                        # Fallback: try to play raw WAV data
                        # This is a simple approach that may not work for all WAV files
                        print(
                            "Warning: wave module not available. "
                            "Install it for proper WAV playback."
                        )
                elif response_format != "wav":
                    # For non-WAV formats, we can't play directly
                    # User should use return_generator=True for these formats
                    pass
        except httpx.HTTPStatusError as e:
            # Re-raise ValueError if we already converted it
            if isinstance(e, ValueError):
                raise
            # Otherwise, provide a generic error message
            raise ValueError(
                f"HTTP error {e.response.status_code}: {e.response.text}\n"
                f"URL: {e.request.url}"
            ) from e
    else:
        # Stream mode: process each chunk as it arrives
        for chunk in text_chunks:
            if not chunk or not chunk.strip():
                continue

            # Payload for this chunk
            payload = {
                "model": model,
                "voice": voice,
                "input": chunk.strip(),
                "response_format": response_format,
            }

            # If return_generator is True, yield chunks directly
            if return_generator:
                # Make streaming request to Groq TTS API for this chunk
                try:
                    with httpx.stream(
                        "POST",
                        url,
                        headers=headers,
                        json=payload,
                        timeout=30.0,
                    ) as response:
                        # Check for authentication errors
                        if response.status_code == 401:
                            error_text = "No additional error details available"
                            try:
                                error_bytes = b""
                                for (
                                    audio_chunk
                                ) in response.iter_bytes():
                                    error_bytes += audio_chunk
                                if error_bytes:
                                    error_text = error_bytes.decode(
                                        "utf-8", errors="ignore"
                                    )
                            except Exception as e:
                                error_text = f"Could not read error response: {str(e)}"

                            raise ValueError(
                                f"Authentication failed (401). Please check your GROQ_API_KEY.\n"
                                f"The API key may be invalid, expired, or not set correctly.\n"
                                f"Error details: {error_text}\n"
                                f"Get your API key from: https://console.groq.com/keys"
                            )

                        response.raise_for_status()

                        # Stream audio chunks for this text chunk and yield them
                        for audio_chunk in response.iter_bytes():
                            if audio_chunk:
                                yield audio_chunk
                except httpx.HTTPStatusError as e:
                    # Re-raise ValueError if we already converted it
                    if isinstance(e, ValueError):
                        raise
                    # Otherwise, provide a generic error message
                    raise ValueError(
                        f"HTTP error {e.response.status_code}: {e.response.text}\n"
                        f"URL: {e.request.url}"
                    ) from e
                continue

            # Buffer to accumulate audio data for this chunk
            buffer = bytearray()

            # Make streaming request to Groq TTS API for this chunk
            try:
                with httpx.stream(
                    "POST",
                    url,
                    headers=headers,
                    json=payload,
                    timeout=30.0,
                ) as response:
                    # Check for authentication errors
                    if response.status_code == 401:
                        error_text = (
                            "No additional error details available"
                        )
                        try:
                            error_bytes = b""
                            for audio_chunk in response.iter_bytes():
                                error_bytes += audio_chunk
                            if error_bytes:
                                error_text = error_bytes.decode(
                                    "utf-8", errors="ignore"
                                )
                        except Exception as e:
                            error_text = f"Could not read error response: {str(e)}"

                        raise ValueError(
                            f"Authentication failed (401). Please check your GROQ_API_KEY.\n"
                            f"The API key may be invalid, expired, or not set correctly.\n"
                            f"Error details: {error_text}\n"
                            f"Get your API key from: https://console.groq.com/keys"
                        )

                    response.raise_for_status()

                    # Stream audio chunks for this text chunk
                    for audio_chunk in response.iter_bytes():
                        if audio_chunk:
                            buffer.extend(audio_chunk)

                    # Process and play audio for this chunk immediately (for WAV format)
                    if response_format == "wav" and len(buffer) > 0:
                        try:
                            import wave
                            import io

                            # Read WAV file from buffer
                            wav_io = io.BytesIO(bytes(buffer))
                            with wave.open(wav_io, "rb") as wav_file:
                                # Get audio parameters
                                frames = wav_file.getnframes()
                                sample_rate = wav_file.getframerate()
                                channels = wav_file.getnchannels()
                                sample_width = wav_file.getsampwidth()

                                # Read audio data
                                audio_bytes = wav_file.readframes(frames)

                                # Convert to numpy array
                                if sample_width == 2:  # 16-bit
                                    audio = np.frombuffer(
                                        audio_bytes, dtype=np.int16
                                    )
                                elif sample_width == 4:  # 32-bit
                                    audio = np.frombuffer(
                                        audio_bytes, dtype=np.int32
                                    )
                                else:
                                    raise ValueError(
                                        f"Unsupported sample width: {sample_width}"
                                    )

                                # Handle stereo audio (convert to mono)
                                if channels > 1:
                                    audio = audio.reshape(-1, channels)
                                    audio = audio[:, 0]  # Take first channel

                                # Play audio
                                if len(audio) > 0:
                                    audio_float = (
                                        audio.astype(np.float32) / 32768.0
                                    )
                                    sd.play(audio_float, sample_rate)
                                    sd.wait()
                        except ImportError:
                            # Fallback: try to play raw WAV data
                            print(
                                "Warning: wave module not available. "
                                "Install it for proper WAV playback."
                            )
                    elif response_format != "wav":
                        # For non-WAV formats, we can't play directly
                        # User should use return_generator=True for these formats
                        pass
            except httpx.HTTPStatusError as e:
                # Re-raise ValueError if we already converted it
                if isinstance(e, ValueError):
                    raise
                # Otherwise, provide a generic error message
                raise ValueError(
                    f"HTTP error {e.response.status_code}: {e.response.text}\n"
                    f"URL: {e.request.url}"
                ) from e


def get_media_type_for_format(output_format: str) -> str:
    """
    Get the appropriate media type (MIME type) for a given audio format.

    This is useful for setting the Content-Type header in FastAPI StreamingResponse.

    Args:
        output_format (str): The audio format string (e.g., "mp3_44100_128", "pcm_44100", "opus_48000_64").

    Returns:
        str: The corresponding media type (e.g., "audio/mpeg", "audio/pcm", "audio/opus").

    Example:
        >>> media_type = get_media_type_for_format("mp3_44100_128")
        >>> # Returns: "audio/mpeg"
    """
    if output_format.startswith("mp3_"):
        return "audio/mpeg"
    elif output_format.startswith("pcm_"):
        return "audio/pcm"
    elif output_format.startswith("opus_"):
        return "audio/opus"
    elif output_format.startswith(
        "ulaw_"
    ) or output_format.startswith("alaw_"):
        return "audio/basic"
    elif output_format in ["aac", "flac"]:
        return f"audio/{output_format}"
    else:
        # Default fallback
        return "audio/pcm"


# # Example 2: Using format_text_for_speech with a long string
# long_text = """
# Welcome to Swarms! This audio is generated in real time.
# Agents speaking to agents. The future of AI is here.
# What do you think? This is amazing! Let's explore together.
# """

# formatted_chunks = format_text_for_speech(long_text)
# print("Formatted chunks:", formatted_chunks)

# stream_tts(formatted_chunks)


# # stream_tts_elevenlabs(formatted_chunks, voice_id="rachel")


def speech_to_text(
    audio_file_path: Optional[str] = None,
    audio_data: Optional[np.ndarray] = None,
    sample_rate: int = 16000,
    model: str = "whisper-1",
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    response_format: str = "text",
    temperature: float = 0.0,
) -> str:
    """
    Convert speech to text using OpenAI's Whisper API.

    This function can transcribe audio from either a file path or raw audio data.
    It supports both file-based and direct audio data transcription.

    Args:
        audio_file_path (Optional[str]): Path to an audio file to transcribe.
            Supported formats: mp3, mp4, mpeg, mpga, m4a, wav, webm.
            If provided, audio_data will be ignored.
        audio_data (Optional[np.ndarray]): Raw audio data as numpy array.
            Should be float32 in range [-1, 1] or int16.
            If provided without audio_file_path, will be saved to a temporary file.
        sample_rate (int): Sample rate of the audio data. Default is 16000.
            Only used when audio_data is provided.
        model (str): The model to use for transcription. Default is "whisper-1".
        language (Optional[str]): The language of the input audio in ISO-639-1 format.
            If None, the model will attempt to detect the language automatically.
        prompt (Optional[str]): An optional text to guide the model's style or continue
            a previous audio segment. The prompt should match the audio language.
        response_format (str): The format of the transcript output.
            Options: "json", "text", "srt", "verbose_json", "vtt". Default is "text".
        temperature (float): The sampling temperature, between 0 and 1.
            Higher values make the output more random. Default is 0.0.

    Returns:
        str: The transcribed text from the audio.

    Raises:
        ValueError: If neither audio_file_path nor audio_data is provided,
            or if OPENAI_API_KEY is not set.
        IOError: If there's an error reading the audio file.
        httpx.HTTPStatusError: If there's an HTTP error from the API.

    Example:
        >>> # From file
        >>> text = speech_to_text(audio_file_path="recording.wav")
        >>>
        >>> # From numpy array
        >>> import sounddevice as sd
        >>> recording = sd.rec(int(3 * 16000), samplerate=16000, channels=1)
        >>> sd.wait()
        >>> text = speech_to_text(audio_data=recording, sample_rate=16000)
    """
    import os
    import tempfile

    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None or not api_key.strip():
        raise ValueError(
            "OpenAI API key not provided. Set OPENAI_API_KEY environment variable.\n"
            "You can get your API key from: https://platform.openai.com/api-keys"
        )

    # Strip any whitespace from the API key
    api_key = api_key.strip()

    # OpenAI Whisper API endpoint
    url = "https://api.openai.com/v1/audio/transcriptions"

    # Headers
    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    # Determine which audio source to use
    use_temp_file = False
    temp_file_path = None

    if audio_file_path:
        # Use the provided file path
        if not os.path.exists(audio_file_path):
            raise IOError(f"Audio file not found: {audio_file_path}")
        file_path = audio_file_path
    elif audio_data is not None:
        # Save audio data to a temporary file
        try:
            import soundfile as sf

            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=".wav"
            )
            temp_file_path = temp_file.name
            temp_file.close()

            # Convert audio data to float32 if needed
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.float32:
                audio_float = audio_data
            else:
                audio_float = audio_data.astype(np.float32)

            # Ensure mono audio
            if len(audio_float.shape) > 1:
                audio_float = (
                    audio_float[:, 0]
                    if audio_float.shape[1] > 0
                    else audio_float
                )

            # Save to temporary file
            sf.write(temp_file_path, audio_float, sample_rate)
            file_path = temp_file_path
            use_temp_file = True
        except ImportError:
            raise ValueError(
                "soundfile library is required for audio_data input. "
                "Install it with: pip install soundfile"
            )
    else:
        raise ValueError(
            "Either audio_file_path or audio_data must be provided."
        )

    # Prepare form data
    files = {
        "file": (
            os.path.basename(file_path),
            open(file_path, "rb"),
            "audio/wav",
        )
    }

    data = {
        "model": model,
        "response_format": response_format,
        "temperature": str(temperature),
    }

    if language:
        data["language"] = language

    if prompt:
        data["prompt"] = prompt

    try:
        # Make request to OpenAI Whisper API
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                url,
                headers=headers,
                files=files,
                data=data,
            )

            # Check for authentication errors
            if response.status_code == 401:
                error_text = "No additional error details available"
                try:
                    if response.text:
                        error_text = response.text
                except Exception as e:
                    error_text = (
                        f"Could not read error response: {str(e)}"
                    )

                raise ValueError(
                    f"Authentication failed (401). Please check your OPENAI_API_KEY.\n"
                    f"The API key may be invalid, expired, or not set correctly.\n"
                    f"Error details: {error_text}\n"
                    f"Get your API key from: https://platform.openai.com/api-keys"
                )

            response.raise_for_status()

            # Parse response based on format
            if response_format == "text":
                return response.text.strip()
            elif response_format == "json":
                result = response.json()
                return result.get("text", "")
            elif response_format == "verbose_json":
                result = response.json()
                return result.get("text", "")
            elif response_format in ["srt", "vtt"]:
                return response.text
            else:
                return response.text.strip()
    except httpx.HTTPStatusError as e:
        # Re-raise ValueError if we already converted it
        if isinstance(e, ValueError):
            raise
        # Otherwise, provide a generic error message
        raise ValueError(
            f"HTTP error {e.response.status_code}: {e.response.text}\n"
            f"URL: {e.request.url}"
        ) from e
    finally:
        # Clean up temporary file if we created one
        if (
            use_temp_file
            and temp_file_path
            and os.path.exists(temp_file_path)
        ):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass
        # Close the file handle
        if "files" in locals() and files.get("file"):
            files["file"][1].close()


def speech_to_text_elevenlabs(
    audio_file_path: Optional[str] = None,
    audio_data: Optional[np.ndarray] = None,
    sample_rate: int = 16000,
    realtime: bool = False,
    # Non-real-time parameters
    model_id: str = "scribe_v1",
    language_code: Optional[str] = None,
    tag_audio_events: bool = True,
    num_speakers: Optional[int] = None,
    timestamps_granularity: Literal["none", "word", "character"] = "word",
    diarize: bool = False,
    diarization_threshold: Optional[float] = None,
    file_format: Literal["pcm_s16le_16", "other"] = "other",
    cloud_storage_url: Optional[str] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    use_multi_channel: bool = False,
    enable_logging: bool = True,
    # Real-time parameters (only used when realtime=True)
    audio_format: Literal[
        "pcm_8000",
        "pcm_16000",
        "pcm_22050",
        "pcm_24000",
        "pcm_44100",
        "pcm_48000",
        "ulaw_8000",
    ] = "pcm_16000",
    commit_strategy: Literal["manual", "vad"] = "manual",
    vad_silence_threshold_secs: float = 1.5,
    vad_threshold: float = 0.4,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 2500,
    include_timestamps: bool = False,
    include_language_detection: bool = False,
) -> Union[str, Generator[dict, None, None]]:
    """
    Convert speech to text using ElevenLabs Speech-to-Text API.

    This function supports both real-time (WebSocket) and non-real-time (file upload) modes.
    It can transcribe audio from either a file path or raw audio data.

    Args:
        audio_file_path (Optional[str]): Path to an audio or video file to transcribe.
            Supported formats: All major audio and video formats (mp3, mp4, wav, etc.).
            File size must be less than 3.0GB. If provided, audio_data will be ignored.
            Only used when realtime=False.
        audio_data (Optional[np.ndarray]): Raw audio data as numpy array.
            Should be float32 in range [-1, 1] or int16.
            If provided without audio_file_path, will be saved to a temporary file.
            Only used when realtime=False.
        sample_rate (int): Sample rate of the audio data. Default is 16000.
            Only used when audio_data is provided and realtime=False.
        realtime (bool): If True, use WebSocket for real-time streaming transcription.
            If False, use file upload API. Default is False.

        # Non-real-time parameters (used when realtime=False)
        model_id (str): The ID of the model to use for transcription.
            Options: 'scribe_v1', 'scribe_v1_experimental'. Default is 'scribe_v1'.
        language_code (Optional[str]): ISO-639-1 or ISO-639-3 language code.
            If None, language is detected automatically.
        tag_audio_events (bool): Whether to tag audio events like (laughter), (footsteps).
            Default is True.
        num_speakers (Optional[int]): Maximum number of speakers (max 32).
            If None, uses maximum supported by model.
        timestamps_granularity (Literal["none", "word", "character"]): Granularity of timestamps.
            Default is "word".
        diarize (bool): Whether to annotate which speaker is talking. Default is False.
        diarization_threshold (Optional[float]): Diarization threshold (0.0-1.0).
            Only used when diarize=True and num_speakers=None.
        file_format (Literal["pcm_s16le_16", "other"]): Format of input audio.
            'pcm_s16le_16' requires 16-bit PCM at 16kHz, mono, little-endian.
            Default is "other".
        cloud_storage_url (Optional[str]): HTTPS URL of file to transcribe.
            Exactly one of file or cloud_storage_url must be provided.
        temperature (Optional[float]): Controls randomness (0.0-2.0). Higher = more diverse.
            If None, uses model default (usually 0).
        seed (Optional[int]): Seed for deterministic sampling (0-2147483647).
        use_multi_channel (bool): Whether audio has multiple channels (max 5).
            Default is False.
        enable_logging (bool): Enable logging for the request. Default is True.

        # Real-time parameters (only used when realtime=True)
        audio_format (Literal[...]): Audio format for real-time streaming.
            Options: 'pcm_8000', 'pcm_16000', 'pcm_22050', 'pcm_24000',
            'pcm_44100', 'pcm_48000', 'ulaw_8000'. Default is 'pcm_16000'.
        commit_strategy (Literal["manual", "vad"]): Strategy for committing transcriptions.
            'manual' requires explicit commit, 'vad' uses voice activity detection.
            Default is "manual".
        vad_silence_threshold_secs (float): Silence threshold in seconds for VAD.
            Default is 1.5.
        vad_threshold (float): Threshold for voice activity detection (0.0-1.0).
            Default is 0.4.
        min_speech_duration_ms (int): Minimum speech duration in milliseconds.
            Default is 250.
        min_silence_duration_ms (int): Minimum silence duration in milliseconds.
            Default is 2500.
        include_timestamps (bool): Include word-level timestamps in committed transcript.
            Default is False.
        include_language_detection (bool): Include language detection in committed transcript.
            Default is False.

    Returns:
        Union[str, Generator[dict, None, None]]:
            - If realtime=False: Returns the transcribed text as a string.
            - If realtime=True: Returns a generator that yields transcription messages
              (partial_transcript, committed_transcript, committed_transcript_with_timestamps, etc.)

    Raises:
        ValueError: If neither audio_file_path nor audio_data is provided (when realtime=False),
            or if ELEVENLABS_API_KEY is not set.
        IOError: If there's an error reading the audio file.
        httpx.HTTPStatusError: If there's an HTTP error from the API.

    Example:
        >>> # Non-real-time: From file
        >>> text = speech_to_text_elevenlabs(audio_file_path="recording.wav")
        >>>
        >>> # Non-real-time: From numpy array
        >>> import sounddevice as sd
        >>> recording = sd.rec(int(3 * 16000), samplerate=16000, channels=1)
        >>> sd.wait()
        >>> text = speech_to_text_elevenlabs(audio_data=recording, sample_rate=16000)
        >>>
        >>> # Real-time: WebSocket streaming
        >>> for message in speech_to_text_elevenlabs(
        ...     audio_data=recording,
        ...     sample_rate=16000,
        ...     realtime=True
        ... ):
        ...     if message.get("message_type") == "committed_transcript":
        ...         print(message["text"])
    """
    import os
    import tempfile
    import base64
    import json

    # Get API key from environment variable
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if api_key is None or not api_key.strip():
        raise ValueError(
            "ElevenLabs API key not provided. Set ELEVENLABS_API_KEY environment variable.\n"
            "You can get your API key from: https://elevenlabs.io/app/settings/api-keys"
        )

    # Strip any whitespace from the API key
    api_key = api_key.strip()

    if realtime:
        # Real-time WebSocket mode
        try:
            try:
                from websockets.sync.client import connect
            except ImportError:
                # Fallback for older websockets versions
                from websockets import connect
        except ImportError:
            raise ValueError(
                "websockets library is required for real-time mode. "
                "Install it with: pip install websockets"
            )

        # Determine audio source for real-time
        if audio_file_path:
            # Load audio file
            try:
                import soundfile as sf
                audio_data, sample_rate = sf.read(audio_file_path)
                # Convert to mono if needed
                if len(audio_data.shape) > 1:
                    audio_data = audio_data[:, 0] if audio_data.shape[1] > 0 else audio_data
            except ImportError:
                raise ValueError(
                    "soundfile library is required for audio_file_path in real-time mode. "
                    "Install it with: pip install soundfile"
                )
        elif audio_data is None:
            raise ValueError(
                "Either audio_file_path or audio_data must be provided for real-time mode."
            )

        # Convert audio to appropriate format
        if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
            # Normalize to int16
            audio_int16 = (audio_data * 32767.0).astype(np.int16)
        elif audio_data.dtype == np.int16:
            audio_int16 = audio_data
        else:
            audio_int16 = audio_data.astype(np.int16)

        # Ensure mono
        if len(audio_int16.shape) > 1:
            audio_int16 = audio_int16[:, 0] if audio_int16.shape[1] > 0 else audio_int16

        # Extract sample rate from audio_format
        format_to_rate = {
            "pcm_8000": 8000,
            "pcm_16000": 16000,
            "pcm_22050": 22050,
            "pcm_24000": 24000,
            "pcm_44100": 44100,
            "pcm_48000": 48000,
            "ulaw_8000": 8000,
        }
        target_sample_rate = format_to_rate.get(audio_format, 16000)

        # Resample if needed
        if sample_rate != target_sample_rate:
            try:
                import scipy.signal
                num_samples = int(len(audio_int16) * target_sample_rate / sample_rate)
                audio_int16 = scipy.signal.resample(audio_int16, num_samples).astype(np.int16)
                sample_rate = target_sample_rate
            except ImportError:
                # Simple resampling without scipy (linear interpolation)
                num_samples = int(len(audio_int16) * target_sample_rate / sample_rate)
                indices = np.linspace(0, len(audio_int16) - 1, num_samples)
                audio_int16 = np.interp(indices, np.arange(len(audio_int16)), audio_int16).astype(np.int16)
                sample_rate = target_sample_rate

        # Build WebSocket URL with query parameters
        base_url = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
        query_params = {
            "model_id": model_id,
            "include_timestamps": str(include_timestamps).lower(),
            "include_language_detection": str(include_language_detection).lower(),
            "audio_format": audio_format,
            "commit_strategy": commit_strategy,
            "vad_silence_threshold_secs": str(vad_silence_threshold_secs),
            "vad_threshold": str(vad_threshold),
            "min_speech_duration_ms": str(min_speech_duration_ms),
            "min_silence_duration_ms": str(min_silence_duration_ms),
            "enable_logging": str(enable_logging).lower(),
        }
        if language_code:
            query_params["language_code"] = language_code

        query_string = "&".join([f"{k}={v}" for k, v in query_params.items()])
        ws_url = f"{base_url}?{query_string}"

        # Headers
        headers = {
            "xi-api-key": api_key,
        }

        def realtime_generator():
            """Generator for real-time transcription messages."""
            try:
                with connect(ws_url, additional_headers=headers) as websocket:
                    # Send audio in chunks
                    chunk_size = int(sample_rate * 0.1)  # 100ms chunks
                    first_chunk = True

                    for i in range(0, len(audio_int16), chunk_size):
                        chunk = audio_int16[i : i + chunk_size]
                        # Encode to base64
                        audio_bytes = chunk.tobytes()
                        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

                        # Prepare message
                        message = {
                            "message_type": "input_audio_chunk",
                            "audio_base_64": audio_base64,
                            "commit": commit_strategy == "vad",  # Auto-commit if VAD
                            "sample_rate": sample_rate,
                        }

                        if first_chunk:
                            # Can optionally send previous_text here
                            first_chunk = False

                        # Send audio chunk
                        websocket.send(json.dumps(message))

                        # Receive and yield messages
                        try:
                            while True:
                                # Set a short timeout to check for messages
                                message_str = websocket.recv(timeout=0.1)
                                message_data = json.loads(message_str)
                                yield message_data

                                # If we got a committed transcript, we can break
                                # (depending on strategy)
                                if message_data.get("message_type") in [
                                    "committed_transcript",
                                    "committed_transcript_with_timestamps",
                                ]:
                                    if commit_strategy == "manual":
                                        break
                        except (TimeoutError, OSError):
                            # No message available, continue sending audio
                            continue
                        except Exception as e:
                            # Connection closed or other error
                            if "ConnectionClosed" in str(type(e)) or "closed" in str(e).lower():
                                break
                            raise

                    # Send final commit if manual strategy
                    if commit_strategy == "manual":
                        final_message = {
                            "message_type": "input_audio_chunk",
                            "audio_base_64": "",
                            "commit": True,
                            "sample_rate": sample_rate,
                        }
                        websocket.send(json.dumps(final_message))

                        # Receive remaining messages
                        try:
                            while True:
                                message_str = websocket.recv(timeout=5.0)
                                message_data = json.loads(message_str)
                                yield message_data

                                if message_data.get("message_type") in [
                                    "committed_transcript",
                                    "committed_transcript_with_timestamps",
                                ]:
                                    break
                        except (TimeoutError, OSError, Exception):
                            # Connection closed or timeout
                            pass

            except Exception as e:
                error_message = {
                    "message_type": "error",
                    "error": str(e),
                }
                yield error_message

        return realtime_generator()

    else:
        # Non-real-time file upload mode
        url = "https://api.elevenlabs.io/v1/speech-to-text"

        # Headers
        headers = {
            "xi-api-key": api_key,
        }

        # Determine which audio source to use
        use_temp_file = False
        temp_file_path = None

        if cloud_storage_url:
            # Use cloud storage URL
            file_path = None
        elif audio_file_path:
            # Use the provided file path
            if not os.path.exists(audio_file_path):
                raise IOError(f"Audio file not found: {audio_file_path}")
            file_path = audio_file_path
        elif audio_data is not None:
            # Save audio data to a temporary file
            try:
                import soundfile as sf

                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".wav"
                )
                temp_file_path = temp_file.name
                temp_file.close()

                # Convert audio data to float32 if needed
                if audio_data.dtype == np.int16:
                    audio_float = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype == np.float32:
                    audio_float = audio_data
                else:
                    audio_float = audio_data.astype(np.float32)

                # Ensure mono audio
                if len(audio_float.shape) > 1:
                    audio_float = (
                        audio_float[:, 0]
                        if audio_float.shape[1] > 0
                        else audio_float
                    )

                # Save to temporary file
                sf.write(temp_file_path, audio_float, sample_rate)
                file_path = temp_file_path
                use_temp_file = True
            except ImportError:
                raise ValueError(
                    "soundfile library is required for audio_data input. "
                    "Install it with: pip install soundfile"
                )
        else:
            raise ValueError(
                "Either audio_file_path, audio_data, or cloud_storage_url must be provided."
            )

        # Prepare multipart form data
        files = None
        data = {
            "model_id": model_id,
            "tag_audio_events": str(tag_audio_events).lower(),
            "timestamps_granularity": timestamps_granularity,
            "diarize": str(diarize).lower(),
            "file_format": file_format,
            "use_multi_channel": str(use_multi_channel).lower(),
            "enable_logging": str(enable_logging).lower(),
        }

        if language_code:
            data["language_code"] = language_code
        if num_speakers is not None:
            data["num_speakers"] = str(num_speakers)
        if diarization_threshold is not None:
            data["diarization_threshold"] = str(diarization_threshold)
        if cloud_storage_url:
            data["cloud_storage_url"] = cloud_storage_url
        if temperature is not None:
            data["temperature"] = str(temperature)
        if seed is not None:
            data["seed"] = str(seed)

        if file_path:
            files = {
                "file": (
                    os.path.basename(file_path),
                    open(file_path, "rb"),
                    "application/octet-stream",
                )
            }

        try:
            # Make request to ElevenLabs API
            with httpx.Client(timeout=300.0) as client:  # Longer timeout for large files
                response = client.post(
                    url,
                    headers=headers,
                    files=files,
                    data=data,
                )

                # Check for authentication errors
                if response.status_code == 401:
                    error_text = "No additional error details available"
                    try:
                        if response.text:
                            error_text = response.text
                    except Exception as e:
                        error_text = f"Could not read error response: {str(e)}"

                    raise ValueError(
                        f"Authentication failed (401). Please check your ELEVENLABS_API_KEY.\n"
                        f"The API key may be invalid, expired, or not set correctly.\n"
                        f"Error details: {error_text}\n"
                        f"Get your API key from: https://elevenlabs.io/app/settings/api-keys"
                    )

                response.raise_for_status()

                # Parse response
                result = response.json()

                # Handle multi-channel response
                if "transcripts" in result:
                    # Multi-channel response
                    transcripts = result["transcripts"]
                    # Combine all transcripts
                    text_parts = [t.get("text", "") for t in transcripts]
                    return " ".join(text_parts)
                else:
                    # Single channel response
                    return result.get("text", "")

        except httpx.HTTPStatusError as e:
            # Re-raise ValueError if we already converted it
            if isinstance(e, ValueError):
                raise
            # Otherwise, provide a generic error message
            raise ValueError(
                f"HTTP error {e.response.status_code}: {e.response.text}\n"
                f"URL: {e.request.url}"
            ) from e
        finally:
            # Clean up temporary file if we created one
            if (
                use_temp_file
                and temp_file_path
                and os.path.exists(temp_file_path)
            ):
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass
            # Close the file handle
            if files and files.get("file"):
                files["file"][1].close()


def speech_to_text_groq(
    audio_file_path: Optional[str] = None,
    audio_data: Optional[np.ndarray] = None,
    sample_rate: int = 16000,
    model: str = "whisper-large-v3-turbo",
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    response_format: str = "text",
    temperature: float = 0.0,
    timestamp_granularities: Optional[List[Literal["word", "segment"]]] = None,
    translate: bool = False,
) -> str:
    """
    Convert speech to text using Groq's fast Whisper API.

    This function can transcribe or translate audio from either a file path or raw audio data.
    It supports both transcription (preserving original language) and translation (to English).

    Args:
        audio_file_path (Optional[str]): Path to an audio file to transcribe/translate.
            Supported formats: mp3, mp4, mpeg, mpga, m4a, wav, webm, flac, ogg.
            Max file size: 25 MB (free tier), 100MB (dev tier).
            If provided, audio_data will be ignored.
        audio_data (Optional[np.ndarray]): Raw audio data as numpy array.
            Should be float32 in range [-1, 1] or int16.
            If provided without audio_file_path, will be saved to a temporary file.
        sample_rate (int): Sample rate of the audio data. Default is 16000.
            Only used when audio_data is provided.
        model (str): The model to use for transcription/translation.
            Options: "whisper-large-v3-turbo" (fast, multilingual, no translation),
                    "whisper-large-v3" (high accuracy, multilingual, supports translation).
            Default is "whisper-large-v3-turbo".
        language (Optional[str]): The language of the input audio in ISO-639-1 format.
            If None, the model will attempt to detect the language automatically.
            For translations, only 'en' is supported.
        prompt (Optional[str]): An optional text to guide the model's style or continue
            a previous audio segment. Limited to 224 tokens.
        response_format (str): The format of the transcript output.
            Options: "json", "text", "verbose_json". Default is "text".
        temperature (float): The sampling temperature, between 0 and 1.
            Higher values make the output more random. Default is 0.0.
        timestamp_granularities (Optional[List[Literal["word", "segment"]]]): 
            Timestamp granularities to populate. Only used when response_format="verbose_json".
            Options: ["word"], ["segment"], or ["word", "segment"].
            Default is None (uses "segment").
        translate (bool): If True, translate audio to English instead of transcribing.
            Only supported by "whisper-large-v3" model. Default is False.

    Returns:
        str: The transcribed or translated text from the audio.

    Raises:
        ValueError: If neither audio_file_path nor audio_data is provided,
            or if GROQ_API_KEY is not set, or if translate=True with unsupported model.
        IOError: If there's an error reading the audio file.
        httpx.HTTPStatusError: If there's an HTTP error from the API.

    Example:
        >>> # Transcription from file
        >>> text = speech_to_text_groq(audio_file_path="recording.wav")
        >>>
        >>> # Translation from file
        >>> text = speech_to_text_groq(
        ...     audio_file_path="recording.wav",
        ...     model="whisper-large-v3",
        ...     translate=True
        ... )
        >>>
        >>> # From numpy array with timestamps
        >>> import sounddevice as sd
        >>> recording = sd.rec(int(3 * 16000), samplerate=16000, channels=1)
        >>> sd.wait()
        >>> text = speech_to_text_groq(
        ...     audio_data=recording,
        ...     sample_rate=16000,
        ...     response_format="verbose_json",
        ...     timestamp_granularities=["word", "segment"]
        ... )
    """
    import os
    import tempfile

    # Get API key from environment variable
    api_key = os.getenv("GROQ_API_KEY")
    if api_key is None or not api_key.strip():
        raise ValueError(
            "Groq API key not provided. Set GROQ_API_KEY environment variable.\n"
            "You can get your API key from: https://console.groq.com/keys"
        )

    # Strip any whitespace from the API key
    api_key = api_key.strip()

    # Validate model
    if model not in GROQ_STT_MODELS:
        raise ValueError(
            f"Invalid model '{model}'. Supported models: {', '.join(GROQ_STT_MODELS)}"
        )

    # Validate translate parameter
    if translate and model != "whisper-large-v3":
        raise ValueError(
            f"Translation is only supported with 'whisper-large-v3' model, not '{model}'."
        )

    # Choose endpoint based on translate flag
    if translate:
        url = "https://api.groq.com/openai/v1/audio/translations"
    else:
        url = "https://api.groq.com/openai/v1/audio/transcriptions"

    # Headers
    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    # Determine which audio source to use
    use_temp_file = False
    temp_file_path = None

    if audio_file_path:
        # Use the provided file path
        if not os.path.exists(audio_file_path):
            raise IOError(f"Audio file not found: {audio_file_path}")
        file_path = audio_file_path
    elif audio_data is not None:
        # Save audio data to a temporary file
        try:
            import soundfile as sf

            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=".wav"
            )
            temp_file_path = temp_file.name
            temp_file.close()

            # Convert audio data to float32 if needed
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.float32:
                audio_float = audio_data
            else:
                audio_float = audio_data.astype(np.float32)

            # Ensure mono audio
            if len(audio_float.shape) > 1:
                audio_float = (
                    audio_float[:, 0]
                    if audio_float.shape[1] > 0
                    else audio_float
                )

            # Save to temporary file
            sf.write(temp_file_path, audio_float, sample_rate)
            file_path = temp_file_path
            use_temp_file = True
        except ImportError:
            raise ValueError(
                "soundfile library is required for audio_data input. "
                "Install it with: pip install soundfile"
            )
    else:
        raise ValueError(
            "Either audio_file_path or audio_data must be provided."
        )

    # Prepare form data
    files = {
        "file": (
            os.path.basename(file_path),
            open(file_path, "rb"),
            "audio/wav",
        )
    }

    data = {
        "model": model,
        "response_format": response_format,
        "temperature": str(temperature),
    }

    if language:
        data["language"] = language

    if prompt:
        data["prompt"] = prompt

    # Add timestamp_granularities if provided and response_format is verbose_json
    if timestamp_granularities and response_format == "verbose_json":
        # Groq API (OpenAI-compatible) expects this as an array
        # Send as JSON-encoded string, which is commonly accepted by OpenAI-compatible APIs
        import json
        data["timestamp_granularities"] = json.dumps(timestamp_granularities)

    try:
        # Make request to Groq API
        with httpx.Client(timeout=300.0) as client:  # Longer timeout for large files
            response = client.post(
                url,
                headers=headers,
                files=files,
                data=data,
            )

            # Check for authentication errors
            if response.status_code == 401:
                error_text = "No additional error details available"
                try:
                    if response.text:
                        error_text = response.text
                except Exception as e:
                    error_text = (
                        f"Could not read error response: {str(e)}"
                    )

                raise ValueError(
                    f"Authentication failed (401). Please check your GROQ_API_KEY.\n"
                    f"The API key may be invalid, expired, or not set correctly.\n"
                    f"Error details: {error_text}\n"
                    f"Get your API key from: https://console.groq.com/keys"
                )

            response.raise_for_status()

            # Parse response based on format
            if response_format == "text":
                return response.text.strip()
            elif response_format == "json":
                result = response.json()
                return result.get("text", "")
            elif response_format == "verbose_json":
                result = response.json()
                # Return the full JSON as a string, or extract text if available
                if isinstance(result, dict) and "text" in result:
                    return result.get("text", "")
                else:
                    # Return the full JSON string representation
                    import json
                    return json.dumps(result, indent=2, default=str)
            else:
                return response.text.strip()
    except httpx.HTTPStatusError as e:
        # Re-raise ValueError if we already converted it
        if isinstance(e, ValueError):
            raise
        # Otherwise, provide a generic error message
        raise ValueError(
            f"HTTP error {e.response.status_code}: {e.response.text}\n"
            f"URL: {e.request.url}"
        ) from e
    finally:
        # Clean up temporary file if we created one
        if (
            use_temp_file
            and temp_file_path
            and os.path.exists(temp_file_path)
        ):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass
        # Close the file handle
        if "files" in locals() and files.get("file"):
            files["file"][1].close()


def record_audio(
    duration: float = 5.0,
    sample_rate: int = 16000,
    channels: int = 1,
) -> np.ndarray:
    """
    Record audio from the default microphone.

    Args:
        duration (float): Duration of recording in seconds. Default is 5.0.
        sample_rate (int): Sample rate for recording. Default is 16000.
        channels (int): Number of audio channels. Default is 1 (mono).

    Returns:
        np.ndarray: Recorded audio data as numpy array (int16 format).

    Example:
        >>> audio = record_audio(duration=3.0)
        >>> text = speech_to_text(audio_data=audio, sample_rate=16000)
    """
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype=np.int16,
    )
    sd.wait()
    print("Recording finished.")
    return recording


class StreamingTTSCallback:
    """
    A callback class that buffers streaming text and converts it to speech in real-time.

    This class accumulates text chunks from the agent's streaming output, detects
    complete sentences, and sends them to TTS as they become available.

    Args:
        voice: The voice to use for TTS. Default is "alloy".
        model: The TTS model to use in format "provider/model_name". Default is "openai/tts-1".
            Examples: "openai/tts-1", "openai/tts-1-hd", "elevenlabs/eleven_multilingual_v2"
        min_sentence_length: Minimum length before sending a sentence to TTS. Default is 10.
    """

    def __init__(
        self,
        voice: str = "alloy",
        model: str = "openai/tts-1",
        min_sentence_length: int = 10,
    ):
        self.voice = voice
        self.model = model
        self.min_sentence_length = min_sentence_length
        self.buffer = ""
        # Pattern to match sentence endings: . ! ? followed by whitespace or end of string
        self.sentence_endings = re.compile(r"[.!?](?:\s+|$)")

    def __call__(self, chunk: str) -> None:
        """
        Process a streaming text chunk.

        Args:
            chunk: The text chunk received from the agent's streaming output.
        """
        if not chunk:
            return

        # Add chunk to buffer
        self.buffer += chunk

        # Check for complete sentences
        sentences = self._extract_complete_sentences()

        # Send complete sentences to TTS
        if sentences:
            for sentence in sentences:
                sentence = sentence.strip()
                if (
                    sentence
                    and len(sentence) >= self.min_sentence_length
                ):
                    try:
                        # Format and stream the sentence
                        formatted = format_text_for_speech(sentence)
                        if formatted:
                            stream_tts(
                                formatted,
                                voice=self.voice,
                                model=self.model,
                                stream_mode=True,
                            )
                    except Exception as e:
                        print(f"Error in TTS streaming: {e}")

    def _extract_complete_sentences(self) -> List[str]:
        """
        Extract complete sentences from the buffer.

        Returns:
            List of complete sentences, removing them from the buffer.
        """
        sentences = []

        # Find all sentence endings
        matches = list(self.sentence_endings.finditer(self.buffer))

        if matches:
            # Extract sentences up to the last complete sentence
            last_end = matches[-1].end()
            text_to_process = self.buffer[:last_end]
            self.buffer = self.buffer[last_end:]

            # Split into sentences using the same pattern
            sentence_list = self.sentence_endings.split(
                text_to_process
            )
            for sentence in sentence_list:
                sentence = sentence.strip()
                if (
                    sentence
                    and len(sentence) >= self.min_sentence_length
                ):
                    sentences.append(sentence)

        return sentences

    def flush(self) -> None:
        """
        Flush any remaining text in the buffer to TTS.
        """
        if self.buffer.strip():
            try:
                formatted = format_text_for_speech(
                    self.buffer.strip()
                )
                if formatted:
                    stream_tts(
                        formatted,
                        voice=self.voice,
                        model=self.model,
                        stream_mode=True,
                    )
            except Exception as e:
                print(f"Error flushing TTS buffer: {e}")
            finally:
                self.buffer = ""
