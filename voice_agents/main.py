"""
- Add groq voice agents
- Add 11even labs voice agents
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
                        error_text = "No additional error details available"
                        try:
                            error_bytes = b""
                            for chunk in response.iter_bytes():
                                error_bytes += chunk
                            if error_bytes:
                                error_text = error_bytes.decode('utf-8', errors='ignore')
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
                    error_text = "No additional error details available"
                    try:
                        error_bytes = b""
                        for chunk in response.iter_bytes():
                            error_bytes += chunk
                        if error_bytes:
                            error_text = error_bytes.decode('utf-8', errors='ignore')
                    except Exception as e:
                        error_text = f"Could not read error response: {str(e)}"
                    
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
                    complete_buffer = bytes(buffer[:complete_samples_size])
                    audio = np.frombuffer(complete_buffer, dtype=np.int16)
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
                                for audio_chunk in response.iter_bytes():
                                    error_bytes += audio_chunk
                                if error_bytes:
                                    error_text = error_bytes.decode('utf-8', errors='ignore')
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
                        error_text = "No additional error details available"
                        try:
                            error_bytes = b""
                            for audio_chunk in response.iter_bytes():
                                error_bytes += audio_chunk
                            if error_bytes:
                                error_text = error_bytes.decode('utf-8', errors='ignore')
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
                        complete_buffer = bytes(buffer[:complete_samples_size])
                        audio = np.frombuffer(complete_buffer, dtype=np.int16)
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


def stream_tts(
    text_chunks: Union[List[str], Iterable[str]],
    model: str = "tts-1",
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
        model (str): The model name to use. Determines the provider:
            - OpenAI models: "tts-1", "tts-1-hd" (default: "tts-1")
            - ElevenLabs models: "eleven_multilingual_v2", "eleven_turbo_v2", etc.
        voice (Optional[str]): Voice identifier. For OpenAI, use voice names like "alloy", "nova", etc.
            For ElevenLabs, use friendly names like "rachel", "domi", etc. or voice IDs.
            If not provided, defaults to "alloy" for OpenAI or requires voice_id for ElevenLabs.
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
        enable_logging (bool): ElevenLabs-specific logging setting. Default is True. Ignored for OpenAI.
    
    Returns:
        Union[None, Generator[bytes, None, None]]: 
            - None if return_generator is False (plays audio)
            - Generator[bytes, None, None] if return_generator is True (yields audio chunks)
    
    Example:
        >>> # Using OpenAI
        >>> stream_tts(["Hello world"], model="tts-1", voice="alloy")
        >>> 
        >>> # Using ElevenLabs
        >>> stream_tts(["Hello world"], model="eleven_multilingual_v2", voice="rachel")
        >>> 
        >>> # Get generator for FastAPI
        >>> generator = stream_tts(
        ...     ["Hello world"], 
        ...     model="tts-1", 
        ...     voice="alloy", 
        ...     return_generator=True
        ... )
    """
    # Detect provider from model name
    model_lower = model.lower()
    
    # Check if it's an OpenAI model
    if model_lower.startswith("tts-1"):
        # Use OpenAI
        if voice is None:
            voice = "alloy"  # Default OpenAI voice
        
        # Set default response_format for OpenAI if not provided
        if response_format is None:
            response_format = "pcm"
        
        return stream_tts_openai(
            text_chunks=text_chunks,
            voice=voice,  # type: ignore
            model=model,
            stream_mode=stream_mode,
            response_format=response_format,
            return_generator=return_generator,
        )
    
    # Check if it's an ElevenLabs model
    elif model_lower.startswith("eleven_"):
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
            model_id=model,
            stability=stability,
            similarity_boost=similarity_boost,
            output_format=output_format,
            optimize_streaming_latency=optimize_streaming_latency,
            enable_logging=enable_logging,
            stream_mode=stream_mode,
            return_generator=return_generator,
        )
    
    else:
        # Unknown model, try to infer provider
        # Default to OpenAI for backward compatibility
        if voice is None:
            voice = "alloy"
        
        if response_format is None:
            response_format = "pcm"
        
        # Try OpenAI first (backward compatibility)
        return stream_tts_openai(
            text_chunks=text_chunks,
            voice=voice,  # type: ignore
            model=model,
            stream_mode=stream_mode,
            response_format=response_format,
            return_generator=return_generator,
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
    actual_voice_id = ELEVENLABS_VOICES.get(voice_id.lower(), voice_id)

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
    elif output_format.startswith("ulaw_") or output_format.startswith("alaw_"):
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
    def process_audio_buffer(buffer: bytearray, sample_rate: int) -> None:
        """Process audio buffer and play it."""
        if len(buffer) > 0:
            if output_format.startswith("pcm_"):
                # For PCM format, convert bytes to numpy array
                # PCM is 16-bit signed integers (2 bytes per sample)
                if len(buffer) >= 2:
                    complete_samples_size = (len(buffer) // 2) * 2
                    complete_buffer = bytes(buffer[:complete_samples_size])
                    audio = np.frombuffer(complete_buffer, dtype=np.int16)
                    
                    # Play audio with the appropriate sample rate
                    if len(audio) > 0:
                        audio_float = audio.astype(np.float32) / 32768.0
                        sd.play(audio_float, sample_rate)
                        sd.wait()
            elif output_format.startswith("ulaw_") or output_format.startswith("alaw_"):
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
                        audio_float = audio.astype(np.float32) / 32768.0
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
        params["optimize_streaming_latency"] = str(optimize_streaming_latency)

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
                    error_text = "No additional error details available"
                    try:
                        # Read the error response body
                        error_bytes = b""
                        for chunk in response.iter_bytes():
                            error_bytes += chunk
                        if error_bytes:
                            error_text = error_bytes.decode('utf-8', errors='ignore')
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
                        error_text = "No additional error details available"
                        try:
                            # Read the error response body
                            error_bytes = b""
                            for audio_chunk in response.iter_bytes():
                                error_bytes += audio_chunk
                            if error_bytes:
                                error_text = error_bytes.decode('utf-8', errors='ignore')
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
    elif output_format.startswith("ulaw_") or output_format.startswith("alaw_"):
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
                audio_float = audio_float[:, 0] if audio_float.shape[1] > 0 else audio_float
            
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
        "file": (os.path.basename(file_path), open(file_path, "rb"), "audio/wav")
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
                    error_text = f"Could not read error response: {str(e)}"
                
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
        if use_temp_file and temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass
        # Close the file handle
        if 'files' in locals() and files.get("file"):
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
        model: The TTS model to use. Default is "tts-1".
        min_sentence_length: Minimum length before sending a sentence to TTS. Default is 10.
    """
    
    def __init__(
        self,
        voice: str = "alloy",
        model: str = "tts-1",
        min_sentence_length: int = 10,
    ):
        self.voice = voice
        self.model = model
        self.min_sentence_length = min_sentence_length
        self.buffer = ""
        # Pattern to match sentence endings: . ! ? followed by whitespace or end of string
        self.sentence_endings = re.compile(r'[.!?](?:\s+|$)')
        
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
                if sentence and len(sentence) >= self.min_sentence_length:
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
            sentence_list = self.sentence_endings.split(text_to_process)
            for sentence in sentence_list:
                sentence = sentence.strip()
                if sentence and len(sentence) >= self.min_sentence_length:
                    sentences.append(sentence)
        
        return sentences
    
    def flush(self) -> None:
        """
        Flush any remaining text in the buffer to TTS.
        """
        if self.buffer.strip():
            try:
                formatted = format_text_for_speech(self.buffer.strip())
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

