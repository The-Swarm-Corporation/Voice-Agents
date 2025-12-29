import re
from typing import Generator, Iterable, List, Optional, Union

import httpx
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

from voice_agents.models_and_voices import (
    ELEVENLABS_TTS_MODELS,
    ELEVENLABS_VOICES,
    GROQ_ORPHEUS_ARABIC_VOICES,
    GROQ_ORPHEUS_ENGLISH_VOICES,
    GROQ_TTS_MODELS,
    OPENAI_TTS_MODELS,
    VOICES,
    VoiceType,
)

from voice_agents.utils import (
    format_text_for_speech,
    get_api_key,
    process_and_play_audio_buffer,
)

load_dotenv()



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
    api_key = get_api_key(
        "OPENAI_API_KEY",
        "https://platform.openai.com/api-keys",
    )

    # Check if model has provider prefix (common mistake)
    if "/" in model:
        raise ValueError(
            f"stream_tts_openai expects model name without provider prefix.\n"
            f"You provided: '{model}'\n"
            f"Expected: 'tts-1' or 'tts-1-hd'\n"
            f"To use provider/model format, use the unified stream_tts() function instead:\n"
            f"  stream_tts(text_chunks, model='{model}', voice='{voice}')"
        )

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

                # Process all buffered data at once
                process_and_play_audio_buffer(buffer, response_format)
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

                    # Process and play audio for this chunk immediately
                    process_and_play_audio_buffer(
                        buffer, response_format, warn_on_empty=True
                    )
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
                "description": groq_english_descriptions.get(
                    voice_name
                ),
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
                "description": groq_arabic_descriptions.get(
                    voice_name
                ),
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
        elif model_lower.startswith(
            "canopylabs/"
        ) or model_lower.startswith("whisper-"):
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
    api_key = get_api_key(
        "ELEVENLABS_API_KEY",
        "https://elevenlabs.io/app/settings/api-keys",
    )

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
    api_key = get_api_key(
        "GROQ_API_KEY",
        "https://console.groq.com/keys",
    )

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
                        import io
                        import wave

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
                                audio = audio[
                                    :, 0
                                ]  # Take first channel

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
                            import io
                            import wave

                            # Read WAV file from buffer
                            wav_io = io.BytesIO(bytes(buffer))
                            with wave.open(wav_io, "rb") as wav_file:
                                # Get audio parameters
                                frames = wav_file.getnframes()
                                sample_rate = wav_file.getframerate()
                                channels = wav_file.getnchannels()
                                sample_width = wav_file.getsampwidth()

                                # Read audio data
                                audio_bytes = wav_file.readframes(
                                    frames
                                )

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
                                    audio = audio.reshape(
                                        -1, channels
                                    )
                                    audio = audio[
                                        :, 0
                                    ]  # Take first channel

                                # Play audio
                                if len(audio) > 0:
                                    audio_float = (
                                        audio.astype(np.float32)
                                        / 32768.0
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
