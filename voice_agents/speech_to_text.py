import os
from typing import Generator, List, Literal, Optional, Union

import httpx
import numpy as np

from voice_agents.utils import (
    get_api_key,
)

from voice_agents.models_and_voices import GROQ_STT_MODELS


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
    import tempfile

    # Get API key from environment variable
    api_key = get_api_key(
        "OPENAI_API_KEY",
        "https://platform.openai.com/api-keys",
    )

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
    timestamps_granularity: Literal[
        "none", "word", "character"
    ] = "word",
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
    import base64
    import json
    import tempfile

    # Get API key from environment variable
    api_key = get_api_key(
        "ELEVENLABS_API_KEY",
        "https://elevenlabs.io/app/settings/api-keys",
    )

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
                    audio_data = (
                        audio_data[:, 0]
                        if audio_data.shape[1] > 0
                        else audio_data
                    )
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
        if (
            audio_data.dtype == np.float32
            or audio_data.dtype == np.float64
        ):
            # Normalize to int16
            audio_int16 = (audio_data * 32767.0).astype(np.int16)
        elif audio_data.dtype == np.int16:
            audio_int16 = audio_data
        else:
            audio_int16 = audio_data.astype(np.int16)

        # Ensure mono
        if len(audio_int16.shape) > 1:
            audio_int16 = (
                audio_int16[:, 0]
                if audio_int16.shape[1] > 0
                else audio_int16
            )

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

                num_samples = int(
                    len(audio_int16)
                    * target_sample_rate
                    / sample_rate
                )
                audio_int16 = scipy.signal.resample(
                    audio_int16, num_samples
                ).astype(np.int16)
                sample_rate = target_sample_rate
            except ImportError:
                # Simple resampling without scipy (linear interpolation)
                num_samples = int(
                    len(audio_int16)
                    * target_sample_rate
                    / sample_rate
                )
                indices = np.linspace(
                    0, len(audio_int16) - 1, num_samples
                )
                audio_int16 = np.interp(
                    indices, np.arange(len(audio_int16)), audio_int16
                ).astype(np.int16)
                sample_rate = target_sample_rate

        # Build WebSocket URL with query parameters
        base_url = (
            "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
        )
        query_params = {
            "model_id": model_id,
            "include_timestamps": str(include_timestamps).lower(),
            "include_language_detection": str(
                include_language_detection
            ).lower(),
            "audio_format": audio_format,
            "commit_strategy": commit_strategy,
            "vad_silence_threshold_secs": str(
                vad_silence_threshold_secs
            ),
            "vad_threshold": str(vad_threshold),
            "min_speech_duration_ms": str(min_speech_duration_ms),
            "min_silence_duration_ms": str(min_silence_duration_ms),
            "enable_logging": str(enable_logging).lower(),
        }
        if language_code:
            query_params["language_code"] = language_code

        query_string = "&".join(
            [f"{k}={v}" for k, v in query_params.items()]
        )
        ws_url = f"{base_url}?{query_string}"

        # Headers
        headers = {
            "xi-api-key": api_key,
        }

        def realtime_generator():
            """Generator for real-time transcription messages."""
            try:
                with connect(
                    ws_url, additional_headers=headers
                ) as websocket:
                    # Send audio in chunks
                    chunk_size = int(
                        sample_rate * 0.1
                    )  # 100ms chunks
                    first_chunk = True

                    for i in range(0, len(audio_int16), chunk_size):
                        chunk = audio_int16[i : i + chunk_size]
                        # Encode to base64
                        audio_bytes = chunk.tobytes()
                        audio_base64 = base64.b64encode(
                            audio_bytes
                        ).decode("utf-8")

                        # Prepare message
                        message = {
                            "message_type": "input_audio_chunk",
                            "audio_base_64": audio_base64,
                            "commit": commit_strategy
                            == "vad",  # Auto-commit if VAD
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
                                message_str = websocket.recv(
                                    timeout=0.1
                                )
                                message_data = json.loads(message_str)
                                yield message_data

                                # If we got a committed transcript, we can break
                                # (depending on strategy)
                                if message_data.get(
                                    "message_type"
                                ) in [
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
                            if (
                                "ConnectionClosed" in str(type(e))
                                or "closed" in str(e).lower()
                            ):
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
                                message_str = websocket.recv(
                                    timeout=5.0
                                )
                                message_data = json.loads(message_str)
                                yield message_data

                                if message_data.get(
                                    "message_type"
                                ) in [
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
                raise IOError(
                    f"Audio file not found: {audio_file_path}"
                )
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
                    audio_float = (
                        audio_data.astype(np.float32) / 32768.0
                    )
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
            with httpx.Client(
                timeout=300.0
            ) as client:  # Longer timeout for large files
                response = client.post(
                    url,
                    headers=headers,
                    files=files,
                    data=data,
                )

                # Check for authentication errors
                if response.status_code == 401:
                    error_text = (
                        "No additional error details available"
                    )
                    try:
                        if response.text:
                            error_text = response.text
                    except Exception as e:
                        error_text = (
                            f"Could not read error response: {str(e)}"
                        )

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
                    text_parts = [
                        t.get("text", "") for t in transcripts
                    ]
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
    timestamp_granularities: Optional[
        List[Literal["word", "segment"]]
    ] = None,
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
    import tempfile

    # Get API key from environment variable
    api_key = get_api_key(
        "GROQ_API_KEY",
        "https://console.groq.com/keys",
    )

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

        data["timestamp_granularities"] = json.dumps(
            timestamp_granularities
        )

    try:
        # Make request to Groq API
        with httpx.Client(
            timeout=300.0
        ) as client:  # Longer timeout for large files
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

