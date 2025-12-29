import os
import re
from typing import List

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 24000


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
    if len(audio_data) == 0:
        print("Warning: Cannot play empty audio data.")
        return
    
    try:
        # Convert int16 to float32 and normalize to [-1, 1] range
        # int16 range is [-32768, 32767]
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # Check for valid audio data
        if np.all(np.isnan(audio_float)) or np.all(audio_float == 0):
            print("Warning: Audio data appears to be silent or invalid.")
            return
        
        sd.play(audio_float, SAMPLE_RATE)
        sd.wait()
    except Exception as e:
        raise ValueError(
            f"Error playing audio: {e}\n"
            f"Audio data shape: {audio_data.shape}, dtype: {audio_data.dtype}"
        ) from e


def get_api_key(env_var_name: str, api_key_url: str) -> str:
    """
    Get and validate API key from environment variable.

    Args:
        env_var_name: Name of the environment variable (e.g., "OPENAI_API_KEY")
        api_key_url: URL where users can get their API key (for error messages)

    Returns:
        str: The API key, stripped of whitespace

    Raises:
        ValueError: If the API key is not set or is empty
    """
    api_key = os.getenv(env_var_name)
    if api_key is None or not api_key.strip():
        raise ValueError(
            f"{env_var_name} not provided. Set {env_var_name} environment variable.\n"
            f"You can get your API key from: {api_key_url}"
        )

    # Strip any whitespace from the API key
    return api_key.strip()


def process_and_play_audio_buffer(
    buffer: bytearray,
    response_format: str,
    warn_on_empty: bool = False,
) -> None:
    """
    Process and play audio buffer based on the response format.

    Supports PCM format (direct playback) and compressed formats (mp3, opus, aac, flac)
    that require pydub for decoding.

    Args:
        buffer: Audio data buffer as bytearray
        response_format: Audio format string (e.g., "pcm", "mp3", "opus", "aac", "flac")
        warn_on_empty: If True, print warning and return instead of raising error when buffer is empty

    Raises:
        ValueError: If format is unsupported, dependencies are missing, or decoding fails
    """
    # Check if buffer is empty
    if len(buffer) == 0:
        if warn_on_empty:
            print(
                f"Warning: No audio data received for {response_format.upper()} format. Skipping playback."
            )
            return
        else:
            raise ValueError(
                f"No audio data received for {response_format.upper()} format."
            )
    
    # Handle PCM format
    if response_format == "pcm":
        if len(buffer) < 2:
            if warn_on_empty:
                print(
                    f"Warning: Buffer too small ({len(buffer)} bytes) for PCM format. Need at least 2 bytes."
                )
                return
            else:
                raise ValueError(
                    f"Buffer too small ({len(buffer)} bytes) for PCM format. Need at least 2 bytes."
                )
        
        # Ensure we have complete samples (multiples of 2 bytes)
        complete_samples_size = (len(buffer) // 2) * 2
        complete_buffer = bytes(buffer[:complete_samples_size])
        
        if len(complete_buffer) == 0:
            if warn_on_empty:
                print("Warning: No complete audio samples in buffer. Skipping playback.")
                return
            else:
                raise ValueError("No complete audio samples in buffer.")
        
        try:
            audio = np.frombuffer(complete_buffer, dtype=np.int16)
            if len(audio) == 0:
                if warn_on_empty:
                    print("Warning: Audio array is empty. Skipping playback.")
                    return
                else:
                    raise ValueError("Audio array is empty.")
            
            play_audio(audio)
        except Exception as e:
            raise ValueError(
                f"Error processing PCM audio: {e}\n"
                f"Buffer size: {len(buffer)} bytes, Complete samples: {len(complete_buffer)} bytes"
            ) from e
        return

    # Handle compressed audio formats
    if response_format in ["mp3", "opus", "aac", "flac"]:
        # Check if buffer has data
        if len(buffer) == 0:
            if warn_on_empty:
                print(
                    f"Warning: No audio data received for {response_format.upper()} format. Skipping playback."
                )
                return
            else:
                raise ValueError(
                    f"No audio data received for {response_format.upper()} format."
                )

        # Decode compressed audio formats
        try:
            import io
            from pydub import AudioSegment
            from pydub.playback import play

            # Create a BytesIO object from the buffer
            audio_bytes = bytes(buffer)

            # Check minimum size for MP3 (MP3 files typically have headers)
            if response_format == "mp3" and len(audio_bytes) < 100:
                print(
                    f"Warning: MP3 buffer too small ({len(audio_bytes)} bytes). May not be a valid MP3 file."
                )

            audio_io = io.BytesIO(audio_bytes)

            # Load audio based on format
            if response_format == "mp3":
                audio_segment = AudioSegment.from_mp3(audio_io)
            elif response_format == "opus":
                audio_segment = AudioSegment.from_ogg(audio_io)
            elif response_format == "aac":
                audio_segment = AudioSegment.from_file(audio_io, format="aac")
            elif response_format == "flac":
                audio_segment = AudioSegment.from_file(audio_io, format="flac")
            else:
                raise ValueError(f"Unsupported format: {response_format}")

            # Play the audio
            if len(audio_segment) > 0:
                play(audio_segment)
            else:
                print(
                    f"Warning: Decoded {response_format.upper()} audio is empty. Skipping playback."
                )
        except ImportError:
            raise ValueError(
                f"To play {response_format.upper()} format, install pydub and simpleaudio:\n"
                "  pip install pydub simpleaudio\n"
                "Or use return_generator=True to get raw audio bytes, or use response_format='pcm'."
            )
        except Exception as e:
            error_msg = str(e)
            # Provide more helpful error messages
            if (
                "not a valid" in error_msg.lower()
                or "could not decode" in error_msg.lower()
            ):
                raise ValueError(
                    f"Error decoding {response_format.upper()} audio: {error_msg}\n"
                    f"Buffer size: {len(buffer)} bytes\n"
                    "This may happen if:\n"
                    "  1. The audio stream was incomplete\n"
                    "  2. The format is not supported\n"
                    "  3. Try using response_format='pcm' instead (works without dependencies)\n"
                    "  4. Or use stream_mode=False to get complete audio file"
                )
            else:
                raise ValueError(
                    f"Error playing {response_format.upper()} audio: {error_msg}\n"
                    "Try using response_format='pcm' or return_generator=True."
                )
    else:
        raise ValueError(
            f"Unsupported response_format: {response_format}. "
            "Supported formats: pcm, mp3, opus, aac, flac. "
            "For non-PCM formats, you may need to install pydub and simpleaudio."
        )


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
