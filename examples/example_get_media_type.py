"""
Example: get_media_type_for_format

Demonstrates how to get the appropriate MIME type (media type) for
different audio formats. Useful for FastAPI StreamingResponse.
"""

from voice_agents import get_media_type_for_format

# Example 1: MP3 formats
print("Example 1: MP3 formats")
mp3_formats = [
    "mp3_22050_32",
    "mp3_24000_48",
    "mp3_44100_64",
    "mp3_44100_128",
    "mp3_44100_192"
]

for fmt in mp3_formats:
    media_type = get_media_type_for_format(fmt)
    print(f"  {fmt:20} -> {media_type}")

print()

# Example 2: PCM formats
print("Example 2: PCM formats")
pcm_formats = [
    "pcm_8000",
    "pcm_16000",
    "pcm_22050",
    "pcm_24000",
    "pcm_32000",
    "pcm_44100",
    "pcm_48000"
]

for fmt in pcm_formats:
    media_type = get_media_type_for_format(fmt)
    print(f"  {fmt:20} -> {media_type}")

print()

# Example 3: Opus formats
print("Example 3: Opus formats")
opus_formats = [
    "opus_48000_32",
    "opus_48000_64",
    "opus_48000_96",
    "opus_48000_128",
    "opus_48000_192"
]

for fmt in opus_formats:
    media_type = get_media_type_for_format(fmt)
    print(f"  {fmt:20} -> {media_type}")

print()

# Example 4: Other formats
print("Example 4: Other formats")
other_formats = [
    "ulaw_8000",
    "alaw_8000",
    "aac",
    "flac"
]

for fmt in other_formats:
    media_type = get_media_type_for_format(fmt)
    print(f"  {fmt:20} -> {media_type}")

print()

# Example 5: Usage with FastAPI
print("Example 5: Usage with FastAPI StreamingResponse")
print("""
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from voice_agents import stream_tts_elevenlabs, get_media_type_for_format

app = FastAPI()

@app.get("/tts")
def text_to_speech(text: str, voice_id: str = "rachel"):
    # Generate audio stream
    audio_generator = stream_tts_elevenlabs(
        [text],
        voice_id=voice_id,
        output_format="mp3_44100_128",
        return_generator=True
    )
    
    # Get appropriate media type
    media_type = get_media_type_for_format("mp3_44100_128")
    
    # Return streaming response
    return StreamingResponse(
        audio_generator,
        media_type=media_type
    )
""")
print("FastAPI integration example (code shown above)")

# Example 6: Format validation helper
print("\nExample 6: Format validation helper")
def validate_format_and_get_media_type(format_str: str):
    """Helper function to validate format and get media type."""
    try:
        media_type = get_media_type_for_format(format_str)
        return media_type, True
    except Exception as e:
        return None, False

test_formats = ["mp3_44100_128", "pcm_44100", "invalid_format", "opus_48000_64"]
for fmt in test_formats:
    media_type, valid = validate_format_and_get_media_type(fmt)
    status = "✓" if valid else "✗"
    print(f"  {status} {fmt:20} -> {media_type or 'Invalid format'}")

