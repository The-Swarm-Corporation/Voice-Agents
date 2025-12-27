"""
Example: Making HTTP requests to Voice Agents API server using httpx

Demonstrates how to make requests to the FastAPI server endpoints:
- Health check
- List models
- List voices
- Voice agent completions (text-to-speech) with OpenAI model
"""

import httpx
import os
from pathlib import Path


# Server configuration
BASE_URL = os.getenv("VOICE_AGENTS_API_URL", "http://localhost:8000")


def example_health_check():
    """Example: Health check endpoint"""
    print("Example 1: Health Check")
    print("-" * 50)

    with httpx.Client() as client:
        response = client.get(f"{BASE_URL}/v1/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    print()


def example_list_models():
    """Example: List available models"""
    print("Example 2: List Available Models")
    print("-" * 50)

    with httpx.Client() as client:
        response = client.get(f"{BASE_URL}/v1/models")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Found {len(data['models'])} models:")
        for model in data["models"]:
            print(f"  - {model['model']} ({model['provider']})")
    print()


def example_list_voices():
    """Example: List available voices"""
    print("Example 3: List Available Voices")
    print("-" * 50)

    with httpx.Client() as client:
        response = client.get(f"{BASE_URL}/v1/voices")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Found {len(data['voices'])} voices:")
        for voice in data["voices"][:5]:  # Show first 5
            if voice["provider"] == "openai":
                print(f"  - {voice['voice']} ({voice['provider']})")
            else:
                print(f"  - {voice['voice']} ({voice['provider']})")
    print()


def example_openai_completion_basic():
    """Example: Basic OpenAI TTS completion"""
    print("Example 4: Basic OpenAI TTS Completion")
    print("-" * 50)

    request_data = {
        "text": "Hello! This is a test of the Voice Agents API with OpenAI's text-to-speech model.",
        "model": "openai/tts-1",
        "voice": "alloy",
        "response_format": "mp3",  # Options: 'pcm', 'mp3', 'opus', 'aac', 'flac'
    }

    print(f"Request: {request_data['text'][:50]}...")
    print(f"Model: {request_data['model']}")
    print(f"Voice: {request_data['voice']}")
    print(f"Format: {request_data['response_format']}")

    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{BASE_URL}/v1/voice-agent-completions",
            json=request_data,
        )
        print(f"Status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type')}")
        print(
            f"Content-Length: {response.headers.get('content-length', 'unknown')}"
        )

        if response.status_code == 200:
            # Save audio to file
            output_file = Path("output_openai_basic.mp3")
            output_file.write_bytes(response.content)
            print(f"Audio saved to: {output_file}")
        else:
            print(f"Error: {response.text}")
    print()


def example_openai_completion_different_voices():
    """Example: OpenAI TTS with different voices"""
    print("Example 5: OpenAI TTS with Different Voices")
    print("-" * 50)

    text = "This is a demonstration of different OpenAI voices."
    voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    for voice in voices:
        print(f"\nRequesting voice: {voice}")
        request_data = {
            "text": text,
            "model": "openai/tts-1",
            "voice": voice,
            "response_format": "mp3",
        }

        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{BASE_URL}/v1/voice-agent-completions",
                json=request_data,
            )

            if response.status_code == 200:
                output_file = Path(f"output_openai_{voice}.mp3")
                output_file.write_bytes(response.content)
                print(f"  ✓ Saved to: {output_file}")
            else:
                print(
                    f"  ✗ Error: {response.status_code} - {response.text}"
                )
    print()


def example_openai_completion_streaming():
    """Example: OpenAI TTS with streaming mode"""
    print("Example 6: OpenAI TTS with Streaming Mode")
    print("-" * 50)

    request_data = {
        "text": "This is a longer text that will be processed in streaming mode. "
        "The server will process chunks in real-time as they arrive.",
        "model": "openai/tts-1",
        "voice": "nova",
        "stream_mode": True,
        "response_format": "mp3",
    }

    print(f"Request: {request_data['text'][:50]}...")
    print(f"Stream mode: {request_data['stream_mode']}")

    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            f"{BASE_URL}/v1/voice-agent-completions",
            json=request_data,
        )
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            output_file = Path("output_openai_streaming.mp3")
            output_file.write_bytes(response.content)
            print(f"Audio saved to: {output_file}")
        else:
            print(f"Error: {response.text}")
    print()


def example_openai_completion_pcm():
    """Example: OpenAI TTS with PCM format (raw audio)"""
    print("Example 7: OpenAI TTS with PCM Format")
    print("-" * 50)

    request_data = {
        "text": "This audio is in PCM format, which is raw uncompressed audio data.",
        "model": "openai/tts-1",
        "voice": "alloy",
        "response_format": "pcm",  # Raw PCM format
    }

    print(f"Request: {request_data['text']}")
    print(f"Format: {request_data['response_format']} (raw PCM)")

    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{BASE_URL}/v1/voice-agent-completions",
            json=request_data,
        )
        print(f"Status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type')}")

        if response.status_code == 200:
            output_file = Path("output_openai_pcm.raw")
            output_file.write_bytes(response.content)
            print(f"Raw PCM audio saved to: {output_file}")
            print(f"Size: {len(response.content)} bytes")
        else:
            print(f"Error: {response.text}")
    print()


def example_openai_completion_long_text():
    """Example: OpenAI TTS with long text"""
    print("Example 8: OpenAI TTS with Long Text")
    print("-" * 50)

    long_text = """
    Welcome to the Voice Agents API! This is a comprehensive example demonstrating
    how to use the text-to-speech endpoint with OpenAI models. The API supports
    various voices, formats, and streaming modes. You can use this to build
    voice-enabled applications, conversational agents, and more. The server
    automatically formats text for optimal speech synthesis, handling abbreviations,
    URLs, and punctuation appropriately.
    """

    request_data = {
        "text": long_text.strip(),
        "model": "openai/tts-1",
        "voice": "nova",
        "response_format": "mp3",
    }

    print(f"Text length: {len(request_data['text'])} characters")

    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            f"{BASE_URL}/v1/voice-agent-completions",
            json=request_data,
        )
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            output_file = Path("output_openai_long.mp3")
            output_file.write_bytes(response.content)
            print(f"Audio saved to: {output_file}")
            print(f"Size: {len(response.content)} bytes")
        else:
            print(f"Error: {response.text}")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("Voice Agents API - HTTP Request Examples with httpx")
    print("=" * 70)
    print(f"Server URL: {BASE_URL}")
    print()

    # Run examples
    try:
        example_health_check()
        example_list_models()
        example_list_voices()
        example_openai_completion_basic()
        # Uncomment to run additional examples:
        # example_openai_completion_different_voices()
        # example_openai_completion_streaming()
        # example_openai_completion_pcm()
        # example_openai_completion_long_text()

        print("=" * 70)
        print("Examples completed!")
        print("=" * 70)

    except httpx.ConnectError:
        print("ERROR: Could not connect to the server.")
        print(f"Make sure the server is running at {BASE_URL}")
        print("Start the server with: python run_server.py")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
