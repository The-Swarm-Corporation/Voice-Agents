from voice_agents import stream_tts_groq, format_text_for_speech
from dotenv import load_dotenv

# Load environment variables (GROQ_API_KEY)
load_dotenv()


def main():
    # Example 1: Basic text-to-speech with Groq
    print("\n--- Example 1: Basic TTS with Groq ---")
    text = "Hello! This is a test of Groq's fast text to speech using the Orpheus model."
    try:
        # Use default English model and 'austin' voice
        stream_tts_groq(
            text_chunks=[text],
            voice="austin",
            model="canopylabs/orpheus-v1-english",
        )
    except Exception as e:
        print(f"Error in Basic TTS: {e}")

    # Example 2: TTS with vocal directions (emotions)
    print("\n--- Example 2: TTS with vocal directions ---")
    # Groq's Orpheus model supports tags like [cheerful], [sad], [excited], etc.
    expressive_text = "[cheerful] Welcome to the future of AI voice agents! [excited] This is incredibly fast and expressive."
    try:
        stream_tts_groq(
            text_chunks=[expressive_text],
            voice="hannah",
            model="canopylabs/orpheus-v1-english",
        )
    except Exception as e:
        print(f"Error in Expressive TTS: {e}")

    # Example 3: Long text with automatic chunking
    print("\n--- Example 3: Streaming long text ---")
    long_text = """
    Voice Agents is a powerful library for building conversational AI. 
    It supports multiple providers like OpenAI, ElevenLabs, and now Groq. 
    By splitting long text into natural sentences, we can achieve low-latency playback.
    This makes the interaction feel more like a real conversation.
    """

    # Format the text into speech-friendly chunks
    chunks = format_text_for_speech(long_text)

    try:
        # stream_mode=True processes each chunk as it arrives
        stream_tts_groq(
            text_chunks=chunks,
            voice="troy",
            model="canopylabs/orpheus-v1-english",
            stream_mode=True,
        )
    except Exception as e:
        print(f"Error in Streaming TTS: {e}")

    # Example 4: Arabic Text to Speech
    print("\n--- Example 4: Arabic TTS with Groq ---")
    arabic_text = "مرحباً بك في عالم الذكاء الاصطناعي. هذا اختبار لتوليد الصوت باللغة العربية."
    try:
        stream_tts_groq(
            text_chunks=[arabic_text],
            voice="salma",
            model="canopylabs/orpheus-arabic-saudi",
        )
    except Exception as e:
        print(f"Error in Arabic TTS: {e}")


if __name__ == "__main__":
    main()
