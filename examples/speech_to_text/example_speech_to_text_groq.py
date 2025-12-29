import os
from voice_agents import speech_to_text_groq, record_audio
from dotenv import load_dotenv

# Load environment variables (GROQ_API_KEY)
load_dotenv()


def main():
    # 1. Transcribe from a recorded audio
    print("\n--- Example 1: Transcribing from live recording ---")
    try:
        # Record 5 seconds of audio
        audio_data = record_audio(duration=5.0)

        # Transcribe using Groq
        text = speech_to_text_groq(
            audio_data=audio_data,
            model="whisper-large-v3-turbo",
            response_format="text",
        )
        print(f"Transcription: {text}")
    except Exception as e:
        print(f"Error in recording/transcription: {e}")

    # 2. Transcribe from an existing file
    print("\n--- Example 2: Transcribing from an existing file ---")
    # Replace with a path to your audio file (mp3, wav, etc.)
    audio_path = "path/to/your/audio.wav"

    if os.path.exists(audio_path):
        try:
            text = speech_to_text_groq(
                audio_file_path=audio_path,
                model="whisper-large-v3-turbo",
            )
            print(f"File Transcription: {text}")
        except Exception as e:
            print(f"Error transcribing file: {e}")
    else:
        print(f"File {audio_path} not found, skipping Example 2.")

    # 3. Translate to English using Groq
    print("\n--- Example 3: Translating audio to English ---")
    try:
        print("Record something in a non-English language...")
        audio_data_translation = record_audio(duration=5.0)

        # Use whisper-large-v3 for translation support
        translated_text = speech_to_text_groq(
            audio_data=audio_data_translation,
            model="whisper-large-v3",
            translate=True,
        )
        print(f"Translated Text (to English): {translated_text}")
    except Exception as e:
        print(f"Error in translation: {e}")

    # 4. Get detailed transcription with timestamps
    print(
        "\n--- Example 4: Detailed transcription with timestamps ---"
    )
    try:
        audio_data_detailed = record_audio(duration=3.0)

        # Get verbose JSON response with word-level timestamps
        detailed_result = speech_to_text_groq(
            audio_data=audio_data_detailed,
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )
        print("Detailed Transcription Result:")
        print(detailed_result)
    except Exception as e:
        print(f"Error in detailed transcription: {e}")


if __name__ == "__main__":
    main()
