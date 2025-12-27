"""
Example: Complete Voice Agent Workflow

Demonstrates a complete voice agent workflow combining:
1. Recording audio
2. Speech-to-text transcription
3. Processing/responding (simulated)
4. Text-to-speech response

This is a full conversational loop example.
"""

from voice_agents import (
    record_audio,
    speech_to_text,
    stream_tts,
    format_text_for_speech,
    StreamingTTSCallback
)

def simple_voice_agent():
    """
    A simple voice agent that listens, transcribes, and responds.
    """
    print("=" * 60)
    print("Simple Voice Agent")
    print("=" * 60)
    print()
    
    # Step 1: Record user input
    print("Step 1: Recording audio (5 seconds)...")
    print("Speak now!")
    audio = record_audio(duration=5.0, sample_rate=16000)
    print("Recording complete!")
    print()
    
    # Step 2: Transcribe speech to text
    print("Step 2: Transcribing speech to text...")
    try:
        user_text = speech_to_text(audio_data=audio, sample_rate=16000)
        print(f"User said: '{user_text}'")
    except Exception as e:
        print(f"Error in transcription: {e}")
        return
    print()
    
    # Step 3: Process/respond (simulated agent logic)
    print("Step 3: Processing user input...")
    # Simple response logic
    user_lower = user_text.lower()
    if "hello" in user_lower or "hi" in user_lower:
        response = "Hello! Nice to meet you. How can I help you today?"
    elif "how are you" in user_lower:
        response = "I'm doing great, thank you for asking! How are you?"
    elif "goodbye" in user_lower or "bye" in user_lower:
        response = "Goodbye! Have a wonderful day!"
    else:
        response = f"I heard you say: {user_text}. That's interesting! Tell me more."
    
    print(f"Agent response: '{response}'")
    print()
    
    # Step 4: Convert response to speech
    print("Step 4: Converting response to speech...")
    chunks = format_text_for_speech(response)
    stream_tts(chunks, model="openai/tts-1", voice="alloy")
    print("Response complete!")
    print()


def interactive_voice_agent():
    """
    An interactive voice agent that runs in a loop.
    """
    print("=" * 60)
    print("Interactive Voice Agent")
    print("=" * 60)
    print("Say 'quit' or 'exit' to end the conversation.")
    print()
    
    conversation_count = 0
    
    while True:
        conversation_count += 1
        print(f"\n--- Conversation Turn {conversation_count} ---")
        
        # Record
        print("Listening... (3 seconds)")
        audio = record_audio(duration=3.0, sample_rate=16000)
        
        # Transcribe
        try:
            user_text = speech_to_text(audio_data=audio, sample_rate=16000)
            print(f"You: {user_text}")
            
            # Check for exit command
            if "quit" in user_text.lower() or "exit" in user_text.lower():
                print("\nEnding conversation. Goodbye!")
                stream_tts(["Goodbye! It was nice talking to you."], model="openai/tts-1", voice="alloy")
                break
            
            # Generate response
            response = f"You said: {user_text}. I'm a simple voice agent, so I'm just echoing back what you said for now."
            print(f"Agent: {response}")
            
            # Speak response
            chunks = format_text_for_speech(response)
            stream_tts(chunks, model="openai/tts-1", voice="alloy")
            
        except Exception as e:
            print(f"Error: {e}")
            print("Let's try again...")


def streaming_voice_agent():
    """
    A voice agent that uses streaming TTS callback for real-time responses.
    """
    print("=" * 60)
    print("Streaming Voice Agent with Callback")
    print("=" * 60)
    print()
    
    # Create TTS callback
    tts_callback = StreamingTTSCallback(voice="alloy", model="openai/tts-1")
    
    # Record user input
    print("Recording audio (5 seconds)...")
    audio = record_audio(duration=5.0, sample_rate=16000)
    
    # Transcribe
    print("Transcribing...")
    user_text = speech_to_text(audio_data=audio, sample_rate=16000)
    print(f"User said: {user_text}")
    
    # Simulate agent generating streaming response
    print("Agent generating streaming response...")
    response = f"Thank you for saying: {user_text}. This is a streaming response that is being converted to speech in real-time as it's generated."
    
    # Simulate word-by-word streaming
    words = response.split()
    for word in words:
        tts_callback(word + " ")
    
    # Flush remaining buffer
    tts_callback.flush()
    print("Streaming response complete!")


def voice_agent_with_elevenlabs():
    """
    A voice agent using ElevenLabs for more natural-sounding speech.
    """
    from voice_agents import stream_tts_elevenlabs
    
    print("=" * 60)
    print("Voice Agent with ElevenLabs TTS")
    print("=" * 60)
    print()
    
    # Record
    print("Recording audio (5 seconds)...")
    audio = record_audio(duration=5.0, sample_rate=16000)
    
    # Transcribe
    print("Transcribing...")
    user_text = speech_to_text(audio_data=audio, sample_rate=16000)
    print(f"User said: {user_text}")
    
    # Generate response
    response = f"I heard you say: {user_text}. This response is being spoken using ElevenLabs' high-quality voice synthesis."
    print(f"Agent: {response}")
    
    # Convert to speech with ElevenLabs using unified function
    print("Converting to speech with ElevenLabs...")
    chunks = format_text_for_speech(response)
    # Using unified stream_tts function
    stream_tts(chunks, model="elevenlabs/eleven_multilingual_v2", voice="rachel")
    # Alternative: Using direct function
    # stream_tts_elevenlabs(chunks, voice_id="rachel")
    print("Response complete!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Complete Voice Agent Examples")
    print("=" * 60)
    print()
    print("Choose an example to run:")
    print("1. Simple Voice Agent (one turn)")
    print("2. Interactive Voice Agent (conversation loop)")
    print("3. Streaming Voice Agent (with callback)")
    print("4. Voice Agent with ElevenLabs")
    print()
    
    # Uncomment to run a specific example:
    # simple_voice_agent()
    # interactive_voice_agent()
    # streaming_voice_agent()
    # voice_agent_with_elevenlabs()
    
    print("\nNote: Examples are commented out to prevent accidental execution.")
    print("Uncomment the desired example in the code to run it.")
    print()
    print("Requirements:")
    print("- OPENAI_API_KEY environment variable set")
    print("- ELEVENLABS_API_KEY environment variable set (for ElevenLabs examples)")
    print("- Microphone connected and working")
    print("- sounddevice and soundfile libraries installed")

