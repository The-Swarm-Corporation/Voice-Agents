"""
Example: StreamingTTSCallback

Demonstrates how to use the StreamingTTSCallback class to convert
streaming text output from agents into real-time speech.
"""

from voice_agents import StreamingTTSCallback

# Example 1: Basic usage with streaming text
print("Example 1: Basic usage with streaming text")
callback = StreamingTTSCallback(voice="alloy", model="tts-1")

# Simulate streaming text from an agent
streaming_text = [
    "Hello! ",
    "This is a ",
    "streaming text ",
    "example. ",
    "The callback will ",
    "detect complete sentences ",
    "and convert them to speech. ",
    "Pretty cool, right?"
]

print("Simulating streaming text chunks...")
for chunk in streaming_text:
    print(f"Received chunk: '{chunk}'")
    callback(chunk)

# Flush any remaining text
print("\nFlushing remaining buffer...")
callback.flush()
print("Streaming complete!\n")

# Example 2: Custom voice and model
print("Example 2: Custom voice and model")
callback_nova = StreamingTTSCallback(voice="nova", model="tts-1-hd")
print("Using nova voice with tts-1-hd model...")

streaming_text2 = [
    "This is using ",
    "a different voice. ",
    "The nova voice ",
    "sounds distinct ",
    "from alloy."
]

for chunk in streaming_text2:
    callback_nova(chunk)
callback_nova.flush()
print("Custom voice example complete!\n")

# Example 3: Minimum sentence length
print("Example 3: Custom minimum sentence length")
# Set minimum length to 20 characters
callback_long = StreamingTTSCallback(
    voice="alloy",
    min_sentence_length=20
)
print("Minimum sentence length set to 20 characters...")

streaming_text3 = [
    "Short. ",  # Too short, won't trigger
    "This is a longer sentence that will trigger. ",  # Will trigger
    "Another short one. ",  # Too short
    "This is also long enough to be converted to speech. "  # Will trigger
]

for chunk in streaming_text3:
    print(f"Processing: '{chunk}'")
    callback_long(chunk)
callback_long.flush()
print("Minimum length example complete!\n")

# Example 4: Simulating agent streaming output
print("Example 4: Simulating agent streaming output")
print("This mimics how an AI agent might stream its response...")

def simulate_agent_response():
    """Simulate an agent generating a response word by word."""
    response = "I can help you with that question. Let me think about the best approach. " \
               "First, I'll analyze the problem. Then, I'll provide a solution. " \
               "Does that sound good to you?"
    
    # Split into word-like chunks (simulating token streaming)
    words = response.split()
    for word in words:
        yield word + " "

callback_agent = StreamingTTSCallback(voice="alloy")
print("Agent is thinking and speaking...")
for word_chunk in simulate_agent_response():
    callback_agent(word_chunk)
callback_agent.flush()
print("Agent response complete!\n")

# Example 5: Error handling
print("Example 5: Error handling")
print("The callback handles errors gracefully...")
callback_safe = StreamingTTSCallback(voice="alloy")

# Even if there's an error in TTS, the callback continues
streaming_text4 = [
    "This will work fine. ",
    "Even if there's an issue, ",
    "the callback continues. ",
    "Pretty robust!"
]

for chunk in streaming_text4:
    try:
        callback_safe(chunk)
    except Exception as e:
        print(f"Error handled: {e}")

callback_safe.flush()
print("Error handling example complete!\n")

# Example 6: Integration with agent frameworks
print("Example 6: Integration with agent frameworks")
print("""
To use with an agent framework (like LangChain, AutoGen, etc.):

from voice_agents import StreamingTTSCallback
from your_agent_framework import Agent

# Create callback
tts_callback = StreamingTTSCallback(voice="alloy")

# Use as streaming callback
agent = Agent(streaming_callback=tts_callback)

# When agent streams output, it will automatically be converted to speech
response = agent.run("Tell me about artificial intelligence")

# Don't forget to flush at the end
tts_callback.flush()
""")
print("Integration example (code shown above)")

