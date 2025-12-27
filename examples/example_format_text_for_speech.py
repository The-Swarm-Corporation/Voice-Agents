"""
Example: format_text_for_speech

Demonstrates how to format long text into speech-friendly chunks.
This function splits text on sentence boundaries while handling abbreviations,
URLs, and other edge cases.
"""

from voice_agents import format_text_for_speech

# Example 1: Simple text with multiple sentences
text1 = "Hello world! This is a test. How are you today?"
chunks1 = format_text_for_speech(text1)
print("Example 1 - Simple text:")
print(f"Original: {text1}")
print(f"Chunks: {chunks1}")
print()

# Example 2: Text with abbreviations (should not split on abbreviations)
text2 = "Dr. Smith went to the U.S.A. He met Mr. Johnson there. They discussed A.I. technology."
chunks2 = format_text_for_speech(text2)
print("Example 2 - Text with abbreviations:")
print(f"Original: {text2}")
print(f"Chunks: {chunks2}")
print()

# Example 3: Long paragraph with various punctuation
text3 = """
Welcome to the Voice Agents library! This is an amazing tool for text-to-speech.
You can use it to create voice agents, build conversational interfaces, and more.
What do you think? Let's explore together!
"""
chunks3 = format_text_for_speech(text3)
print("Example 3 - Long paragraph:")
print(f"Original: {text3.strip()}")
print(f"Chunks: {chunks3}")
print()

# Example 4: Text with URLs and email addresses
text4 = "Visit https://example.com for more info. Contact us at info@example.com. Thanks!"
chunks4 = format_text_for_speech(text4)
print("Example 4 - Text with URLs:")
print(f"Original: {text4}")
print(f"Chunks: {chunks4}")
print()

# Example 5: Text with decimal numbers
text5 = "The value is 3.14. The price is $19.99. That's great!"
chunks5 = format_text_for_speech(text5)
print("Example 5 - Text with decimals:")
print(f"Original: {text5}")
print(f"Chunks: {chunks5}")
print()

