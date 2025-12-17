"""
Dynamic Tool: text_stats
Created: 2025-12-17T11:00:00.229927
Description: Count words, characters, and lines in a text string. Returns detailed statistics including total characters, characters without spaces, word count, line count, and average word length.

Dependencies: []
"""

def run(text: str) -> str:
    """Count words, characters, and lines in a text string."""
    if not text:
        return "No text provided"
    
    # Count characters (total and without spaces)
    total_chars = len(text)
    chars_no_spaces = len(text.replace(' ', ''))
    
    # Count words (split by whitespace)
    words = text.split()
    word_count = len(words)
    
    # Count lines
    lines = text.split('\n')
    line_count = len(lines)
    
    # Calculate average word length
    if word_count > 0:
        avg_word_length = sum(len(word.strip('.,!?;:"()[]{}')) for word in words) / word_count
    else:
        avg_word_length = 0
    
    # Format results
    result = f"""Text Statistics:
================
Total characters: {total_chars}
Characters (no spaces): {chars_no_spaces}
Words: {word_count}
Lines: {line_count}
Average word length: {avg_word_length:.1f} characters

Text preview (first 100 chars):
{text[:100]}{'...' if len(text) > 100 else ''}
"""
    
    return result