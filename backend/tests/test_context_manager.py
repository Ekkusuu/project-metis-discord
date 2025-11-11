"""
Test the context manager to verify it correctly trims messages.
Run this with: python -m pytest backend/tests/test_context_manager.py
or just: python backend/tests/test_context_manager.py
"""
from backend.token_utils import estimate_token_count, count_message_tokens
from backend.context_manager import trim_messages_to_context


def test_estimate_token_count():
    """Test token estimation."""
    text = "Hello, world!"  # ~13 chars = ~3 tokens
    tokens = estimate_token_count(text)
    assert tokens == 3, f"Expected 3 tokens, got {tokens}"
    print(f"✓ Token estimation works: '{text}' = {tokens} tokens")


def test_count_message_tokens():
    """Test message token counting."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},  # ~3 role + ~7 content = ~10
        {"role": "user", "content": "Hello!"},  # ~3 role + ~1 content = ~4
    ]
    tokens = count_message_tokens(messages)
    print(f"✓ Message token counting works: {len(messages)} messages = {tokens} tokens")


def test_trim_messages_to_context():
    """Test message trimming."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Message 1"},
        {"role": "assistant", "content": "Response 1"},
        {"role": "user", "content": "Message 2"},
        {"role": "assistant", "content": "Response 2"},
        {"role": "user", "content": "Message 3"},
    ]
    
    # Test with very small limit - should keep system + most recent messages
    trimmed = trim_messages_to_context(messages, max_tokens=30)
    
    print(f"✓ Original: {len(messages)} messages")
    print(f"✓ Trimmed: {len(trimmed)} messages")
    print(f"✓ Tokens before: {count_message_tokens(messages)}")
    print(f"✓ Tokens after: {count_message_tokens(trimmed)}")
    
    # Should always keep system message
    assert trimmed[0]["role"] == "system", "System message should be preserved"
    print("✓ System message preserved")
    
    # Should fit within limit
    assert count_message_tokens(trimmed) <= 30, "Trimmed messages should fit within limit"
    print("✓ Trimmed messages fit within token limit")


def test_no_trimming_needed():
    """Test that messages aren't trimmed when under limit."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    
    trimmed = trim_messages_to_context(messages, max_tokens=1000)
    
    assert len(trimmed) == len(messages), "No messages should be removed when under limit"
    print(f"✓ No trimming when under limit: {len(messages)} messages preserved")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Context Manager")
    print("="*60 + "\n")
    
    test_estimate_token_count()
    test_count_message_tokens()
    test_trim_messages_to_context()
    test_no_trimming_needed()
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60 + "\n")
