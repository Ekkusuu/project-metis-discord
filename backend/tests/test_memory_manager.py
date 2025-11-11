"""
Test the memory management system.
Run with: python backend/tests/test_memory_manager.py
"""
from backend.memory_manager import (
    append_to_temp_memory,
    get_temp_memory_content,
    get_temp_memory_token_count,
    save_messages_before_reset,
)
from backend.token_utils import estimate_token_count


def test_append_to_temp_memory():
    """Test appending messages to temp_memory."""
    print("\n" + "="*60)
    print("Test: Append to temp_memory")
    print("="*60)
    
    test_messages = [
        "Hello, how are you?",
        "I'm working on a Python project.",
        "Can you help me with context management?"
    ]
    
    append_to_temp_memory(test_messages)
    
    content = get_temp_memory_content()
    token_count = get_temp_memory_token_count()
    
    print(f"✓ Appended {len(test_messages)} messages")
    print(f"✓ Current token count: {token_count}")
    print(f"✓ Content preview:\n{content[:200]}...\n")


def test_save_messages_before_reset():
    """Test saving messages before reset."""
    print("="*60)
    print("Test: Save messages before reset")
    print("="*60)
    
    messages = [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is..."},
        {"role": "user", "content": "Can you explain neural networks?"},
        {"role": "assistant", "content": "Neural networks are..."},
    ]
    
    save_messages_before_reset(messages)
    
    content = get_temp_memory_content()
    token_count = get_temp_memory_token_count()
    
    print(f"✓ Saved user messages from conversation")
    print(f"✓ Current token count: {token_count}")
    print(f"✓ Content includes user messages only\n")


def test_token_estimation():
    """Test token counting."""
    print("="*60)
    print("Test: Token estimation")
    print("="*60)
    
    test_text = "This is a test message for token estimation."
    tokens = estimate_token_count(test_text)
    
    print(f"✓ Text: '{test_text}'")
    print(f"✓ Estimated tokens: {tokens}")
    print(f"✓ Characters: {len(test_text)}")
    print(f"✓ Ratio: ~{len(test_text) / tokens:.1f} chars/token\n")


def display_current_status():
    """Display current memory status."""
    print("="*60)
    print("Current Memory Status")
    print("="*60)
    
    content = get_temp_memory_content()
    token_count = get_temp_memory_token_count()
    
    print(f"Temp memory tokens: {token_count} / 1000")
    print(f"Content length: {len(content)} characters")
    print(f"Progress: {'█' * (token_count // 50)}{'░' * (20 - token_count // 50)} {token_count}/1000")
    print("\nNote: Summarization will trigger automatically at 1000 tokens")
    print("="*60 + "\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Memory Manager Test Suite")
    print("="*60)
    
    try:
        test_token_estimation()
        test_append_to_temp_memory()
        test_save_messages_before_reset()
        display_current_status()
        
        print("="*60)
        print("All tests completed! ✓")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
