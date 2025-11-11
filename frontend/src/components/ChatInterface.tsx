import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import './ChatInterface.css';
import 'highlight.js/styles/github-dark.css';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'ai';
  timestamp: Date;
  tokensPerSecond?: number;
}

function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null);
  const [editingText, setEditingText] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const API_URL = (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000';

  // Load chat history on mount
  useEffect(() => {
    loadChatHistory();
  }, []);

  // Save chat history whenever messages change (but not on initial load)
  useEffect(() => {
    if (!isLoading && messages.length > 0) {
      saveChatHistory();
    }
  }, [messages]);

  const loadChatHistory = async () => {
    try {
      const response = await fetch(`${API_URL}/history/load`);
      if (response.ok) {
        const data = await response.json();
        setMessages(data.messages.map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp),
        })));
      }
    } catch (error) {
      console.error('Failed to load chat history:', error);
      // Fallback to default message
      setMessages([
        {
          id: '1',
          text: 'Hello! How can I assist you today?',
          sender: 'ai',
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const saveChatHistory = async () => {
    try {
      await fetch(`${API_URL}/history/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: messages.map((msg) => ({
            ...msg,
            timestamp: msg.timestamp.toISOString(),
          })),
          lastUpdated: new Date().toISOString(),
        }),
      });
    } catch (error) {
      console.error('Failed to save chat history:', error);
    }
  };

  const handleResetChat = async () => {
    if (!confirm('Are you sure you want to reset the chat? This will delete all messages.')) {
      return;
    }

    try {
      const response = await fetch(`${API_URL}/history/reset`, {
        method: 'POST',
      });
      
      if (response.ok) {
        const data = await response.json();
        setMessages(data.messages.map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp),
        })));
      }
    } catch (error) {
      console.error('Failed to reset chat:', error);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const autoResizeTextarea = () => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${textarea.scrollHeight}px`;
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    autoResizeTextarea();
  }, [input]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: input,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);
    
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }

    // Streaming API call
    try {
      const history = [
        ...messages.map((m) => ({
          role: m.sender === 'user' ? 'user' : 'assistant',
          content: m.text,
        })),
        { role: 'user', content: userMessage.text },
      ];

      const resp = await fetch(`${API_URL}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: history, max_tokens: 512 }),
      });

      if (!resp.ok) {
        const detail = await resp.text();
        throw new Error(`Backend error ${resp.status}: ${detail}`);
      }

      // Dispatch event to show RAG results immediately (before response completes)
      window.dispatchEvent(new CustomEvent('ragRetrievalComplete'));

      // Read NDJSON stream
      const reader = resp.body?.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let firstTokenReceived = false;
      const aiMessageId = (Date.now() + 1).toString();

      if (!reader) throw new Error('No response body');

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const chunk = JSON.parse(line);
            if (chunk.delta !== undefined && chunk.delta !== '') {
              // On first token, create the AI message and hide typing indicator
              if (!firstTokenReceived) {
                firstTokenReceived = true;
                setIsTyping(false);
                const aiMessage: Message = {
                  id: aiMessageId,
                  text: chunk.delta,
                  sender: 'ai',
                  timestamp: new Date(),
                };
                setMessages((prev) => [...prev, aiMessage]);
              } else {
                // Append subsequent tokens to the AI message
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === aiMessageId
                      ? { ...msg, text: msg.text + chunk.delta }
                      : msg
                  )
                );
              }
            }
            // Check for final chunk with performance stats
            if (chunk.done && chunk.tokens_per_second) {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === aiMessageId
                    ? { ...msg, tokensPerSecond: chunk.tokens_per_second }
                    : msg
                )
              );
              
              // Handle context trimming - remove messages that were trimmed by backend
              if (chunk.trimmed_messages && Array.isArray(chunk.trimmed_messages)) {
                const trimmedCount = history.length - chunk.trimmed_messages.length;
                if (trimmedCount > 0) {
                  console.log(`Context limit reached: removed ${trimmedCount} oldest message(s)`);
                  // Remove oldest messages from frontend state (keep system + trimmed messages + new AI response)
                  setMessages((prev) => {
                    // Find how many messages to remove from the start (excluding initial AI greeting)
                    const messagesToKeep = prev.length - trimmedCount;
                    return prev.slice(-messagesToKeep);
                  });
                }
              }
            }
          } catch (parseErr) {
            console.warn('Failed to parse chunk:', line, parseErr);
          }
        }
      }

      // If no tokens were received, hide typing indicator
      if (!firstTokenReceived) {
        setIsTyping(false);
      }

      // Dispatch event to notify components that message is complete
      window.dispatchEvent(new CustomEvent('messageComplete'));
    } catch (err: any) {
      setIsTyping(false);
      const errMsg: Message = {
        id: (Date.now() + 2).toString(),
        text: `Error contacting AI backend: ${err?.message || String(err)}`,
        sender: 'ai',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errMsg]);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleEditMessage = (messageId: string, currentText: string) => {
    setEditingMessageId(messageId);
    setEditingText(currentText);
  };

  const handleCancelEdit = () => {
    setEditingMessageId(null);
    setEditingText('');
  };

  const handleSaveEdit = async (messageId: string) => {
    if (!editingText.trim()) return;

    // Find the index of the message being edited
    const messageIndex = messages.findIndex((m) => m.id === messageId);
    if (messageIndex === -1) return;

    // Delete all messages after this one
    const updatedMessages = messages.slice(0, messageIndex);
    
    // Update the edited message
    const editedMessage: Message = {
      ...messages[messageIndex],
      text: editingText,
      timestamp: new Date(),
    };

    setMessages([...updatedMessages, editedMessage]);
    setEditingMessageId(null);
    setEditingText('');
    setIsTyping(true);

    // Send the edited message to get a new AI response
    try {
      const history = [
        ...updatedMessages.map((m) => ({
          role: m.sender === 'user' ? 'user' : 'assistant',
          content: m.text,
        })),
        { role: 'user', content: editingText },
      ];

      const resp = await fetch(`${API_URL}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: history, max_tokens: 512 }),
      });

      if (!resp.ok) {
        const detail = await resp.text();
        throw new Error(`Backend error ${resp.status}: ${detail}`);
      }

      window.dispatchEvent(new CustomEvent('ragRetrievalComplete'));

      const reader = resp.body?.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let firstTokenReceived = false;
      const aiMessageId = (Date.now() + 1).toString();

      if (!reader) throw new Error('No response body');

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const chunk = JSON.parse(line);
            if (chunk.delta !== undefined && chunk.delta !== '') {
              if (!firstTokenReceived) {
                firstTokenReceived = true;
                setIsTyping(false);
                const aiMessage: Message = {
                  id: aiMessageId,
                  text: chunk.delta,
                  sender: 'ai',
                  timestamp: new Date(),
                };
                setMessages((prev) => [...prev, aiMessage]);
              } else {
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === aiMessageId
                      ? { ...msg, text: msg.text + chunk.delta }
                      : msg
                  )
                );
              }
            }
            if (chunk.done && chunk.tokens_per_second) {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === aiMessageId
                    ? { ...msg, tokensPerSecond: chunk.tokens_per_second }
                    : msg
                )
              );
              
              // Handle context trimming
              if (chunk.trimmed_messages && Array.isArray(chunk.trimmed_messages)) {
                const trimmedCount = history.length - chunk.trimmed_messages.length;
                if (trimmedCount > 0) {
                  console.log(`Context limit reached: removed ${trimmedCount} oldest message(s)`);
                  setMessages((prev) => {
                    const messagesToKeep = prev.length - trimmedCount;
                    return prev.slice(-messagesToKeep);
                  });
                }
              }
            }
          } catch (parseErr) {
            console.warn('Failed to parse chunk:', line, parseErr);
          }
        }
      }

      if (!firstTokenReceived) {
        setIsTyping(false);
      }

      window.dispatchEvent(new CustomEvent('messageComplete'));
    } catch (err: any) {
      setIsTyping(false);
      const errMsg: Message = {
        id: (Date.now() + 2).toString(),
        text: `Error contacting AI backend: ${err?.message || String(err)}`,
        sender: 'ai',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errMsg]);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <div className="header-content">
          <div className="header-text">
            <h1>Metis AI</h1>
            <p className="subtitle">Your AI Assistant</p>
          </div>
          <button onClick={handleResetChat} className="reset-button" title="Reset chat">
            🔄 Reset
          </button>
        </div>
      </div>

      <div className="messages-container">
        {isLoading ? (
          <div className="loading-history">
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
            <p>Loading chat history...</p>
          </div>
        ) : (
          <>
            {messages.map((message) => (
          <div
            key={message.id}
            className={`message ${message.sender === 'user' ? 'user-message' : 'ai-message'}`}
          >
            <div className="message-content">
              {editingMessageId === message.id ? (
                // Edit mode
                <div className="edit-mode">
                  <textarea
                    value={editingText}
                    onChange={(e) => setEditingText(e.target.value)}
                    className="edit-textarea"
                    autoFocus
                  />
                  <div className="edit-buttons">
                    <button onClick={() => handleSaveEdit(message.id)} className="save-edit-button">
                      Save
                    </button>
                    <button onClick={handleCancelEdit} className="cancel-edit-button">
                      Cancel
                    </button>
                  </div>
                </div>
              ) : (
                // Normal display mode
                <>
                  <div className="message-text">
                    <ReactMarkdown 
                      rehypePlugins={[rehypeHighlight]}
                    >
                      {message.text
                        // Normalize Windows newlines first
                        .replace(/\r\n/g, '\n')
                        .replace(/\r/g, '\n')
                        // Replace 3+ newlines with 2 to avoid tall gaps
                        .replace(/\n{3,}/g, '\n\n')
                        // Ensure numbered list marker sticks to following content (e.g., "1. **Title**")
                        .replace(/(\d+\.)[ \t]*\n+/g, '$1 ')
                        // Normalize a single newline before list numbers (avoid extra space above items)
                        .replace(/\n+(\d+\.)/g, '\n$1')
                        // Normalize a single newline before bullets and remove stray indent spaces
                        .replace(/\n+[ \t]*([-*])\s/g, '\n$1 ')
                        // Convert remaining single newlines to hard line breaks without touching paragraph breaks
                        .replace(/([^\n])\n(?!\n)/g, '$1  \n')
                      }
                    </ReactMarkdown>
                  </div>
                  {message.sender === 'user' && (
                    <button
                      onClick={() => handleEditMessage(message.id, message.text)}
                      className="edit-message-button"
                      title="Edit message"
                    >
                      ✏️
                    </button>
                  )}
                  {message.tokensPerSecond && (
                    <div className="message-meta" style={{ fontSize: '0.75rem', opacity: 0.6, marginTop: '4px' }}>
                      {message.tokensPerSecond} tokens/sec
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        ))}
        {isTyping && (
          <div className="message ai-message">
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
          </>
        )}
      </div>

      <div className="input-container">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message..."
          rows={1}
          className="message-input"
          disabled={isTyping}
        />
        <button onClick={handleSend} className="send-button" disabled={isTyping}>
          {isTyping ? 'Wait...' : 'Send'}
        </button>
      </div>
    </div>
  );
}

export default ChatInterface;
