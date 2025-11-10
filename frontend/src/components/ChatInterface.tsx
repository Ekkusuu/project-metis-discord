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
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: 'Hello! How can I assist you today?',
      sender: 'ai',
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const API_URL = (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000';

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

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1>Metis AI</h1>
        <p className="subtitle">Your AI Assistant</p>
      </div>

      <div className="messages-container">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`message ${message.sender === 'user' ? 'user-message' : 'ai-message'}`}
          >
            <div className="message-content">
              <div className="message-text">
                <ReactMarkdown 
                  rehypePlugins={[rehypeHighlight]}
                >
                  {message.text
                    .replace(/\n{3,}/g, '\n\n')  // Replace 3+ newlines with 2
                    .replace(/(\d+\.)\n+/g, '$1 ')  // Remove ALL newlines after list numbers
                    .replace(/\n+(\d+\.)/g, '\n$1')  // Single newline before list numbers
                    .replace(/\n+([-*])\s/g, '\n$1 ')  // Single newline before bullet points
                  }
                </ReactMarkdown>
              </div>
              {message.tokensPerSecond && (
                <div className="message-meta" style={{ fontSize: '0.75rem', opacity: 0.6, marginTop: '4px' }}>
                  {message.tokensPerSecond} tokens/sec
                </div>
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
