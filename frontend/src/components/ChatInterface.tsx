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
  discord_user?: string;
  channel_name?: string;
}

// Component: render message text and collapse any <think> sections by default
function MessageText({ rawText, messageId }: { rawText: string; messageId: string }) {
  const [expanded, setExpanded] = useState<Record<number, boolean>>({});

  const normalize = (t: string) =>
    t
      .replace(/\r\n/g, '\n')
      .replace(/\r/g, '\n')
      .replace(/\n{3,}/g, '\n\n')
      .replace(/(\d+\.)[ \t]*\n+/g, '$1 ')
      .replace(/\n+(\d+\.)/g, '\n$1')
      .replace(/\n+[ \t]*([-*])\s/g, '\n$1 ')
      .replace(/([^\n])\n(?!\n)/g, '$1  \n');

  // Parse think blocks. Support three cases:
  // - <think>...</think>
  // - stray </think> (treat preceding text as thinking)
  // - stray <think> without close (treat rest as thinking)
  const parseParts = (text: string) => {
    const parts: Array<{ type: 'text' | 'think'; content: string }> = [];

    // If there's an opening tag without a closing tag, treat remainder as think
    if (/\<think\>/i.test(text) && !/\<\/think\>/i.test(text)) {
      const openIdx = text.search(/\<think\>/i);
      if (openIdx > -1) {
        const before = text.slice(0, openIdx);
        const thinkContent = text.slice(openIdx + 7);
        if (before) parts.push({ type: 'text', content: before });
        parts.push({ type: 'think', content: thinkContent });
        return parts;
      }
    }

    // Find normal <think>...</think> blocks
    const re = /<think>([\s\S]*?)<\/think>/i;
    let remaining = text;
    let m: RegExpExecArray | null;
    while ((m = re.exec(remaining)) !== null) {
      const idx = m.index;
      const before = remaining.slice(0, idx);
      if (before) parts.push({ type: 'text', content: before });
      parts.push({ type: 'think', content: m[1] });
      remaining = remaining.slice(idx + m[0].length);
    }

    // If there were no normal blocks but there's a stray closing tag, treat up-to-close as think
    if (parts.length === 0 && /<\/think>/i.test(remaining) && !/\<think\>/i.test(text)) {
      const closeIdx = remaining.search(/<\/think>/i);
      if (closeIdx > -1) {
        const thinkContent = remaining.slice(0, closeIdx);
        const after = remaining.slice(closeIdx + 8);
        parts.push({ type: 'think', content: thinkContent });
        if (after) parts.push({ type: 'text', content: after });
        return parts;
      }
    }

    if (remaining) parts.push({ type: 'text', content: remaining });
    return parts;
  };

  const parts = parseParts(rawText);

  const toggle = (i: number) => setExpanded((s) => ({ ...s, [i]: !s[i] }));

  return (
    <>
      {parts.map((p, i) =>
        p.type === 'text' ? (
          <ReactMarkdown key={`${messageId}-txt-${i}`} rehypePlugins={[rehypeHighlight]}>
            {normalize(p.content)}
          </ReactMarkdown>
        ) : (
          <div key={`${messageId}-think-${i}`} className="think-block">
            <button
              className="think-toggle"
              onClick={() => toggle(i)}
              aria-expanded={!!expanded[i]}
            >
              {expanded[i] ? 'Hide reasoning' : 'Show reasoning'}
            </button>
            {expanded[i] && (
              <div className="think-content">
                <ReactMarkdown rehypePlugins={[rehypeHighlight]}>{normalize(p.content)}</ReactMarkdown>
              </div>
            )}
          </div>
        )
      )}
    </>
  );
}
function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [discordConnected, setDiscordConnected] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const API_URL = (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000';

  // Load chat history on mount and connect to Discord stream
  useEffect(() => {
    loadChatHistory();
    connectToDiscordStream();
    
    return () => {
      // Cleanup: close EventSource on unmount
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  // Note: Chat history saving is disabled - messages come from Discord
  // The Discord bot handles all message storage and retrieval

  const connectToDiscordStream = () => {
    try {
      const eventSource = new EventSource(`${API_URL}/discord/stream`);
      eventSourceRef.current = eventSource;

      eventSource.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          
          // Only add message if it has content (prevent empty bubbles)
          if (!message.text || message.text.trim() === '') {
            console.log('Skipping empty message:', message);
            return;
          }
          
          // Add Discord message to the messages state
          setMessages((prev) => {
            // Check if message already exists (by id)
            if (prev.some(m => m.id === message.id)) {
              return prev;
            }
            return [...prev, {
              id: message.id,
              text: message.text,
              sender: message.sender,
              timestamp: new Date(message.timestamp),
              tokensPerSecond: message.tokensPerSecond
            }];
          });
          
          setDiscordConnected(true);
        } catch (err) {
          console.error('Failed to parse Discord message:', err);
        }
      };

      eventSource.onerror = (err) => {
        console.error('Discord stream error:', err);
        setDiscordConnected(false);
        
        // Attempt to reconnect after 5 seconds
        setTimeout(() => {
          if (eventSourceRef.current) {
            eventSourceRef.current.close();
          }
          connectToDiscordStream();
        }, 5000);
      };

      eventSource.onopen = () => {
        console.log('Connected to Discord message stream');
        setDiscordConnected(true);
      };
    } catch (error) {
      console.error('Failed to connect to Discord stream:', error);
      setDiscordConnected(false);
    }
  };

  const loadChatHistory = async () => {
    try {
      // Load existing Discord messages instead of chat history
      const response = await fetch(`${API_URL}/discord/messages`);
      if (response.ok) {
        const data = await response.json();
        setMessages(data.messages.map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp),
        })));
      }
    } catch (error) {
      console.error('Failed to load Discord messages:', error);
      // Start with empty message list - Discord messages will come via stream
      setMessages([]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearMessages = async () => {
    if (!confirm('Clear all Discord messages from the UI? (This will not affect Discord conversation)')) {
      return;
    }

    try {
      const response = await fetch(`${API_URL}/discord/clear`, {
        method: 'POST',
      });
      
      if (response.ok) {
        setMessages([]);
      }
    } catch (error) {
      console.error('Failed to clear messages:', error);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Input is disabled - all messages come from Discord

  return (
    <div className="chat-container">
      <div className="chat-header">
        <div className="header-content">
          <div className="header-text">
            <h1>Metis AI - Discord Bot</h1>
            <p className="subtitle">
              {discordConnected ? '🟢 Connected to Discord' : '🔴 Connecting...'}
            </p>
          </div>
          <button onClick={handleClearMessages} className="reset-button" title="Clear messages">
            🗑️ Clear
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
              <div className="message-text">
                <MessageText rawText={message.text} messageId={message.id} />
              </div>
              {message.tokensPerSecond && (
                <div className="message-meta" style={{ fontSize: '0.75rem', opacity: 0.6, marginTop: '4px' }}>
                  {message.tokensPerSecond} tokens/sec
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
          </>
        )}
      </div>

      <div className="input-container" style={{ backgroundColor: '#2a2a2a', padding: '1rem', borderTop: '1px solid #444' }}>
        <div style={{ textAlign: 'center', color: '#888', fontSize: '0.9rem' }}>
          💬 Input disabled - All messages come from Discord
        </div>
        <textarea
          value=""
          placeholder="Messages are sent from Discord..."
          rows={1}
          className="message-input"
          disabled={true}
          style={{ opacity: 0.5, cursor: 'not-allowed' }}
        />
        <button className="send-button" disabled={true} style={{ opacity: 0.5, cursor: 'not-allowed' }}>
          Disabled
        </button>
      </div>
    </div>
  );
}

export default ChatInterface;
