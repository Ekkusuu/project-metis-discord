import { useState, useEffect } from 'react';
import './ChatContext.css';

interface ContextMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
  timestamp?: string;
}

interface ChatContextProps {
  messages?: ContextMessage[];
}

function ChatContext({ messages: externalMessages }: ChatContextProps) {
  const [messages, setMessages] = useState<ContextMessage[]>([]);
  const API_URL = (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000';

  const fetchContext = async () => {
    try {
      const resp = await fetch(`${API_URL}/chat/context`);
      const data = await resp.json();
      setMessages(data.messages || []);
    } catch (error) {
      console.error('Failed to fetch chat context:', error);
    }
  };

  useEffect(() => {
    if (externalMessages) {
      setMessages(externalMessages);
    } else {
      // Initial fetch
      fetchContext();

      // Listen for message completion events
      const handleMessageComplete = () => {
        fetchContext();
      };

      window.addEventListener('messageComplete', handleMessageComplete);
      return () => window.removeEventListener('messageComplete', handleMessageComplete);
    }
  }, [externalMessages]);

  const getRoleLabel = (role: string) => {
    switch (role) {
      case 'system':
        return '⚙️ System';
      case 'user':
        return '👤 You';
      case 'assistant':
        return '🤖 Metis';
      default:
        return role;
    }
  };

  const getRoleClass = (role: string) => {
    switch (role) {
      case 'system':
        return 'system';
      case 'user':
        return 'user';
      case 'assistant':
        return 'assistant';
      default:
        return '';
    }
  };

  return (
    <div className="chat-context">
      <div className="context-header">
        <h2>Chat Context</h2>
        <span className="message-count">{messages.length} messages</span>
      </div>

      <div className="context-list">
        {messages.length === 0 ? (
          <div className="empty-state">
            <p>No context yet</p>
            <p className="empty-hint">Start a conversation to see the context</p>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <div key={idx} className={`context-message ${getRoleClass(msg.role)}`}>
              <div className="message-role">{getRoleLabel(msg.role)}</div>
              <div className="message-content">{msg.content}</div>
              {msg.timestamp && (
                <div className="message-timestamp">{new Date(msg.timestamp).toLocaleTimeString()}</div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default ChatContext;
