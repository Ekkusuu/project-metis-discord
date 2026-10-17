import { useCallback, useEffect, useState } from 'react';
import { AI_MESSAGE_EVENT, getChatContext } from '../api';
import type { ContextMessage } from '../api';
import './ChatContext.css';

const ROLE_LABELS: Record<ContextMessage['role'], string> = {
  system: '⚙️ System',
  user: '👤 User',
  assistant: '🤖 Metis',
};

function ChatContext() {
  const [messages, setMessages] = useState<ContextMessage[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState<Record<number, boolean>>({});

  const fetchContext = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getChatContext();
      setMessages(data.messages ?? []);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchContext();
    window.addEventListener(AI_MESSAGE_EVENT, fetchContext);
    return () => window.removeEventListener(AI_MESSAGE_EVENT, fetchContext);
  }, [fetchContext]);

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Prompt context</h2>
        <button
          type="button"
          className="ctx-refresh"
          onClick={fetchContext}
          disabled={loading}
          title="Refresh"
        >
          {loading ? '…' : '🔄'} Refresh
        </button>
      </div>
      <div className="ctx-count">{messages.length} messages sent to the LLM</div>

      {error && <div className="ctx-error">Failed to load context: {error}</div>}

      {messages.length === 0 && !error ? (
        <div className="panel-empty">No context yet</div>
      ) : (
        messages.map((msg, i) => {
          const isSystem = msg.role === 'system';
          const open = !isSystem || !!expanded[i];
          return (
            <div key={i} className={`ctx-message ${msg.role}`}>
              <div className="ctx-head">
                <span className="ctx-role">{ROLE_LABELS[msg.role] ?? msg.role}</span>
                {msg.timestamp && (
                  <span className="ctx-time">{new Date(msg.timestamp).toLocaleTimeString()}</span>
                )}
                {isSystem && (
                  <button
                    type="button"
                    className="ctx-toggle"
                    onClick={() => setExpanded((s) => ({ ...s, [i]: !s[i] }))}
                    aria-expanded={open}
                  >
                    {open ? 'hide' : `show (${msg.content.length} chars)`}
                  </button>
                )}
              </div>
              <div className={`ctx-content ${open ? '' : 'collapsed'}`}>{msg.content}</div>
            </div>
          );
        })
      )}
    </div>
  );
}

export default ChatContext;
