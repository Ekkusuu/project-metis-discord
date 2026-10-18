import { memo, useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { AI_MESSAGE_EVENT, API_URL, clearMessages, getMessages } from '../api';
import type { DiscordMessage } from '../api';
import { useEventSource } from '../hooks/useEventSource';
import MessageText from './MessageText';
import StatusBar from './StatusBar';
import Icon from './Icon';
import './ChatInterface.css';

const NEAR_BOTTOM_PX = 120;

/** Merge by id (later entries win) and keep chronological order by seq. */
function mergeMessages(
  current: DiscordMessage[],
  incoming: DiscordMessage[],
): DiscordMessage[] {
  const byId = new Map<string, DiscordMessage>();
  for (const m of current) byId.set(m.id, m);
  for (const m of incoming) {
    if (m.text && m.text.trim() !== '') byId.set(m.id, m);
  }
  return [...byId.values()].sort((a, b) => a.seq - b.seq);
}

/** Resolve the display name and body of a message, stripping the `[name]:` / `[name]` prefix. */
function describeMessage(m: DiscordMessage): { author: string; body: string } {
  if (m.sender === 'ai') return { author: 'Metis', body: m.text };

  let author = m.discord_user;
  let body = m.text;
  if (author && body.startsWith(`[${author}]`)) {
    body = body.slice(author.length + 2).replace(/^:?\s*/, '');
  } else {
    const match = /^\[([^\]]+)\]:?\s*/.exec(body);
    if (match) {
      author ??= match[1];
      body = body.slice(match[0].length);
    }
  }
  return { author: author ?? 'User', body };
}

function formatTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

const MessageItem = memo(function MessageItem({ message }: { message: DiscordMessage }) {
  const { author, body } = describeMessage(message);
  const isAi = message.sender === 'ai';
  const time = formatTime(message.timestamp);

  return (
    <div className={`message ${isAi ? 'ai' : 'user'}`}>
      {isAi && <img className="message-avatar" src="/metis-avatar.png" alt="" />}
      <div className="message-bubble">
        <div className="message-meta">
          <span className="message-author">{author}</span>
          {message.channel_name && <span className="message-channel">#{message.channel_name}</span>}
          {message.mode === 'voice' && (
            <span className="message-badge" title="Voice chat">
              <Icon name="mic" size={12} /> voice
            </span>
          )}
          {time && <span className="message-time">{time}</span>}
        </div>
        <div className="message-body">
          <MessageText text={body} />
        </div>
        {message.tokensPerSecond != null && (
          <div className="message-tps">{message.tokensPerSecond.toFixed(1)} tokens/sec</div>
        )}
      </div>
    </div>
  );
});

function ChatInterface() {
  const [messages, setMessages] = useState<DiscordMessage[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hasNew, setHasNew] = useState(false);

  const listRef = useRef<HTMLDivElement>(null);
  const nearBottomRef = useRef(true);
  const prevCountRef = useRef(0);

  const connected = useEventSource(`${API_URL}/discord/stream`, {
    onMessage: (event) => {
      let message: DiscordMessage;
      try {
        message = JSON.parse(event.data) as DiscordMessage;
      } catch (err) {
        console.error('Failed to parse Discord message:', err);
        return;
      }
      if (!message.text || message.text.trim() === '') return;
      setMessages((prev) =>
        prev.some((m) => m.id === message.id) ? prev : mergeMessages(prev, [message]),
      );
      if (message.sender === 'ai') {
        window.dispatchEvent(new CustomEvent(AI_MESSAGE_EVENT));
      }
    },
    events: {
      clear: () => setMessages([]),
    },
  });

  // Load history on mount and again after every (re)connect to fill any gap.
  useEffect(() => {
    let cancelled = false;
    getMessages()
      .then(({ messages: history }) => {
        if (!cancelled) setMessages((prev) => mergeMessages(prev, history));
      })
      .catch((err: unknown) => console.error('Failed to load Discord messages:', err))
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [connected]);

  const scrollToBottom = useCallback((behavior: ScrollBehavior) => {
    const el = listRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior });
  }, []);

  // Follow new messages only when the user is already near the bottom.
  useLayoutEffect(() => {
    const grew = messages.length > prevCountRef.current;
    const initial = prevCountRef.current === 0;
    prevCountRef.current = messages.length;
    if (!grew) return;
    if (nearBottomRef.current) {
      scrollToBottom(initial ? 'auto' : 'smooth');
    } else {
      setHasNew(true);
    }
  }, [messages, scrollToBottom]);

  const handleScroll = () => {
    const el = listRef.current;
    if (!el) return;
    const near = el.scrollHeight - el.scrollTop - el.clientHeight < NEAR_BOTTOM_PX;
    nearBottomRef.current = near;
    if (near) setHasNew(false);
  };

  const handleClear = async () => {
    if (!confirm('Clear all messages from the monitor? (This will not affect the Discord conversation)')) {
      return;
    }
    try {
      await clearMessages();
      setMessages([]);
      setError(null);
    } catch (err) {
      setError(`Failed to clear messages: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  return (
    <main className="chat">
      <header className="chat-header">
        <img className="chat-logo" src="/metis-avatar.png" alt="" />
        <div className="chat-title">
          <h1>Metis</h1>
          <StatusBar />
        </div>
        <button type="button" className="chat-clear" onClick={handleClear} title="Clear messages">
          <Icon name="trash" size={15} /> Clear
        </button>
      </header>

      {error && (
        <div className="chat-error" role="alert">
          {error}
        </div>
      )}

      <div className="chat-messages-wrap">
        <div className="chat-messages" ref={listRef} onScroll={handleScroll}>
          {isLoading ? (
            <div className="chat-empty">Loading messages…</div>
          ) : messages.length === 0 ? (
            <div className="chat-empty">Waiting for Discord messages…</div>
          ) : (
            messages.map((message) => <MessageItem key={message.id} message={message} />)
          )}
        </div>
        {hasNew && (
          <button
            type="button"
            className="chat-new"
            onClick={() => {
              scrollToBottom('smooth');
              setHasNew(false);
            }}
          >
            <Icon name="arrowDown" size={14} /> New messages
          </button>
        )}
      </div>

      <footer className="chat-footer">
        <span>Read-only monitor · messages come from Discord</span>
        <span className={`chat-live ${connected ? 'on' : 'off'}`}>
          <span className="chat-live-dot" aria-hidden="true" />
          {connected ? 'Live' : 'Reconnecting…'}
        </span>
      </footer>
    </main>
  );
}

export default ChatInterface;
