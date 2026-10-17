import { useEffect, useState } from 'react';
import { getStatus } from '../api';
import type { BackendStatus } from '../api';
import Icon from './Icon';
import './StatusBar.css';

const POLL_INTERVAL_MS = 10_000;

function Dot({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className="status-item">
      <span className={`status-dot ${ok ? 'ok' : 'bad'}`} aria-hidden="true" />
      {label}
    </span>
  );
}

function StatusBar() {
  const [status, setStatus] = useState<BackendStatus | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const next = await getStatus();
        if (cancelled) return;
        setStatus(next);
        setOffline(false);
      } catch {
        if (!cancelled) setOffline(true);
      }
    };
    poll();
    const timer = window.setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  if (offline) {
    return (
      <div className="status-bar">
        <span className="status-item offline">
          <span className="status-dot bad" aria-hidden="true" />
          Backend offline
        </span>
      </div>
    );
  }

  if (!status) {
    return <div className="status-bar muted">Connecting…</div>;
  }

  return (
    <div className="status-bar">
      <span className="status-bot">{status.bot_user ?? 'Metis'}</span>
      {status.personality && <span className="status-pill">{status.personality}</span>}
      <Dot ok={status.llm_ready} label={status.llm_ready ? 'LLM ready' : 'LLM loading'} />
      <Dot
        ok={status.discord_connected}
        label={status.discord_connected ? 'Discord connected' : 'Discord offline'}
      />
      {status.voice_channels.length > 0 && (
        <span className="status-pill voice">
          <Icon name="mic" size={12} /> In voice: {status.voice_channels.join(', ')}
        </span>
      )}
    </div>
  );
}

export default StatusBar;
