import { useEffect, useRef, useState } from 'react';

type Listener = (event: MessageEvent<string>) => void;

export interface EventSourceHandlers {
  /** Handler for default (unnamed) `message` events. */
  onMessage?: Listener;
  /** Handlers for named events, keyed by event name. The set of names is fixed per connection. */
  events?: Record<string, Listener>;
}

const RECONNECT_DELAY_MS = 3000;

/**
 * Opens a single EventSource for `url`. Transient errors are left to the browser's
 * built-in reconnect (which preserves Last-Event-ID); only if the connection is
 * permanently CLOSED do we schedule one new attempt after 3s.
 * Returns whether the stream is currently open.
 */
export function useEventSource(url: string, handlers: EventSourceHandlers): boolean {
  const [connected, setConnected] = useState(false);
  const handlersRef = useRef(handlers);

  useEffect(() => {
    handlersRef.current = handlers;
  });

  const eventNames = Object.keys(handlers.events ?? {}).join(',');

  useEffect(() => {
    const names = eventNames ? eventNames.split(',') : [];
    let source: EventSource | null = null;
    let timer: number | undefined;
    let disposed = false;

    const connect = () => {
      const es = new EventSource(url);
      source = es;
      es.onopen = () => setConnected(true);
      es.onmessage = (event) => handlersRef.current.onMessage?.(event);
      for (const name of names) {
        es.addEventListener(name, (event) =>
          handlersRef.current.events?.[name]?.(event as MessageEvent<string>),
        );
      }
      es.onerror = () => {
        setConnected(false);
        if (es.readyState === EventSource.CLOSED && !disposed) {
          timer = window.setTimeout(connect, RECONNECT_DELAY_MS);
        }
      };
    };

    connect();

    return () => {
      disposed = true;
      window.clearTimeout(timer);
      source?.close();
    };
  }, [url, eventNames]);

  return connected;
}
