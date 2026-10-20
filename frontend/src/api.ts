// The built dashboard is served by the backend itself, so it talks to its own origin. The Vite dev
// server (`python metis.py --dev`) runs elsewhere and gets VITE_API_URL from the launcher.
export const API_URL: string =
  import.meta.env.VITE_API_URL || (import.meta.env.DEV ? 'http://127.0.0.1:8730' : '');

/** Window event dispatched whenever an AI message arrives on the Discord stream. */
export const AI_MESSAGE_EVENT = 'metis:ai-message';

export interface DiscordMessage {
  id: string;
  seq: number;
  text: string;
  sender: 'user' | 'ai';
  timestamp: string;
  discord_user?: string;
  channel_name?: string;
  mode?: 'text' | 'voice';
  tokensPerSecond?: number;
}

export interface BackendStatus {
  llm_ready: boolean;
  discord_connected: boolean;
  bot_user: string | null;
  personality: string;
  rag_enabled: boolean;
  tools_enabled: boolean;
  voice_channels: string[];
  people_known?: number;
}

export interface RagStats {
  total_chunks: number;
  total_files: number;
  collection_name: string;
  enabled: boolean;
}

export interface RagResult {
  source_file: string;
  distance: number;
  rerank_score?: number | null;
  text_preview: string;
  chunk_index: number;
  used: boolean;
  rejection_reason?: string;
}

export interface RagRetrieval {
  queries?: string[];
  original_query?: string;
  message?: string;
  results: RagResult[];
}

export interface ToolCall {
  tool: string;
  arguments: Record<string, unknown>;
}

export interface ToolResult {
  tool: string;
  success: boolean;
  execution_time: number;
  error?: string | null;
}

export interface ToolExecution {
  tool_calls: ToolCall[];
  results: ToolResult[];
  message?: string;
  timings?: Record<string, number>;
}

export interface RagStreamPayload {
  results: RagRetrieval | Record<string, never>;
  tools: ToolExecution | Record<string, never>;
}

export interface ContextMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
  timestamp?: string;
}

export interface Fact {
  id: string;
  text: string;
  added: string;
}

export interface Person {
  id: string;
  name: string;
  aliases: string[];
  first_seen: string;
  last_seen: string;
  message_count: number;
  facts: Fact[];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, init);
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === 'string') detail = body.detail;
    } catch {
      // body was not JSON; keep the status text
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export const getMessages = () => request<{ messages: DiscordMessage[] }>('/discord/messages');
export const clearMessages = () => request<unknown>('/discord/clear', { method: 'POST' });
export const getStatus = () => request<BackendStatus>('/status');
export const getRagStats = () => request<RagStats>('/rag/stats');
export const reindexRag = (clearExisting: boolean) =>
  request<{ status: string; total_chunks: number; folders: unknown }>(
    `/rag/reindex?clear_existing=${clearExisting}`,
    { method: 'POST' },
  );
export const clearRag = () =>
  request<{ status: string; deleted_count: number }>('/rag/clear', { method: 'POST' });
export const getChatContext = () => request<{ messages: ContextMessage[] }>('/chat/context');

export const getPeople = () => request<{ people: Person[] }>('/memory/people');
export const getNotes = () => request<{ notes: Fact[] }>('/memory/notes');
export const deletePersonFact = (personId: string, factId: string) =>
  request<{ status: string }>(
    `/memory/people/${encodeURIComponent(personId)}/facts/${encodeURIComponent(factId)}`,
    { method: 'DELETE' },
  );
export const forgetPerson = (personId: string) =>
  request<{ status: string }>(`/memory/people/${encodeURIComponent(personId)}`, {
    method: 'DELETE',
  });
export const deleteNote = (factId: string) =>
  request<{ status: string }>(`/memory/notes/${encodeURIComponent(factId)}`, {
    method: 'DELETE',
  });
export const learnNow = () =>
  request<{ status: string; learned: number }>('/memory/learn', { method: 'POST' });

export interface Personality {
  name: string;
  description: string;
  has_voice: boolean;
}

export const getPersonalities = () =>
  request<{ current: string; available: Personality[] }>('/personality');
export const setPersonality = (name: string) =>
  request<{ current: string; saved: boolean }>('/personality', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });
