import { useCallback, useEffect, useState } from 'react';
import { clearRag, getRagStats, reindexRag } from '../api';
import type { RagRetrieval, RagStats } from '../api';
import './RagPanel.css';

interface Notice {
  text: string;
  type: 'success' | 'error';
}

const fileName = (path: string) => path.split(/[\\/]/).pop() || path;

function RagPanel({ retrieval }: { retrieval: RagRetrieval | null }) {
  const [stats, setStats] = useState<RagStats | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<Notice | null>(null);

  const fetchStats = useCallback(async () => {
    try {
      setStats(await getRagStats());
    } catch (err) {
      console.error('Failed to fetch RAG stats:', err);
    }
  }, []);

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(null), 5000);
    return () => window.clearTimeout(timer);
  }, [notice]);

  const run = async (
    action: () => Promise<string>,
    failLabel: string,
    confirmText?: string,
  ) => {
    if (confirmText && !confirm(confirmText)) return;
    setBusy(true);
    setNotice(null);
    try {
      setNotice({ text: `✓ ${await action()}`, type: 'success' });
      await fetchStats();
    } catch (err) {
      setNotice({
        text: `✗ ${failLabel}: ${err instanceof Error ? err.message : String(err)}`,
        type: 'error',
      });
    } finally {
      setBusy(false);
    }
  };

  const handleReindex = () =>
    run(async () => `Reindexed ${(await reindexRag(false)).total_chunks} chunks`, 'Reindex failed');

  const handleFullReindex = () =>
    run(
      async () => `Full reindex complete: ${(await reindexRag(true)).total_chunks} chunks`,
      'Full reindex failed',
      'Clear and rebuild all embeddings?',
    );

  const handleClearAll = () =>
    run(
      async () => `Cleared ${(await clearRag()).deleted_count} chunks`,
      'Clear failed',
      'Delete all embeddings? This will require a full reindex.',
    );

  const queries = retrieval?.queries ?? [];

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Knowledge base</h2>
        {stats && (
          <span className={`badge ${stats.enabled ? 'ok' : 'bad'}`}>
            {stats.enabled ? 'Enabled' : 'Disabled'}
          </span>
        )}
      </div>

      <div className="rag-stats">
        <div className="rag-stat">
          <span className="rag-stat-label">Chunks</span>
          <span className="rag-stat-value">{stats?.total_chunks ?? '–'}</span>
        </div>
        <div className="rag-stat">
          <span className="rag-stat-label">Files</span>
          <span className="rag-stat-value">{stats?.total_files ?? '–'}</span>
        </div>
        <div className="rag-stat wide">
          <span className="rag-stat-label">Collection</span>
          <span className="rag-stat-small">{stats?.collection_name ?? '–'}</span>
        </div>
      </div>

      <div className="rag-actions">
        <button type="button" className="rag-button primary" onClick={handleReindex} disabled={busy}>
          {busy ? 'Working…' : '🔄 Check for changes'}
        </button>
        <button type="button" className="rag-button" onClick={handleFullReindex} disabled={busy}>
          🔃 Full reindex
        </button>
        <button type="button" className="rag-button danger" onClick={handleClearAll} disabled={busy}>
          🗑️ Clear all
        </button>
      </div>

      {notice && <div className={`rag-notice ${notice.type}`}>{notice.text}</div>}

      <h3 className="panel-section-title">Last retrieval</h3>
      {!retrieval || (queries.length === 0 && retrieval.results.length === 0 && !retrieval.message) ? (
        <div className="panel-empty">No retrieval yet</div>
      ) : (
        <div className="rag-retrieval">
          {retrieval.original_query && (
            <div className="rag-query-box">
              <div className="rag-query-label">Question</div>
              <div>{retrieval.original_query}</div>
            </div>
          )}
          {queries.length > 0 && (
            <div className="rag-query-box">
              <div className="rag-query-label">Generated queries</div>
              <ol className="rag-query-list">
                {queries.map((q, i) => (
                  <li key={i}>{q}</li>
                ))}
              </ol>
            </div>
          )}
          {retrieval.message && <div className="rag-message">{retrieval.message}</div>}

          {retrieval.results.map((result, i) => (
            <div key={i} className={`rag-result ${result.used ? 'used' : 'rejected'}`}>
              <div className="rag-result-head">
                <span className="rag-result-status">{result.used ? '✓' : '✗'}</span>
                <span className="rag-result-file" title={result.source_file}>
                  {fileName(result.source_file)}
                </span>
              </div>
              <div className="rag-result-meta">
                <span>dist {result.distance.toFixed(4)}</span>
                {result.rerank_score != null && (
                  <span className="rag-rerank">rerank {result.rerank_score.toFixed(4)}</span>
                )}
                <span>chunk #{result.chunk_index}</span>
              </div>
              {!result.used && result.rejection_reason && (
                <div className="rag-rejection">Rejected: {result.rejection_reason}</div>
              )}
              <div className="rag-result-preview">{result.text_preview}…</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default RagPanel;
