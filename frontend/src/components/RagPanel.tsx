import { useState, useEffect } from 'react';
import './RagPanel.css';

interface RagStats {
  total_chunks: number;
  total_files: number;
  collection_name: string;
  enabled: boolean;
}

interface RagResult {
  source_file: string;
  distance: number;
  rerank_score?: number;  // Optional rerank score
  text_preview: string;
  chunk_index: number;
  used: boolean;
  rejection_reason?: string;
}

interface RagRetrievalData {
  query?: string;
  queries?: string[];
  original_query?: string;
  results: RagResult[];
}

function RagPanel() {
  const [stats, setStats] = useState<RagStats | null>(null);
  const [ragData, setRagData] = useState<RagRetrievalData>({ results: [] });
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState<{ text: string; type: 'success' | 'error' | 'info' } | null>(null);
  const API_URL = (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000';

  const fetchStats = async () => {
    try {
      const resp = await fetch(`${API_URL}/rag/stats`);
      const data = await resp.json();
      setStats(data);
    } catch (error) {
      console.error('Failed to fetch RAG stats:', error);
    }
  };

  const fetchRagResults = async () => {
    try {
      const resp = await fetch(`${API_URL}/rag/last-retrieval`);
      const data = await resp.json();
      // Handle both old array format and new object format
      if (Array.isArray(data.results)) {
        setRagData({
          query: data.results.query,
          original_query: data.results.original_query,
          results: data.results.results || []
        });
      } else if (data.results) {
        setRagData(data.results);
      } else {
        setRagData({ results: [] });
      }
    } catch (error) {
      console.error('Failed to fetch RAG results:', error);
    }
  };

  useEffect(() => {
    // Initial fetch
    fetchStats();
    fetchRagResults();

    // Listen for RAG retrieval completion (shows embeddings immediately)
    const handleRagRetrieval = () => {
      fetchRagResults();
    };

    // Listen for message completion events (updates stats)
    const handleMessageComplete = () => {
      fetchStats();
      fetchRagResults();
    };

    window.addEventListener('ragRetrievalComplete', handleRagRetrieval);
    window.addEventListener('messageComplete', handleMessageComplete);
    
    return () => {
      window.removeEventListener('ragRetrievalComplete', handleRagRetrieval);
      window.removeEventListener('messageComplete', handleMessageComplete);
    };
  }, []);

  const handleReindex = async () => {
    setIsLoading(true);
    setMessage(null);
    try {
      const resp = await fetch(`${API_URL}/rag/reindex?clear_existing=false`, {
        method: 'POST',
      });
      const data = await resp.json();
      setMessage({
        text: `✓ Reindexed ${data.total_chunks} chunks`,
        type: 'success',
      });
      await fetchStats();
    } catch (error) {
      setMessage({
        text: `✗ Reindex failed: ${error}`,
        type: 'error',
      });
    } finally {
      setIsLoading(false);
      setTimeout(() => setMessage(null), 5000);
    }
  };

  const handleClearAll = async () => {
    if (!confirm('Are you sure you want to delete all embeddings? This will require a full reindex.')) {
      return;
    }

    setIsLoading(true);
    setMessage(null);
    try {
      const resp = await fetch(`${API_URL}/rag/clear`, {
        method: 'POST',
      });
      const data = await resp.json();
      setMessage({
        text: `✓ Cleared ${data.deleted_count} chunks`,
        type: 'success',
      });
      await fetchStats();
    } catch (error) {
      setMessage({
        text: `✗ Clear failed: ${error}`,
        type: 'error',
      });
    } finally {
      setIsLoading(false);
      setTimeout(() => setMessage(null), 5000);
    }
  };

  const handleFullReindex = async () => {
    if (!confirm('Are you sure you want to clear and rebuild all embeddings?')) {
      return;
    }

    setIsLoading(true);
    setMessage(null);
    try {
      const resp = await fetch(`${API_URL}/rag/reindex?clear_existing=true`, {
        method: 'POST',
      });
      const data = await resp.json();
      setMessage({
        text: `✓ Full reindex complete: ${data.total_chunks} chunks`,
        type: 'success',
      });
      await fetchStats();
    } catch (error) {
      setMessage({
        text: `✗ Full reindex failed: ${error}`,
        type: 'error',
      });
    } finally {
      setIsLoading(false);
      setTimeout(() => setMessage(null), 5000);
    }
  };

  return (
    <div className="rag-panel">
      <div className="rag-header">
        <h2>RAG System</h2>
        {stats?.enabled ? (
          <span className="status-badge enabled">Enabled</span>
        ) : (
          <span className="status-badge disabled">Disabled</span>
        )}
      </div>

      <div className="rag-stats">
        <div className="stat-item">
          <span className="stat-label">Chunks</span>
          <span className="stat-value">{stats?.total_chunks ?? '–'}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Files</span>
          <span className="stat-value">{stats?.total_files ?? '–'}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Collection</span>
          <span className="stat-value-small">{stats?.collection_name ?? '–'}</span>
        </div>
      </div>

      <div className="rag-actions">
        <button
          className="rag-button primary"
          onClick={handleReindex}
          disabled={isLoading}
        >
          <span>🔄</span>
          <span>{isLoading ? 'Processing...' : 'Check for Changes'}</span>
        </button>

        <button
          className="rag-button secondary"
          onClick={handleFullReindex}
          disabled={isLoading}
        >
          <span>🔃</span>
          <span>Full Reindex</span>
        </button>

        <button
          className="rag-button danger"
          onClick={handleClearAll}
          disabled={isLoading}
        >
          <span>🗑️</span>
          <span>Clear All</span>
        </button>
      </div>

      {message && (
        <div className={`rag-message ${message.type}`}>
          {message.text}
        </div>
      )}

      {( (ragData.queries && ragData.queries.length > 0) || ragData.results.length > 0 ) && (
        <div className="rag-results">
          <h3>Last Retrieval</h3>
          
          {/* Display the generated queries if available */}
          {ragData.queries && ragData.queries.length > 0 ? (
            <div className="query-display">
              <div className="query-label">Generated Queries:</div>
              <div className="query-text">
                {ragData.queries.map((q, i) => (
                  <div key={i} className="query-line">{i + 1}. {q}</div>
                ))}
              </div>
            </div>
          ) : ragData.query ? (
            <div className="query-display">
              <div className="query-label">Generated Query:</div>
              <div className="query-text">{ragData.query}</div>
            </div>
          ) : null}
          
          <div className="results-list">
            {ragData.results.map((result, idx) => (
              <div key={idx} className={`result-item ${result.used ? 'used' : 'rejected'}`}>
                <div className="result-header">
                  <span className="result-status">{result.used ? '✓' : '✗'}</span>
                  <span className="result-file">{result.source_file.split('\\').pop()}</span>
                </div>
                <div className="result-meta">
                  <span>Distance: {result.distance.toFixed(4)}</span>
                  {result.rerank_score !== undefined && result.rerank_score !== null && (
                    <span className="rerank-score">Rerank: {result.rerank_score.toFixed(4)}</span>
                  )}
                  <span>Chunk #{result.chunk_index}</span>
                  {!result.used && result.rejection_reason && (
                    <span className="rejection-reason">Rejected: {result.rejection_reason}</span>
                  )}
                </div>
                <div className="result-preview">{result.text_preview}...</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default RagPanel;
