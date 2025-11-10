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
  text_preview: string;
  chunk_index: number;
  used: boolean;
}

function RagPanel() {
  const [stats, setStats] = useState<RagStats | null>(null);
  const [ragResults, setRagResults] = useState<RagResult[]>([]);
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
      setRagResults(data.results || []);
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

      {ragResults.length > 0 && (
        <div className="rag-results">
          <h3>Last Retrieval</h3>
          <div className="results-list">
            {ragResults.map((result, idx) => (
              <div key={idx} className={`result-item ${result.used ? 'used' : 'rejected'}`}>
                <div className="result-header">
                  <span className="result-status">{result.used ? '✓' : '✗'}</span>
                  <span className="result-file">{result.source_file.split('\\').pop()}</span>
                </div>
                <div className="result-meta">
                  <span>Distance: {result.distance.toFixed(4)}</span>
                  <span>Chunk #{result.chunk_index}</span>
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
