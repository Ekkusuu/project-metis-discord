import { useState } from 'react';
import { API_URL } from '../api';
import type { RagRetrieval, RagStreamPayload, ToolExecution } from '../api';
import { useEventSource } from '../hooks/useEventSource';
import RagPanel from './RagPanel';
import ToolsPanel from './ToolsPanel';
import ChatContext from './ChatContext';
import './Sidebar.css';

type Tab = 'rag' | 'tools' | 'context';

const TABS: Array<{ id: Tab; label: string }> = [
  { id: 'rag', label: '📚 Knowledge' },
  { id: 'tools', label: '🛠️ Tools' },
  { id: 'context', label: '💬 Context' },
];

const isRetrieval = (v: unknown): v is RagRetrieval =>
  typeof v === 'object' && v !== null && Array.isArray((v as RagRetrieval).results);

const isToolExecution = (v: unknown): v is ToolExecution =>
  typeof v === 'object' && v !== null && Array.isArray((v as ToolExecution).tool_calls);

function Sidebar() {
  const [activeTab, setActiveTab] = useState<Tab>('rag');
  const [retrieval, setRetrieval] = useState<RagRetrieval | null>(null);
  const [tools, setTools] = useState<ToolExecution | null>(null);

  useEventSource(`${API_URL}/rag/stream`, {
    onMessage: (event) => {
      try {
        const data = JSON.parse(event.data) as Partial<RagStreamPayload>;
        setRetrieval(isRetrieval(data.results) ? data.results : null);
        setTools(isToolExecution(data.tools) ? data.tools : null);
      } catch (err) {
        console.error('Failed to parse RAG stream update:', err);
      }
    },
  });

  return (
    <aside className="sidebar">
      <div className="sidebar-tabs" role="tablist">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={activeTab === tab.id}
            className={`sidebar-tab ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="sidebar-content">
        {activeTab === 'rag' && <RagPanel retrieval={retrieval} />}
        {activeTab === 'tools' && <ToolsPanel tools={tools} />}
        {activeTab === 'context' && <ChatContext />}
      </div>
    </aside>
  );
}

export default Sidebar;
