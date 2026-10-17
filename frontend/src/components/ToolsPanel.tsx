import type { ToolExecution } from '../api';
import Icon from './Icon';
import './ToolsPanel.css';

/** Execution times from the backend are in seconds. */
function formatDuration(seconds: number): string {
  return seconds < 1 ? `${Math.round(seconds * 1000)} ms` : `${seconds.toFixed(2)} s`;
}

function formatValue(value: unknown): string {
  const text = typeof value === 'string' ? JSON.stringify(value) : (JSON.stringify(value) ?? String(value));
  return text.length > 300 ? `${text.slice(0, 300)}…` : text;
}

function ToolsPanel({ tools }: { tools: ToolExecution | null }) {
  const rowCount = tools ? Math.max(tools.tool_calls.length, tools.results.length) : 0;
  const timings = tools?.timings ? Object.entries(tools.timings) : [];

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Tool calls</h2>
      </div>

      {!tools ? (
        <div className="panel-empty">No tool activity yet</div>
      ) : (
        <>
          {rowCount === 0 && (
            <div className="panel-empty">{tools.message ?? 'No tools were used for the last reply'}</div>
          )}

          {Array.from({ length: rowCount }, (_, i) => {
            const call = tools.tool_calls[i];
            const result = tools.results[i];
            const name = call?.tool ?? result?.tool ?? 'unknown';
            const args = call ? Object.entries(call.arguments) : [];
            const state = result ? (result.success ? 'success' : 'failure') : 'pending';
            return (
              <div key={i} className={`tool-card ${state}`}>
                <div className="tool-head">
                  <span className="tool-name">{name}</span>
                  {result ? (
                    <span className={`badge ${result.success ? 'ok' : 'bad'}`}>
                      <Icon name={result.success ? 'check' : 'x'} size={12} />
                      {result.success ? 'success' : 'failed'}
                    </span>
                  ) : (
                    <span className="tool-pending">
                      <Icon name="spinner" size={12} /> running…
                    </span>
                  )}
                </div>
                {args.length > 0 && (
                  <dl className="tool-args">
                    {args.map(([key, value]) => (
                      <div key={key} className="tool-arg">
                        <dt>{key}:</dt>
                        <dd>{formatValue(value)}</dd>
                      </div>
                    ))}
                  </dl>
                )}
                {result && <div className="tool-time">{formatDuration(result.execution_time)}</div>}
                {result?.error && <div className="tool-error">{result.error}</div>}
              </div>
            );
          })}

          {rowCount > 0 && tools.message && <div className="tool-message">{tools.message}</div>}

          {timings.length > 0 && (
            <>
              <h3 className="panel-section-title">Timings</h3>
              <dl className="tool-timings">
                {timings.map(([key, value]) => (
                  <div key={key} className="tool-timing">
                    <dt>{key}</dt>
                    <dd>{formatDuration(value)}</dd>
                  </div>
                ))}
              </dl>
            </>
          )}
        </>
      )}
    </div>
  );
}

export default ToolsPanel;
