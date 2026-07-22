import { useState, useEffect } from 'react';
import './QueryHistory.css';
import { api } from '../hooks/useApi';
import type { QueryStat, TelemetryData } from '../hooks/useApi';

interface QueryHistoryProps {
  telemetry: TelemetryData | null;
}

export const QueryHistory: React.FC<QueryHistoryProps> = ({ telemetry }) => {
  const [queries, setQueries] = useState<QueryStat[]>([]);
  const [selectedQueryId, setSelectedQueryId] = useState<string | null>(null);
  const [selectedLogs, setSelectedLogs] = useState<string[]>([]);
  const [searchTerm, setSearchTerm] = useState<string>('');

  useEffect(() => {
    if (telemetry?.queries) {
      setQueries(telemetry.queries);
    }
  }, [telemetry]);

  const handleInspectLogs = async (queryId: string) => {
    setSelectedQueryId(queryId);
    try {
      const logs = await api.getQueryLogs(queryId);
      setSelectedLogs(logs);
    } catch (err) {
      setSelectedLogs(['Failed to load query logs']);
    }
  };

  const filteredQueries = queries.filter(
    (q: QueryStat) =>
      q.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      q.status.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="page-container history-page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Query History & Execution Logs</h1>
          <p className="page-subtitle">
            Complete audit trail of streaming queries, execution times, processed chunk counts, and log tracebacks.
          </p>
        </div>
        <div className="search-box">
          <input
            type="text"
            placeholder="Search query ID or status..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
      </div>

      <div className="history-grid">
        {}
        <div className="glass-panel table-panel">
          <div className="table-responsive">
            <table className="history-table">
              <thead>
                <tr>
                  <th>Query ID</th>
                  <th>Status</th>
                  <th>Chunks</th>
                  <th>Duration</th>
                  <th>Started At</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredQueries.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="empty-td">
                      No query history found.
                    </td>
                  </tr>
                ) : (
                  filteredQueries.slice().reverse().map((q: QueryStat) => (
                    <tr
                      key={q.id}
                      className={selectedQueryId === q.id ? 'active-row' : ''}
                    >
                      <td className="qid-td"><code>{q.id}</code></td>
                      <td>
                        <span className={`status-badge ${q.status.toLowerCase()}`}>
                          {q.status}
                        </span>
                      </td>
                      <td>{q.chunks_processed} chunks</td>
                      <td>{q.duration_sec ? `${q.duration_sec}s` : 'Running...'}</td>
                      <td className="time-td">
                        {new Date(q.start_time * 1000).toLocaleTimeString()}
                      </td>
                      <td>
                        <button
                          className="btn-sm"
                          onClick={() => handleInspectLogs(q.id)}
                        >
                          View Logs
                        </button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        {}
        {selectedQueryId && (
          <div className="glass-panel logs-panel">
            <div className="panel-header">
              <h3>Logs for Query: <code>{selectedQueryId}</code></h3>
              <button className="close-btn" onClick={() => setSelectedQueryId(null)}>
                Close [X]
              </button>
            </div>
            <div className="terminal-box">
              {selectedLogs.map((log: string, idx: number) => (
                <div key={idx} className="terminal-line">
                  {log}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
