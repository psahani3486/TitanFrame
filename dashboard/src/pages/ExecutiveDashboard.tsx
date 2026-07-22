import React from 'react';
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import './ExecutiveDashboard.css';
import { formatBytes } from '../hooks/useApi';
import type { TelemetryData, DatasetInfo } from '../hooks/useApi';
import { LivePipelineAnimation } from '../components/LivePipelineAnimation';

interface ExecutiveDashboardProps {
  telemetry: TelemetryData | null;
  datasets: DatasetInfo[];
  onNavigate: (page: any) => void;
}

export const ExecutiveDashboard: React.FC<ExecutiveDashboardProps> = ({
  telemetry,
  datasets,
  onNavigate,
}) => {
  const memory = telemetry?.memory;
  const queries = telemetry?.queries || [];
  const metrics = telemetry?.metrics;

  const completedQueries = queries.filter((q) => q.status === 'COMPLETED');
  const totalQueries = queries.length;
  const avgDuration =
    completedQueries.length > 0
      ? (
          completedQueries.reduce((acc, q) => acc + (q.duration_sec || 0), 0) /
          completedQueries.length
        ).toFixed(2)
      : '0.00';

  const ramAllocated = memory?.ram_allocated_bytes || 0;
  const ramBudget = memory?.ram_budget_bytes || 0;
  const ramPct = ramBudget > 0 ? Math.min(100, (ramAllocated / ramBudget) * 100) : 0;

  const ramTimeline = memory?.ram_timeline || [
    { timestamp: '10:00:00', ram_mb: 45, throughput: 2.1 },
    { timestamp: '10:00:05', ram_mb: 72, throughput: 3.4 },
    { timestamp: '10:00:10', ram_mb: 110, throughput: 4.2 },
    { timestamp: '10:00:15', ram_mb: 85, throughput: 3.8 },
  ];

  return (
    <div className="page-container executive-page">
      {}
      <div className="page-header">
        <div>
          <h1 className="page-title">Executive Dashboard</h1>
          <p className="page-subtitle">
            Real-time engine telemetry, data pipeline metrics, and system health overview.
          </p>
        </div>
        <div className="header-actions-row">
          <button className="btn btn-accent" onClick={() => onNavigate('sql')}>
            Launch SQL Workspace
          </button>
        </div>
      </div>

      {}
      <LivePipelineAnimation
        activeStageIdx={metrics?.active_pipeline_stage || 0}
        currentStageName={metrics?.current_stage || 'Engine Ready'}
        rowsPerSec={metrics?.rows_per_sec || 0}
      />

      {}
      <div className="kpi-grid">
        <div className="kpi-card violet">
          <div className="kpi-icon-badge">QRY</div>
          <div className="kpi-details">
            <span className="kpi-label">Total Executed Queries</span>
            <span className="kpi-value">{totalQueries}</span>
            <span className="kpi-sub">{completedQueries.length} Completed Successfully</span>
          </div>
        </div>

        <div className="kpi-card cyan">
          <div className="kpi-icon-badge">SPD</div>
          <div className="kpi-details">
            <span className="kpi-label">Avg Execution Speed</span>
            <span className="kpi-value">{avgDuration}s</span>
            <span className="kpi-sub">Out-of-Core Vector Engine</span>
          </div>
        </div>

        <div className="kpi-card gold">
          <div className="kpi-icon-badge">DATA</div>
          <div className="kpi-details">
            <span className="kpi-label">Active Datasets</span>
            <span className="kpi-value">{datasets.length}</span>
            <span className="kpi-sub">
              {datasets.reduce((acc, d) => acc + d.size_bytes, 0) > 0
                ? formatBytes(datasets.reduce((acc, d) => acc + d.size_bytes, 0))
                : 'Workspace Data'}
            </span>
          </div>
        </div>

        <div className="kpi-card emerald">
          <div className="kpi-icon-badge">NVME</div>
          <div className="kpi-details">
            <span className="kpi-label">NVMe Spill Events</span>
            <span className="kpi-value">{memory?.spill_events_count || 0}</span>
            <span className="kpi-sub">Arrow IPC Streaming</span>
          </div>
        </div>
      </div>

      {}
      <div className="grid-2-col" style={{ marginBottom: '1.5rem' }}>
        <div className="glass-panel">
          <div className="panel-header">
            <h3>Live RAM Allocation Timeline (MB)</h3>
            <span className="panel-tag">{ramPct.toFixed(1)}% Allocated</span>
          </div>
          <div style={{ width: '100%', height: '220px' }}>
            <ResponsiveContainer>
              <AreaChart data={ramTimeline}>
                <XAxis dataKey="timestamp" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={11} unit="MB" />
                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)' }} />
                <Area type="monotone" dataKey="ram_mb" stroke="#38bdf8" fill="rgba(56,189,248,0.2)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="glass-panel">
          <div className="panel-header">
            <h3>Engine Throughput Rate (M Rows / Sec)</h3>
            <span className="panel-tag gold">
              {metrics?.rows_per_sec ? (metrics.rows_per_sec / 1000000).toFixed(1) : '0.0'}M Rows/s
            </span>
          </div>
          <div style={{ width: '100%', height: '220px' }}>
            <ResponsiveContainer>
              <LineChart data={ramTimeline}>
                <XAxis dataKey="timestamp" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={11} unit="M" />
                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)' }} />
                <Line type="monotone" dataKey="throughput" stroke="#fbbf24" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {}
      <div className="grid-2-col">
        {}
        <div className="glass-panel">
          <div className="panel-header">
            <h3>Host Memory & NVMe Spilling</h3>
          </div>

          <div className="bar-stat-group">
            <div className="bar-header">
              <span>RAM Allocation ({formatBytes(ramAllocated)})</span>
              <span>Budget: {ramBudget > 0 ? formatBytes(ramBudget) : 'Unlimited'}</span>
            </div>
            <div className="progress-bg">
              <div className="progress-fill ram" style={{ width: `${ramPct}%` }}></div>
            </div>
          </div>

          <div className="bar-stat-group" style={{ marginTop: '1.25rem' }}>
            <div className="bar-header">
              <span>NVMe Spill Storage ({formatBytes(memory?.spill_allocated_bytes || 0)})</span>
              <span>Threshold: {((telemetry?.config?.spill_threshold || 0.85) * 100).toFixed(0)}%</span>
            </div>
            <div className="progress-bg">
              <div
                className="progress-fill spill"
                style={{
                  width: `${Math.min(
                    100,
                    ((memory?.spill_allocated_bytes || 0) /
                      (memory?.spill_budget_bytes || 100 * 1024 * 1024 * 1024)) *
                      100
                  )}%`,
                }}
              ></div>
            </div>
          </div>

          <div className="engine-features-pills" style={{ marginTop: '1.5rem' }}>
            <div className="feature-pill">
              <span className="dot active"></span> Predicate Pushdown: ON
            </div>
            <div className="feature-pill">
              <span className="dot active"></span> Projection Pruning: ON
            </div>
            <div className="feature-pill">
              <span className="dot active"></span> Out-of-Core Spill: ENABLED
            </div>
          </div>
        </div>

        {}
        <div className="glass-panel">
          <div className="panel-header">
            <h3>Recent Pipeline Queries</h3>
            <button className="btn-link" onClick={() => onNavigate('history')}>
              View All History &gt;&gt;
            </button>
          </div>

          {queries.length === 0 ? (
            <div className="empty-panel">
              <p>No queries executed yet. Run analytics in the SQL Workspace or Datasets tab.</p>
            </div>
          ) : (
            <div className="recent-queries-list">
              {queries.slice(-4).reverse().map((q) => (
                <div key={q.id} className="query-row-card">
                  <div className="query-row-info">
                    <span className="query-id">{q.id}</span>
                    <span className="query-sub">Chunks: {q.chunks_processed}</span>
                  </div>
                  <div className="query-row-meta">
                    {q.duration_sec && <span className="query-dur">{q.duration_sec}s</span>}
                    <span className={`status-badge ${q.status.toLowerCase()}`}>{q.status}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {}
      <div className="glass-panel" style={{ marginTop: '1.5rem' }}>
        <div className="panel-header">
          <h3>Workspace Datasets Overview</h3>
          <button className="btn-link" onClick={() => onNavigate('datasets')}>
            Manage Datasets &gt;&gt;
          </button>
        </div>

        <div className="dataset-chips-grid">
          {datasets.map((d) => (
            <div key={d.path} className="dataset-chip-card" onClick={() => onNavigate('datasets')}>
              <span className="chip-icon-badge">{d.name.endsWith('.csv') ? 'CSV' : 'PARQUET'}</span>
              <div className="chip-text">
                <span className="chip-title">{d.name}</span>
                <span className="chip-sub">{d.size_formatted}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
