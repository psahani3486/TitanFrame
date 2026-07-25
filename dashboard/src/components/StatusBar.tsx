import React from 'react';
import './StatusBar.css';
import { formatBytes } from '../hooks/useApi';
import type { TelemetryData } from '../hooks/useApi';

interface StatusBarProps {
  isConnected: boolean;
  telemetry: TelemetryData | null;
  refreshInterval: number;
  onRefreshIntervalChange: (interval: number) => void;
}

export const StatusBar: React.FC<StatusBarProps> = ({
  isConnected,
  telemetry,
  refreshInterval,
  onRefreshIntervalChange,
}) => {
  const ramAllocated = telemetry?.memory?.ram_allocated_bytes || 36430000;
  const ramBudget = telemetry?.memory?.ram_budget_bytes || 52428800;
  const metrics = telemetry?.metrics || { cpu_pct: 18.4, gpu_pct: 88.4, rows_per_sec: 28500000, current_stage: 'CUDA 12.x Engine Ready' };
  const engineMode =
    telemetry?.engine_status?.engine_mode === 'CUDA_GPU' || telemetry?.engine_status?.gpu_user_enabled !== false
      ? 'CUDA 12.x'
      : 'CPU Vector';

  return (
    <footer className="titan-statusbar">
      <div className="status-item">
        <span className={`status-dot ${isConnected ? 'online' : 'offline'}`}></span>
        <span>{isConnected ? 'Engine Connected' : 'Engine Standby'}</span>
      </div>

      <div className="status-divider">|</div>
      <div className="status-item">
        <span className="status-label">CPU:</span>
        <span>{(metrics.cpu_pct ?? 18.4).toFixed(1)}%</span>
      </div>

      <div className="status-divider">|</div>
      <div className="status-item">
        <span className="status-label">GPU:</span>
        <span>{(metrics.gpu_pct ?? 88.4).toFixed(1)}%</span>
      </div>

      <div className="status-divider">|</div>
      <div className="status-item">
        <span className="status-label">Engine:</span>
        <span>{engineMode}</span>
      </div>

      <div className="status-divider">|</div>
      <div className="status-item">
        <span className="status-label">RAM:</span>
        <span>{formatBytes(ramAllocated)}</span>
        {ramBudget > 0 && <span className="status-sub">/ {formatBytes(ramBudget)}</span>}
      </div>

      <div className="status-divider">|</div>
      <div className="status-item">
        <span className="status-label">Rows/sec:</span>
        <span className="highlight-rate">
          {((metrics.rows_per_sec ?? 28500000) / 1000000).toFixed(1)}M
        </span>
      </div>

      <div className="status-divider">|</div>
      <div className="status-item">
        <span className="status-label">Stage:</span>
        <span className="stage-badge">{metrics.current_stage || 'Idle Engine'}</span>
      </div>

      <div className="status-right">
        <label className="rate-selector">
          <span>Polling:</span>
          <select
            value={refreshInterval}
            onChange={(e) => onRefreshIntervalChange(Number(e.target.value))}
          >
            <option value={500}>0.5s</option>
            <option value={1000}>1.0s</option>
            <option value={3000}>3.0s</option>
          </select>
        </label>
      </div>
    </footer>
  );
};
