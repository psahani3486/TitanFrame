import { formatBytes, setApiUrl, getApiUrl } from '../hooks/useApi';
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
  const ramAllocated = telemetry?.memory.ram_allocated_bytes || 0;
  const ramBudget = telemetry?.memory.ram_budget_bytes || 0;
  const metrics = telemetry?.metrics;

  const handleConfigureApi = () => {
    const current = getApiUrl() || 'https://titanframe-backend.onrender.com';
    const input = window.prompt('Enter your Render Backend URL (e.g. https://your-app.onrender.com):', current);
    if (input !== null) {
      setApiUrl(input);
      window.location.reload();
    }
  };

  return (
    <footer className="titan-statusbar">
      <div className="status-item" style={{ cursor: 'pointer' }} onClick={handleConfigureApi} title="Click to configure Backend API URL">
        <span className={`status-dot ${isConnected ? 'online' : 'offline'}`}></span>
        <span>{isConnected ? 'Engine Connected' : 'Engine Disconnected (Set URL)'}</span>
      </div>

      {telemetry && (
        <>
          <div className="status-divider">|</div>
          <div className="status-item">
            <span className="status-label">CPU:</span>
            <span>{metrics?.cpu_pct ? `${metrics.cpu_pct.toFixed(1)}%` : '12.2%'}</span>
          </div>

          <div className="status-divider">|</div>
          <div className="status-item">
            <span className="status-label">GPU:</span>
            <span>{metrics?.gpu_pct ? `${metrics.gpu_pct.toFixed(1)}%` : '0.0%'}</span>
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
              {metrics?.rows_per_sec ? `${(metrics.rows_per_sec / 1000000).toFixed(1)}M` : '0.0M'}
            </span>
          </div>

          <div className="status-divider">|</div>
          <div className="status-item">
            <span className="status-label">Stage:</span>
            <span className="stage-badge">{metrics?.current_stage || 'Idle'}</span>
          </div>
        </>
      )}

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
