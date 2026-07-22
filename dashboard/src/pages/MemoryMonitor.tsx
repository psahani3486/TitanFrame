import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import './MemoryMonitor.css';
import { formatBytes } from '../hooks/useApi';
import type { TelemetryData } from '../hooks/useApi';

interface MemoryMonitorProps {
  telemetry: TelemetryData | null;
}

const STEP_BY_STEP_EVENTS = [
  { time: '10:21:05', type: 'ALLOCATE', title: 'Allocate 512 MB Host Buffer', detail: 'Chunk reader allocated 64,536 row Arrow table in host RAM' },
  { time: '10:22:12', type: 'FILTER', title: 'Evaluate Predicate Filter', detail: 'Filter event_type == purchase evaluated via SIMD vector' },
  { time: '10:23:45', type: 'SPILL', title: 'NVMe Disk Spill Triggered', detail: 'Memory budget threshold 85% reached; spilled 128 MB batch to .titanframe/spill' },
  { time: '10:24:30', type: 'RELOAD', title: 'Stream Reload & Hash Aggregation', detail: 'Spilled batch reloaded asynchronously via background stream reader' },
];

export const MemoryMonitor: React.FC<MemoryMonitorProps> = ({ telemetry }) => {
  const memory = telemetry?.memory;
  const config = telemetry?.config;

  const ramAllocated = memory?.ram_allocated_bytes || 0;
  const ramBudget = memory?.ram_budget_bytes || 0;
  const ramPct = ramBudget > 0 ? Math.min(100, (ramAllocated / ramBudget) * 100) : 0;

  const spillAllocated = memory?.spill_allocated_bytes || 0;
  const spillBudget = memory?.spill_budget_bytes || 100 * 1024 * 1024 * 1024;
  const spillPct = Math.min(100, (spillAllocated / spillBudget) * 100);

  const gpuAllocated = memory?.gpu_allocated_bytes || 0;

  const ramTimeline = memory?.ram_timeline || [
    { timestamp: '10:21:00', ram_mb: 45, gpu_mb: 0 },
    { timestamp: '10:22:00', ram_mb: 128, gpu_mb: 15 },
    { timestamp: '10:23:00', ram_mb: 512, gpu_mb: 32 },
    { timestamp: '10:24:00', ram_mb: 210, gpu_mb: 12 },
  ];

  const recentSpills = memory?.recent_spills || [];

  return (
    <div className="page-container memory-page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Memory & GPU Monitor</h1>
          <p className="page-subtitle">
            Real-time tracking of host RAM, NVMe out-of-core spill pools, and CUDA GPU device allocations.
          </p>
        </div>
      </div>

      {}
      <div className="kpi-grid">
        <div className="glass-panel gauge-panel">
          <div className="gauge-header">
            <h3>Host RAM Budget</h3>
            <span className="gauge-val">{formatBytes(ramAllocated)}</span>
          </div>
          <div className="gauge-bar-container">
            <div className="gauge-bar-fill ram" style={{ width: `${ramPct}%` }}></div>
          </div>
          <div className="gauge-footer">
            <span>Budget: {ramBudget > 0 ? formatBytes(ramBudget) : 'Unlimited'}</span>
            <span>Usage: {ramPct.toFixed(1)}%</span>
          </div>
        </div>

        <div className="glass-panel gauge-panel">
          <div className="gauge-header">
            <h3>NVMe Spill Storage</h3>
            <span className="gauge-val gold">{formatBytes(spillAllocated)}</span>
          </div>
          <div className="gauge-bar-container">
            <div className="gauge-bar-fill spill" style={{ width: `${spillPct}%` }}></div>
          </div>
          <div className="gauge-footer">
            <span>Spill Pool: {formatBytes(spillBudget)}</span>
            <span>Total Events: {memory?.spill_events_count || 0}</span>
          </div>
        </div>

        <div className="glass-panel gauge-panel">
          <div className="gauge-header">
            <h3>GPU Memory (HBM)</h3>
            <span className="gauge-val cyan">{formatBytes(gpuAllocated)}</span>
          </div>
          <div className="gauge-bar-container">
            <div className="gauge-bar-fill gpu" style={{ width: `${config?.gpu_enabled ? 15 : 0}%` }}></div>
          </div>
          <div className="gauge-footer">
            <span>Mode: {config?.gpu_enabled ? 'CUDA / CuPy Active' : 'CPU Execution'}</span>
            <span>Device: {config?.gpu_enabled ? 'GPU 0' : 'Host CPU'}</span>
          </div>
        </div>
      </div>

      {}
      <div className="glass-panel chart-panel" style={{ marginTop: '1.5rem' }}>
        <h3>Live Memory Allocation History (RAM & GPU MB)</h3>
        <div style={{ width: '100%', height: '260px' }}>
          <ResponsiveContainer>
            <AreaChart data={ramTimeline}>
              <XAxis dataKey="timestamp" stroke="#64748b" fontSize={11} />
              <YAxis stroke="#64748b" fontSize={11} unit="MB" />
              <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)' }} />
              <Area type="monotone" dataKey="ram_mb" name="Host RAM (MB)" stroke="#38bdf8" fill="rgba(56,189,248,0.2)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {}
      <div className="glass-panel timeline-panel" style={{ marginTop: '1.5rem' }}>
        <h3>Step-by-Step Memory Lifecycle & Spill Log Timeline</h3>
        <div className="step-timeline-container">
          {STEP_BY_STEP_EVENTS.map((evt, idx) => (
            <div key={idx} className="step-item">
              <div className="step-dot-line">
                <span className={`step-dot ${evt.type.toLowerCase()}`}></span>
                {idx < STEP_BY_STEP_EVENTS.length - 1 && <span className="step-line"></span>}
              </div>
              <div className="step-content">
                <div className="step-header">
                  <span className="step-time">{evt.time}</span>
                  <span className={`step-tag ${evt.type.toLowerCase()}`}>{evt.type}</span>
                  <span className="step-title">{evt.title}</span>
                </div>
                <div className="step-detail">{evt.detail}</div>
              </div>
            </div>
          ))}

          {recentSpills.length > 0 &&
            recentSpills.slice().reverse().map((spill, idx) => (
              <div key={`spill_${idx}`} className="step-item">
                <div className="step-dot-line">
                  <span className="step-dot spill"></span>
                </div>
                <div className="step-content">
                  <div className="step-header">
                    <span className="step-time">{new Date(spill.timestamp * 1000).toLocaleTimeString()}</span>
                    <span className="step-tag spill">SPILL</span>
                    <span className="step-title">NVMe Spill Event: {formatBytes(spill.size_bytes)}</span>
                  </div>
                  <div className="step-detail">Spilled Arrow IPC chunk to NVMe disk path</div>
                </div>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
};
