import { useState, useEffect } from 'react';
import { api, getApiUrl, setApiUrl } from '../hooks/useApi';
import type { SystemInfo, TelemetryData } from '../hooks/useApi';

interface SettingsProps {
  telemetry: TelemetryData | null;
}

export const Settings: React.FC<SettingsProps> = ({ telemetry }) => {
  const config = telemetry?.config;

  const [activeSubTab, setActiveSubTab] = useState<'general' | 'advanced'>('general');

  const [apiUrlInput, setApiUrlInput] = useState<string>(getApiUrl());
  const [ramLimitMB, setRamLimitMB] = useState<number>(50);
  const [spillThresholdPct, setSpillThresholdPct] = useState<number>(85);
  const [chunkSizeRows, setChunkSizeRows] = useState<number>(65536);
  const [gpuToggle, setGpuToggle] = useState<boolean>(false);
  const [optimizerToggle, setOptimizerToggle] = useState<boolean>(true);


  const [numThreads, setNumThreads] = useState<number>(16);
  const [simdToggle, setSimdToggle] = useState<boolean>(true);
  const [arrowBufferKB, setArrowBufferKB] = useState<number>(512);
  const [compressionMode, setCompressionMode] = useState<string>('LZ4');
  const [schedulerPolicy, setSchedulerPolicy] = useState<string>('WORK_STEALING');

  const [successMsg, setSuccessMsg] = useState<string | null>(null);
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);

  useEffect(() => {
    if (config) {
      if (config.cpu_memory_limit) {
        setRamLimitMB(Math.round(config.cpu_memory_limit / (1024 * 1024)));
      }
      if (config.spill_threshold) {
        setSpillThresholdPct(Math.round(config.spill_threshold * 100));
      }
      if (config.chunk_size) {
        setChunkSizeRows(config.chunk_size);
      }
      setGpuToggle(Boolean(config.gpu_enabled));
      setOptimizerToggle(Boolean(config.enable_query_optimizer));
    }
  }, [config]);

  useEffect(() => {
    api.getSystemInfo().then(setSystemInfo).catch(console.error);
  }, []);

  const handleApplyConfig = async () => {
    try {
      await api.updateConfig({
        cpu_memory_limit: ramLimitMB > 0 ? ramLimitMB * 1024 * 1024 : null,
        spill_threshold: spillThresholdPct / 100,
        chunk_size: chunkSizeRows,
        gpu_enabled: gpuToggle,
        enable_query_optimizer: optimizerToggle,
      });
      setSuccessMsg('Configuration updated successfully!');
      setTimeout(() => setSuccessMsg(null), 3000);
    } catch (err: any) {
      alert(`Config update failed: ${err.message}`);
    }
  };

  return (
    <div className="page-container settings-page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Settings & Cluster Configuration</h1>
          <p className="page-subtitle">
            Configure host RAM memory limits, NVMe spill trigger thresholds, GPU kernel options, and query optimizer flags.
          </p>
        </div>
      </div>

      <div className="subtab-navigation" style={{ marginBottom: '1.25rem' }}>
        <button
          className={`subtab-btn ${activeSubTab === 'general' ? 'active' : ''}`}
          onClick={() => setActiveSubTab('general')}
        >
          General Tuning
        </button>
        <button
          className={`subtab-btn ${activeSubTab === 'advanced' ? 'active' : ''}`}
          onClick={() => setActiveSubTab('advanced')}
        >
          Advanced Engine Architecture
        </button>
      </div>

      <div className="settings-grid">
        {}
        <div className="glass-panel settings-panel">
          {activeSubTab === 'general' ? (
            <>
              <h3>Backend Connection & Engine Tuning</h3>
              {successMsg && <div className="success-banner">Status: {successMsg}</div>}

              <div className="form-group" style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '1rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.08)', marginBottom: '1.5rem' }}>
                <label style={{ color: '#38bdf8', fontWeight: 600 }}>Production Backend API URL (Render):</label>
                <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
                  <input
                    type="text"
                    placeholder="https://your-backend.onrender.com"
                    value={apiUrlInput}
                    onChange={(e) => setApiUrlInput(e.target.value)}
                    style={{ flex: 1 }}
                  />
                  <button
                    className="btn btn-accent"
                    onClick={() => {
                      setApiUrl(apiUrlInput);
                      setSuccessMsg('Backend API URL saved! Reloading connection...');
                      setTimeout(() => window.location.reload(), 800);
                    }}
                  >
                    Save URL
                  </button>
                </div>
                <span className="hint">
                  Enter your deployed Render backend service URL (e.g. <code>https://titanframe-backend.onrender.com</code>).
                </span>
              </div>

              <div className="form-group">
                <label>Host RAM Limit (MB):</label>
                <input
                  type="number"
                  value={ramLimitMB}
                  onChange={(e) => setRamLimitMB(Number(e.target.value))}
                />
                <span className="hint">
                  Set lower limit (e.g. 50 MB) to simulate out-of-core NVMe memory spilling.
                </span>
              </div>

              <div className="form-group">
                <label>Spill Trigger Threshold ({spillThresholdPct}%):</label>
                <input
                  type="range"
                  min="50"
                  max="98"
                  value={spillThresholdPct}
                  onChange={(e) => setSpillThresholdPct(Number(e.target.value))}
                />
              </div>

              <div className="form-group">
                <label>Chunk Size (Rows per Chunk):</label>
                <select
                  value={chunkSizeRows}
                  onChange={(e) => setChunkSizeRows(Number(e.target.value))}
                >
                  <option value={16384}>16,384 rows</option>
                  <option value={65536}>65,536 rows (Default)</option>
                  <option value={131072}>131,072 rows</option>
                  <option value={262144}>262,144 rows</option>
                </select>
              </div>

              <div className="form-group checkbox-row">
                <input
                  type="checkbox"
                  id="gpuCheck"
                  checked={gpuToggle}
                  onChange={(e) => setGpuToggle(e.target.checked)}
                />
                <label htmlFor="gpuCheck">Enable GPU Acceleration (CuPy / Triton Kernels)</label>
              </div>

              <div className="form-group checkbox-row">
                <input
                  type="checkbox"
                  id="optCheck"
                  checked={optimizerToggle}
                  onChange={(e) => setOptimizerToggle(e.target.checked)}
                />
                <label htmlFor="optCheck">
                  Enable Query Optimizer (Predicate Pushdown & Projection Pruning)
                </label>
              </div>
            </>
          ) : (
            <>
              <h3>Advanced Architectural Controls</h3>
              {successMsg && <div className="success-banner">Status: {successMsg}</div>}

              <div className="form-group">
                <label>Thread Pool Workers:</label>
                <select value={numThreads} onChange={(e) => setNumThreads(Number(e.target.value))}>
                  <option value={4}>4 Threads</option>
                  <option value={8}>8 Threads</option>
                  <option value={16}>16 Threads (Optimal)</option>
                  <option value={32}>32 Threads</option>
                </select>
              </div>

              <div className="form-group checkbox-row">
                <input
                  type="checkbox"
                  id="simdCheck"
                  checked={simdToggle}
                  onChange={(e) => setSimdToggle(e.target.checked)}
                />
                <label htmlFor="simdCheck">Enable Hardware SIMD Vectorization (AVX-512 / NEON)</label>
              </div>

              <div className="form-group">
                <label>Apache Arrow IPC Buffer Size (KB):</label>
                <input
                  type="number"
                  value={arrowBufferKB}
                  onChange={(e) => setArrowBufferKB(Number(e.target.value))}
                />
              </div>

              <div className="form-group">
                <label>NVMe Spill Compression Mode:</label>
                <select value={compressionMode} onChange={(e) => setCompressionMode(e.target.value)}>
                  <option value="LZ4">LZ4 (High Speed, Low CPU)</option>
                  <option value="ZSTD">ZSTD (High Compression Ratio)</option>
                  <option value="UNCOMPRESSED">Uncompressed (Zero CPU Overhead)</option>
                </select>
              </div>

              <div className="form-group">
                <label>Query Worker Scheduler Policy:</label>
                <select value={schedulerPolicy} onChange={(e) => setSchedulerPolicy(e.target.value)}>
                  <option value="WORK_STEALING">Work Stealing Pool (Default)</option>
                  <option value="ROUND_ROBIN">Round Robin</option>
                  <option value="FIFO">FIFO Priority</option>
                </select>
              </div>
            </>
          )}

          <button className="btn btn-accent" onClick={handleApplyConfig} style={{ marginTop: '1rem' }}>
            Apply Engine Configuration
          </button>
        </div>

        {}
        <div className="glass-panel sysinfo-panel">
          <h3>Environment & Cluster Info</h3>

          <div className="info-list">
            <div className="info-row">
              <span className="info-key">TitanFrame Version</span>
              <span className="info-val">v{systemInfo?.titanframe_version || '1.0.0'}</span>
            </div>
            <div className="info-row">
              <span className="info-key">Operating System</span>
              <span className="info-val">{systemInfo?.os} {systemInfo?.os_release}</span>
            </div>
            <div className="info-row">
              <span className="info-key">Python Runtime</span>
              <span className="info-val">v{systemInfo?.python_version}</span>
            </div>
            <div className="info-row">
              <span className="info-key">CPU Cores</span>
              <span className="info-val">{systemInfo?.cpu_count} Thread Pool Worker(s)</span>
            </div>
            <div className="info-row">
              <span className="info-key">NVMe Spill Path</span>
              <span className="info-val path">
                <code>{config?.nvme_spill_path || '~/.titanframe/spill'}</code>
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
