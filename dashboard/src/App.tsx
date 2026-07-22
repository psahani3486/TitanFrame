import { useState, useEffect } from 'react';
import './index.css';
import './App.css';
import { useTelemetry, api } from './hooks/useApi';
import type { DatasetInfo } from './hooks/useApi';
import { Sidebar } from './components/Sidebar';
import type { PageId } from './components/Sidebar';
import { StatusBar } from './components/StatusBar';


// Page Modules
import { ExecutiveDashboard } from './pages/ExecutiveDashboard';
import { DatasetExplorer } from './pages/DatasetExplorer';
import { SqlWorkspace } from './pages/SqlWorkspace';
import { QueryPlanVisualizer } from './pages/QueryPlanVisualizer';
import { BenchmarkDashboard } from './pages/BenchmarkDashboard';
import { MemoryMonitor } from './pages/MemoryMonitor';
import { QueryHistory } from './pages/QueryHistory';
import { Settings } from './pages/Settings';

export default function App() {
  const [activePage, setActivePage] = useState<PageId>('executive');
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(false);
  const [refreshInterval, setRefreshInterval] = useState<number>(1000);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selectedDatasetForQuery, setSelectedDatasetForQuery] = useState<string | undefined>();

  const { telemetry, isConnected } = useTelemetry(refreshInterval);

  useEffect(() => {
    api.getDatasets().then(setDatasets).catch(console.error);
  }, []);

  const handleSelectDatasetForQuery = (path: string) => {
    setSelectedDatasetForQuery(path);
    setActivePage('sql');
  };

  const activeQueriesCount =
    telemetry?.queries.filter((q) => q.status === 'RUNNING').length || 0;

  return (
    <div className="titan-app-shell">
      <div className="shell-body">
        {/* Persistent Collapsible Sidebar */}
        <Sidebar
          activePage={activePage}
          onSelectPage={setActivePage}
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
          activeQueriesCount={activeQueriesCount}
        />

        {/* Dynamic Main Page Content View */}
        <main className="shell-content">
          {activePage === 'executive' && (
            <ExecutiveDashboard
              telemetry={telemetry}
              datasets={datasets}
              onNavigate={setActivePage}
            />
          )}

          {activePage === 'datasets' && (
            <DatasetExplorer
              datasets={datasets}
              onSelectForQuery={handleSelectDatasetForQuery}
            />
          )}

          {activePage === 'sql' && (
            <SqlWorkspace
              datasets={datasets}
              selectedDatasetPath={selectedDatasetForQuery}
              onNavigateToDag={() => setActivePage('visualizer')}
            />
          )}

          {activePage === 'visualizer' && (
            <QueryPlanVisualizer telemetry={telemetry} />
          )}

          {activePage === 'benchmarks' && (
            <BenchmarkDashboard datasets={datasets} />
          )}

          {activePage === 'memory' && (
            <MemoryMonitor telemetry={telemetry} />
          )}

          {activePage === 'history' && (
            <QueryHistory telemetry={telemetry} />
          )}

          {activePage === 'settings' && (
            <Settings telemetry={telemetry} />
          )}
        </main>
      </div>

      {/* Persistent Bottom Status Bar */}
      <StatusBar
        isConnected={isConnected}
        telemetry={telemetry}
        refreshInterval={refreshInterval}
        onRefreshIntervalChange={setRefreshInterval}
      />
    </div>
  );
}
