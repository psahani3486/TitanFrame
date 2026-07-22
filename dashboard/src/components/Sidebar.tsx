import React from 'react';
import './Sidebar.css';

export type PageId =
  | 'executive'
  | 'datasets'
  | 'sql'
  | 'visualizer'
  | 'benchmarks'
  | 'memory'
  | 'history'
  | 'settings';

interface SidebarProps {
  activePage: PageId;
  onSelectPage: (page: PageId) => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
  activeQueriesCount: number;
}

const NAV_ITEMS: Array<{ id: PageId; label: string; iconText: string }> = [
  { id: 'executive', label: 'Executive Dashboard', iconText: 'DB' },
  { id: 'datasets', label: 'Dataset Explorer', iconText: 'DS' },
  { id: 'sql', label: 'SQL Workspace', iconText: 'SQL' },
  { id: 'visualizer', label: 'Query Plan Visualizer', iconText: 'DAG' },
  { id: 'benchmarks', label: 'Benchmark Dashboard', iconText: 'BM' },
  { id: 'memory', label: 'Memory & GPU Monitor', iconText: 'MEM' },
  { id: 'history', label: 'Query History & Logs', iconText: 'LOG' },
  { id: 'settings', label: 'Settings & Config', iconText: 'CFG' },
];

export const Sidebar: React.FC<SidebarProps> = ({
  activePage,
  onSelectPage,
  collapsed,
  onToggleCollapse,
  activeQueriesCount,
}) => {
  return (
    <aside className={`titan-sidebar ${collapsed ? 'collapsed' : ''}`}>
      {}
      <div className="sidebar-brand" onClick={onToggleCollapse} title="Toggle sidebar">
        <div className="brand-logo-text">TF</div>
        {!collapsed && (
          <div className="brand-text">
            <span className="brand-name">TitanFrame</span>
            <span className="brand-tag">Studio v1.0</span>
          </div>
        )}
      </div>

      {}
      <nav className="sidebar-nav">
        {NAV_ITEMS.map((item) => {
          const isActive = activePage === item.id;
          return (
            <button
              key={item.id}
              className={`nav-item ${isActive ? 'active' : ''}`}
              onClick={() => onSelectPage(item.id)}
              title={collapsed ? item.label : undefined}
            >
              <span className="nav-icon-badge">{item.iconText}</span>
              {!collapsed && <span className="nav-label">{item.label}</span>}
              {!collapsed && item.id === 'history' && activeQueriesCount > 0 && (
                <span className="nav-badge pulse">{activeQueriesCount}</span>
              )}
            </button>
          );
        })}
      </nav>

      {}
      <div className="sidebar-footer">
        <button className="collapse-btn" onClick={onToggleCollapse}>
          {collapsed ? '>>' : '<< Collapse Navigation'}
        </button>
      </div>
    </aside>
  );
};
