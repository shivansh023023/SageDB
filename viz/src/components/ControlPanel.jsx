import React from 'react';
import { Settings, ZoomIn, ZoomOut, RefreshCw } from 'lucide-react';

const ControlPanel = ({ params, setParams, onRefresh }) => {
  const handleChange = (key, value) => {
    setParams(prev => ({ ...prev, [key]: parseFloat(value) }));
  };

  return (
    <div className="absolute top-4 left-4 bg-slate-800/90 backdrop-blur-md p-4 rounded-xl border border-slate-700 shadow-xl w-64 text-slate-200 z-10">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-bold text-lg flex items-center gap-2">
          <Settings size={18} /> Controls
        </h3>
        <button 
          onClick={onRefresh}
          className="p-1.5 hover:bg-slate-700 rounded-lg transition-colors"
          title="Refresh Data"
        >
          <RefreshCw size={16} />
        </button>
      </div>

      <div className="space-y-4">
        <div>
          <label className="text-xs font-medium text-slate-400 mb-1 block">Node Size</label>
          <input
            type="range"
            min="1"
            max="10"
            step="0.5"
            value={params.nodeSize}
            onChange={(e) => handleChange('nodeSize', e.target.value)}
            className="w-full accent-blue-500 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer"
          />
          <div className="text-right text-xs text-slate-500">{params.nodeSize}</div>
        </div>

        <div>
          <label className="text-xs font-medium text-slate-400 mb-1 block">Link Width</label>
          <input
            type="range"
            min="0.5"
            max="5"
            step="0.5"
            value={params.linkWidth}
            onChange={(e) => handleChange('linkWidth', e.target.value)}
            className="w-full accent-blue-500 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer"
          />
          <div className="text-right text-xs text-slate-500">{params.linkWidth}</div>
        </div>

        <div>
          <label className="text-xs font-medium text-slate-400 mb-1 block">Repulsion Strength</label>
          <input
            type="range"
            min="10"
            max="500"
            step="10"
            value={params.chargeStrength}
            onChange={(e) => handleChange('chargeStrength', e.target.value)}
            className="w-full accent-blue-500 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer"
          />
          <div className="text-right text-xs text-slate-500">{params.chargeStrength}</div>
        </div>

        <div>
          <label className="text-xs font-medium text-slate-400 mb-1 block">Link Distance</label>
          <input
            type="range"
            min="10"
            max="200"
            step="5"
            value={params.linkDistance}
            onChange={(e) => handleChange('linkDistance', e.target.value)}
            className="w-full accent-blue-500 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer"
          />
          <div className="text-right text-xs text-slate-500">{params.linkDistance}</div>
        </div>
      </div>
    </div>
  );
};

export default ControlPanel;
