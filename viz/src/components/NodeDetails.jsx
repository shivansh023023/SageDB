import React from 'react';
import { X, Database, Network, FileText } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const NodeDetails = ({ node, onClose }) => {
  return (
    <AnimatePresence>
      {node && (
        <motion.div
          initial={{ x: '100%' }}
          animate={{ x: 0 }}
          exit={{ x: '100%' }}
          transition={{ type: 'spring', damping: 25, stiffness: 200 }}
          className="absolute top-0 right-0 h-full w-96 bg-slate-800/95 backdrop-blur-md border-l border-slate-700 shadow-2xl z-20 overflow-y-auto"
        >
          <div className="p-6">
            <div className="flex items-start justify-between mb-6">
              <h2 className="text-xl font-bold text-white break-words pr-4">
                {node.text.length > 50 ? node.text.substring(0, 50) + '...' : node.text}
              </h2>
              <button
                onClick={onClose}
                className="p-1 hover:bg-slate-700 rounded-full transition-colors text-slate-400 hover:text-white"
              >
                <X size={20} />
              </button>
            </div>

            <div className="space-y-6">
              {/* Basic Info */}
              <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700/50">
                <div className="flex items-center gap-2 text-blue-400 mb-2 font-semibold">
                  <Database size={16} />
                  <span>Node Info</span>
                </div>
                <div className="space-y-2 text-sm text-slate-300">
                  <div className="flex justify-between">
                    <span className="text-slate-500">ID:</span>
                    <span className="font-mono text-xs">{node.uuid.substring(0, 8)}...</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Type:</span>
                    <span>{node.metadata?.type || 'Unknown'}</span>
                  </div>
                </div>
              </div>

              {/* Metadata */}
              {node.metadata && Object.keys(node.metadata).length > 0 && (
                <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700/50">
                  <div className="flex items-center gap-2 text-emerald-400 mb-2 font-semibold">
                    <FileText size={16} />
                    <span>Metadata</span>
                  </div>
                  <pre className="text-xs text-slate-300 overflow-x-auto whitespace-pre-wrap font-mono bg-slate-950 p-2 rounded">
                    {JSON.stringify(node.metadata, null, 2)}
                  </pre>
                </div>
              )}

              {/* Connections (Placeholder - would need to fetch neighbors) */}
              <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700/50">
                <div className="flex items-center gap-2 text-purple-400 mb-2 font-semibold">
                  <Network size={16} />
                  <span>Connections</span>
                </div>
                <p className="text-xs text-slate-500 italic">
                  Select neighbors to explore connections...
                </p>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default NodeDetails;
