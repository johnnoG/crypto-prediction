import React from 'react';
import RateLimitDashboard from '../RateLimitDashboard';
import Header from '../Header';

const RateLimitPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      <Header />
      
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-7xl mx-auto">
          {/* Page Header */}
          <div className="mb-8">
            <h1 className="text-4xl font-bold text-white mb-2">
              API Rate Limit Optimizer
            </h1>
            <p className="text-gray-400">
              Intelligent rate limiting, request batching, and predictive analytics
              to maximize API reliability and minimize costs.
            </p>
          </div>

          {/* Info Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            <div className="bg-blue-900/20 border border-blue-800 rounded-lg p-4">
              <h3 className="text-blue-400 font-semibold mb-2">🎯 Smart Rate Limiting</h3>
              <p className="text-gray-300 text-sm">
                Priority-based queuing and predictive throttling prevent rate limit errors
                while maximizing throughput.
              </p>
            </div>
            
            <div className="bg-green-900/20 border border-green-800 rounded-lg p-4">
              <h3 className="text-green-400 font-semibold mb-2">⚡ Request Batching</h3>
              <p className="text-gray-300 text-sm">
                Automatic request merging and deduplication can save 35-50% of API calls,
                reducing costs and improving speed.
              </p>
            </div>
            
            <div className="bg-purple-900/20 border border-purple-800 rounded-lg p-4">
              <h3 className="text-purple-400 font-semibold mb-2">📊 Predictive Analytics</h3>
              <p className="text-gray-300 text-sm">
                Real-time forecasting and auto-upgrade recommendations help you make
                data-driven decisions about API tiers.
              </p>
            </div>
          </div>

          {/* Dashboard */}
          <RateLimitDashboard />
        </div>
      </div>
    </div>
  );
};

export default RateLimitPage;

