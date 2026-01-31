import React, { useState, useEffect } from 'react';
import { Activity, AlertCircle, TrendingUp, TrendingDown, Zap, CheckCircle, XCircle, RefreshCw } from 'lucide-react';
import { API_BASE_URL } from '../lib/api';

interface APIStats {
  config: {
    requests_per_minute: number;
    requests_per_hour: number;
    free_tier: boolean;
    pro_cost_per_month?: number;
  };
  usage: {
    requests_last_minute: number;
    requests_last_hour: number;
    requests_last_day: number;
    utilization_percent: number;
  };
  performance: {
    total_requests: number;
    success_rate: number;
    average_duration_ms: number;
    rate_limited_requests: number;
  };
  predictions: {
    predicted_rpm: number;
    predicted_time_to_limit_seconds: number;
    should_upgrade: boolean;
  };
}

interface BatchingStats {
  total_requests: number;
  batched_requests: number;
  api_calls_saved: number;
  average_batch_size: number;
  efficiency_percent: number;
  current_pending_batches: number;
}

interface Recommendation {
  type: string;
  api: string;
  priority: string;
  reason: string;
  action: string;
  expected_benefit: string;
}

const RateLimitDashboard: React.FC = () => {
  const [apis, setApis] = useState<Record<string, APIStats>>({});
  const [batchingStats, setBatchingStats] = useState<BatchingStats | null>(null);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const fetchData = async () => {
    try {
      setError(null);

      // Fetch rate limit status
      const statusResponse = await fetch(`${API_BASE_URL}/rate-limit/status`);
      const statusData = await statusResponse.json();
      
      if (statusData.success) {
        setApis(statusData.apis);
      }

      // Fetch batching stats
      const batchResponse = await fetch(`${API_BASE_URL}/rate-limit/batching/stats`);
      const batchData = await batchResponse.json();
      
      if (batchData.success) {
        setBatchingStats(batchData.batching);
      }

      // Fetch recommendations
      const recResponse = await fetch(`${API_BASE_URL}/rate-limit/recommendations`);
      const recData = await recResponse.json();
      
      if (recData.success) {
        setRecommendations(recData.recommendations);
      }

      setLoading(false);
    } catch (err) {
      console.error('Error fetching rate limit data:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();

    if (autoRefresh) {
      const interval = setInterval(fetchData, 10000); // Refresh every 10s
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const getUtilizationColor = (percent: number) => {
    if (percent < 50) return 'text-green-400';
    if (percent < 80) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getUtilizationBg = (percent: number) => {
    if (percent < 50) return 'bg-green-500';
    if (percent < 80) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const getPriorityBadge = (priority: string) => {
    const colors = {
      high: 'bg-red-900/30 text-red-400 border-red-800',
      medium: 'bg-yellow-900/30 text-yellow-400 border-yellow-800',
      low: 'bg-blue-900/30 text-blue-400 border-blue-800',
    };
    return colors[priority as keyof typeof colors] || colors.low;
  };

  if (loading) {
    return (
      <div className="bg-gray-900 rounded-lg p-6">
        <div className="flex items-center justify-center space-x-2 text-gray-400">
          <RefreshCw className="w-5 h-5 animate-spin" />
          <span>Loading rate limit data...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Activity className="w-6 h-6 text-blue-400" />
          <h2 className="text-2xl font-bold text-white">API Rate Limit Monitor</h2>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`px-4 py-2 rounded-lg transition-colors ${
              autoRefresh
                ? 'bg-blue-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            <RefreshCw className={`w-4 h-4 inline mr-2 ${autoRefresh ? 'animate-spin' : ''}`} />
            {autoRefresh ? 'Auto-refresh ON' : 'Auto-refresh OFF'}
          </button>
          
          <button
            onClick={fetchData}
            className="px-4 py-2 rounded-lg bg-gray-800 text-gray-400 hover:bg-gray-700 transition-colors"
          >
            <RefreshCw className="w-4 h-4 inline mr-2" />
            Refresh Now
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-900/20 border border-red-800 rounded-lg p-4">
          <div className="flex items-center space-x-2 text-red-400">
            <AlertCircle className="w-5 h-5" />
            <span>{error}</span>
          </div>
        </div>
      )}

      {/* Batching Stats Overview */}
      {batchingStats && (
        <div className="bg-gray-900 rounded-lg p-6">
          <h3 className="text-lg font-bold text-white mb-4 flex items-center">
            <Zap className="w-5 h-5 mr-2 text-yellow-400" />
            Request Batching Efficiency
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            <div className="bg-gray-800 rounded-lg p-4">
              <div className="text-sm text-gray-400 mb-1">Total Requests</div>
              <div className="text-2xl font-bold text-white">
                {batchingStats.total_requests.toLocaleString()}
              </div>
            </div>
            
            <div className="bg-gray-800 rounded-lg p-4">
              <div className="text-sm text-gray-400 mb-1">API Calls Saved</div>
              <div className="text-2xl font-bold text-green-400">
                {batchingStats.api_calls_saved.toLocaleString()}
              </div>
            </div>
            
            <div className="bg-gray-800 rounded-lg p-4">
              <div className="text-sm text-gray-400 mb-1">Efficiency</div>
              <div className="text-2xl font-bold text-blue-400">
                {batchingStats.efficiency_percent.toFixed(1)}%
              </div>
            </div>
            
            <div className="bg-gray-800 rounded-lg p-4">
              <div className="text-sm text-gray-400 mb-1">Avg Batch Size</div>
              <div className="text-2xl font-bold text-purple-400">
                {batchingStats.average_batch_size.toFixed(1)}
              </div>
            </div>
            
            <div className="bg-gray-800 rounded-lg p-4">
              <div className="text-sm text-gray-400 mb-1">Pending</div>
              <div className="text-2xl font-bold text-yellow-400">
                {batchingStats.current_pending_batches}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* API Status Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {Object.entries(apis).map(([apiName, stats]) => (
          <div key={apiName} className="bg-gray-900 rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-white capitalize">{apiName}</h3>
              {stats.config.free_tier && (
                <span className="px-2 py-1 rounded bg-gray-800 text-xs text-gray-400">
                  Free Tier
                </span>
              )}
            </div>

            {/* Utilization Bar */}
            <div className="mb-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-400">Current Utilization</span>
                <span className={`text-sm font-bold ${getUtilizationColor(stats.usage.utilization_percent)}`}>
                  {stats.usage.utilization_percent.toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-gray-800 rounded-full h-2">
                <div
                  className={`h-2 rounded-full ${getUtilizationBg(stats.usage.utilization_percent)} transition-all duration-500`}
                  style={{ width: `${Math.min(stats.usage.utilization_percent, 100)}%` }}
                />
              </div>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-2 gap-3 mb-4">
              <div className="bg-gray-800 rounded p-3">
                <div className="text-xs text-gray-400 mb-1">Requests (1m)</div>
                <div className="text-lg font-bold text-white">
                  {stats.usage.requests_last_minute} / {stats.config.requests_per_minute}
                </div>
              </div>
              
              <div className="bg-gray-800 rounded p-3">
                <div className="text-xs text-gray-400 mb-1">Requests (1h)</div>
                <div className="text-lg font-bold text-white">
                  {stats.usage.requests_last_hour} / {stats.config.requests_per_hour}
                </div>
              </div>
              
              <div className="bg-gray-800 rounded p-3">
                <div className="text-xs text-gray-400 mb-1">Success Rate</div>
                <div className="text-lg font-bold text-green-400">
                  {stats.performance.success_rate.toFixed(1)}%
                </div>
              </div>
              
              <div className="bg-gray-800 rounded p-3">
                <div className="text-xs text-gray-400 mb-1">Avg Duration</div>
                <div className="text-lg font-bold text-blue-400">
                  {stats.performance.average_duration_ms.toFixed(0)}ms
                </div>
              </div>
            </div>

            {/* Predictions */}
            <div className="bg-gray-800 rounded p-3 mb-3">
              <div className="text-xs text-gray-400 mb-2">Predictions</div>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Predicted RPM:</span>
                  <span className="text-white font-medium">
                    {stats.predictions.predicted_rpm.toFixed(1)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Time to limit:</span>
                  <span className="text-white font-medium">
                    {stats.predictions.predicted_time_to_limit_seconds === Infinity
                      ? '∞'
                      : `${Math.round(stats.predictions.predicted_time_to_limit_seconds)}s`}
                  </span>
                </div>
              </div>
            </div>

            {/* Upgrade Warning */}
            {stats.predictions.should_upgrade && (
              <div className="bg-yellow-900/20 border border-yellow-800 rounded p-3">
                <div className="flex items-start space-x-2">
                  <AlertCircle className="w-4 h-4 text-yellow-400 mt-0.5" />
                  <div className="flex-1">
                    <div className="text-sm font-medium text-yellow-400 mb-1">
                      Upgrade Recommended
                    </div>
                    <div className="text-xs text-yellow-300">
                      Consider upgrading to Pro tier for better performance
                      {stats.config.pro_cost_per_month && (
                        <span className="block mt-1">
                          Cost: ${stats.config.pro_cost_per_month}/month
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Rate Limited Warnings */}
            {stats.performance.rate_limited_requests > 0 && (
              <div className="bg-red-900/20 border border-red-800 rounded p-3 mt-3">
                <div className="flex items-center space-x-2 text-red-400 text-sm">
                  <XCircle className="w-4 h-4" />
                  <span>
                    Rate limited {stats.performance.rate_limited_requests} times
                  </span>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Recommendations */}
      {recommendations.length > 0 && (
        <div className="bg-gray-900 rounded-lg p-6">
          <h3 className="text-lg font-bold text-white mb-4 flex items-center">
            <TrendingUp className="w-5 h-5 mr-2 text-green-400" />
            Optimization Recommendations
          </h3>
          
          <div className="space-y-3">
            {recommendations.map((rec, index) => (
              <div
                key={index}
                className={`border rounded-lg p-4 ${getPriorityBadge(rec.priority)}`}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <span className="text-sm font-bold uppercase">
                      {rec.type}
                    </span>
                    <span className="text-xs px-2 py-0.5 rounded bg-gray-800">
                      {rec.api}
                    </span>
                  </div>
                  <span className={`text-xs px-2 py-0.5 rounded ${
                    rec.priority === 'high' ? 'bg-red-800 text-red-200' :
                    rec.priority === 'medium' ? 'bg-yellow-800 text-yellow-200' :
                    'bg-blue-800 text-blue-200'
                  }`}>
                    {rec.priority} priority
                  </span>
                </div>
                
                <div className="space-y-2 text-sm">
                  <div>
                    <span className="text-gray-400">Reason: </span>
                    <span className="text-white">{rec.reason}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Action: </span>
                    <span className="text-white">{rec.action}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Expected Benefit: </span>
                    <span className="text-green-400">{rec.expected_benefit}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {recommendations.length === 0 && (
        <div className="bg-green-900/20 border border-green-800 rounded-lg p-6">
          <div className="flex items-center space-x-3 text-green-400">
            <CheckCircle className="w-6 h-6" />
            <div>
              <div className="font-bold mb-1">All Systems Optimal</div>
              <div className="text-sm text-green-300">
                No recommendations at this time. Your API usage is well optimized!
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default RateLimitDashboard;

