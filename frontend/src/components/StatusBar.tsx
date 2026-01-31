import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../lib/api';
import { StatusBarSkeleton } from './LoadingStates';
import ConnectionStatus from './ConnectionStatus';

function StatusBar() {
  const { data: health, isError: healthError, isLoading: healthLoading } = useQuery({
    queryKey: ['health'],
    queryFn: () => apiClient.getHealth(),
    refetchInterval: 120000,
    retry: 1,
  });

  const { data: cacheStatus, isError: cacheError, isLoading: cacheLoading } = useQuery({
    queryKey: ['cache-status'],
    queryFn: () => apiClient.getCacheStatus(),
    refetchInterval: 300000,
    retry: 1,
  });

  const isHealthy = health?.status === 'ok' && !healthError;
  const isCacheHealthy = cacheStatus?.healthy && !cacheError;
  const isLoading = healthLoading || cacheLoading;

  if (isLoading) {
    return <StatusBarSkeleton />;
  }

  return (
    <div className="relative overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-slate-900/60 backdrop-blur-md"></div>
      
      {/* Gradient border */}
      <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-slate-700 to-transparent"></div>
      
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-2.5">
        <div className="flex items-center justify-between">
          {/* Left side - Status indicators */}
          <div className="flex items-center gap-6">
            {/* API Status */}
            <div className="flex items-center gap-2">
              <div className="relative">
                <div className={`w-2 h-2 rounded-full ${isHealthy ? 'bg-emerald-400' : 'bg-rose-400'}`}></div>
                {isHealthy && (
                  <div className="absolute inset-0 w-2 h-2 rounded-full bg-emerald-400 animate-ping opacity-75"></div>
                )}
              </div>
              <span className="text-xs font-medium text-slate-400">
                API: <span className={isHealthy ? 'text-emerald-400' : 'text-rose-400'}>{isHealthy ? 'Online' : 'Offline'}</span>
              </span>
            </div>

            {/* Divider */}
            <div className="w-px h-4 bg-slate-700"></div>

            {/* Cache Status */}
            <div className="flex items-center gap-2">
              <div className="relative">
                <div className={`w-2 h-2 rounded-full ${isCacheHealthy ? 'bg-emerald-400' : 'bg-amber-400'}`}></div>
              </div>
              <span className="text-xs font-medium text-slate-400">
                Cache: <span className={isCacheHealthy ? 'text-emerald-400' : 'text-amber-400'}>{cacheStatus?.backend || 'memory'}</span>
                {isCacheHealthy && <span className="text-emerald-400 ml-1">✓</span>}
              </span>
            </div>

            {/* Divider */}
            <div className="w-px h-4 bg-slate-700"></div>

            {/* Connection Status */}
            <ConnectionStatus 
              isConnected={isHealthy}
              isUsingCache={!isHealthy && !healthError}
              lastUpdate={undefined}
              error={healthError ? 'Connection failed' : undefined}
            />
          </div>

          {/* Right side - Last updated */}
          <div className="flex items-center gap-2 text-xs">
            <svg className="w-3.5 h-3.5 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span className="text-slate-500">Updated</span>
            <span className="font-mono text-slate-300">{new Date().toLocaleTimeString()}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default StatusBar;
