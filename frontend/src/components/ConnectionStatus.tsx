import React from 'react';
import { AlertCircle, Wifi, WifiOff, Clock } from 'lucide-react';

interface ConnectionStatusProps {
  isConnected: boolean;
  isUsingCache: boolean;
  lastUpdate?: string;
  error?: string;
}

export const ConnectionStatus: React.FC<ConnectionStatusProps> = ({
  isConnected,
  isUsingCache,
  lastUpdate,
  error
}) => {
  const getStatusColor = () => {
    if (error) return 'text-red-500';
    if (isConnected && !isUsingCache) return 'text-green-500';
    if (isConnected && isUsingCache) return 'text-yellow-500';
    return 'text-gray-500';
  };

  const getStatusIcon = () => {
    if (error) return <AlertCircle className="w-4 h-4" />;
    if (isConnected && !isUsingCache) return <Wifi className="w-4 h-4" />;
    if (isConnected && isUsingCache) return <Clock className="w-4 h-4" />;
    return <WifiOff className="w-4 h-4" />;
  };

  const getStatusText = () => {
    if (error) {
      if (error.includes('429') || error.includes('rate limit')) {
        return 'Rate Limited';
      }
      return 'Connection Error';
    }
    if (isConnected && !isUsingCache) return 'Live Data';
    if (isConnected && isUsingCache) return 'Cached Data';
    return 'Disconnected';
  };

  return (
    <div className={`flex items-center gap-2 text-sm ${getStatusColor()}`}>
      {getStatusIcon()}
      <span>{getStatusText()}</span>
      {lastUpdate && (
        <span className="text-xs text-gray-400">
          ({new Date(lastUpdate).toLocaleTimeString()})
        </span>
      )}
    </div>
  );
};

export default ConnectionStatus;
