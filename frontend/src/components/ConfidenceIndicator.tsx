import React from 'react';

interface ConfidenceIndicatorProps {
  confidence: number; // 0-1
  size?: 'sm' | 'md' | 'lg';
  showPercentage?: boolean;
  className?: string;
}

const ConfidenceIndicator: React.FC<ConfidenceIndicatorProps> = ({
  confidence,
  size = 'md',
  showPercentage = true,
  className = ''
}) => {
  const percentage = Math.round(confidence * 100);
  
  const sizeClasses = {
    sm: 'w-12 h-2',
    md: 'w-16 h-3',
    lg: 'w-20 h-4'
  };

  const getConfidenceColor = (conf: number) => {
    if (conf >= 0.8) return 'bg-green-500';
    if (conf >= 0.6) return 'bg-yellow-500';
    if (conf >= 0.4) return 'bg-orange-500';
    return 'bg-red-500';
  };

  const getConfidenceTextColor = (conf: number) => {
    if (conf >= 0.8) return 'text-green-700';
    if (conf >= 0.6) return 'text-yellow-700';
    if (conf >= 0.4) return 'text-orange-700';
    return 'text-red-700';
  };

  const getConfidenceLabel = (conf: number) => {
    if (conf >= 0.8) return 'High';
    if (conf >= 0.6) return 'Medium';
    if (conf >= 0.4) return 'Low';
    return 'Very Low';
  };

  return (
    <div className={`flex items-center space-x-2 ${className}`}>
      <div className="flex-1">
        <div className={`${sizeClasses[size]} bg-gray-200 rounded-full overflow-hidden`}>
          <div
            className={`h-full ${getConfidenceColor(confidence)} transition-all duration-500 ease-out`}
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>
      {showPercentage && (
        <div className="flex flex-col items-end">
          <span className={`text-xs font-bold ${getConfidenceTextColor(confidence)}`}>
            {percentage}%
          </span>
          <span className="text-xs text-gray-500">
            {getConfidenceLabel(confidence)}
          </span>
        </div>
      )}
    </div>
  );
};

export default ConfidenceIndicator;
