import React from 'react';
import { LineChart, Line, ResponsiveContainer } from 'recharts';

interface MiniSparklineProps {
  data: Array<{ value: number; timestamp?: string }>;
  color?: string;
  height?: number;
  showGradient?: boolean;
  className?: string;
}

const MiniSparkline: React.FC<MiniSparklineProps> = ({
  data,
  color = '#3B82F6',
  height = 40,
  showGradient = true,
  className = ''
}) => {
  if (!data || data.length === 0) {
    return (
      <div 
        className={`flex items-center justify-center bg-gray-100 rounded-lg ${className}`}
        style={{ height }}
      >
        <span className="text-xs text-gray-400">No data</span>
      </div>
    );
  }

  // Determine if the trend is positive or negative
  const firstValue = data[0]?.value || 0;
  const lastValue = data[data.length - 1]?.value || 0;
  const isPositive = lastValue >= firstValue;
  const lineColor = isPositive ? '#10B981' : '#EF4444';

  return (
    <div className={`relative ${className}`} style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 2, right: 2, left: 2, bottom: 2 }}>
          <defs>
            {showGradient && (
              <linearGradient id={`gradient-${color.replace('#', '')}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={lineColor} stopOpacity={0.3} />
                <stop offset="100%" stopColor={lineColor} stopOpacity={0.05} />
              </linearGradient>
            )}
          </defs>
          <Line
            type="monotone"
            dataKey="value"
            stroke={lineColor}
            strokeWidth={2}
            dot={false}
            activeDot={false}
            fill={showGradient ? `url(#gradient-${color.replace('#', '')})` : 'none'}
          />
        </LineChart>
      </ResponsiveContainer>
      
      {/* Trend indicator */}
      <div className="absolute top-1 right-1">
        <div className={`w-2 h-2 rounded-full ${
          isPositive ? 'bg-green-500' : 'bg-red-500'
        }`} />
      </div>
    </div>
  );
};

export default MiniSparkline;
