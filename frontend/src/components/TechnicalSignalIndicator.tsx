import React from 'react';

interface TechnicalSignalIndicatorProps {
  rsi?: number;
  regime?: string;
  macd?: number;
  bollingerPosition?: number;
  volatility?: number;
  size?: 'sm' | 'md';
  className?: string;
}

const TechnicalSignalIndicator: React.FC<TechnicalSignalIndicatorProps> = ({
  rsi,
  regime,
  macd,
  bollingerPosition,
  volatility,
  size = 'sm',
  className = ''
}) => {
  const sizeClasses = {
    sm: 'w-6 h-6 text-xs',
    md: 'w-8 h-8 text-sm'
  };

  const getRSIIcon = (rsiValue: number) => {
    if (rsiValue > 70) return (
      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
        <circle cx="10" cy="10" r="8" />
      </svg>
    );
    if (rsiValue < 30) return (
      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
        <circle cx="10" cy="10" r="8" />
      </svg>
    );
    return (
      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
        <circle cx="10" cy="10" r="8" />
      </svg>
    );
  };

  const getRSIColor = (rsiValue: number) => {
    if (rsiValue > 70) return 'text-red-600 bg-red-100';
    if (rsiValue < 30) return 'text-green-600 bg-green-100';
    return 'text-yellow-600 bg-yellow-100';
  };

  const getRegimeIcon = (regimeValue: string) => {
    switch (regimeValue?.toLowerCase()) {
      case 'bullish': return (
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
        </svg>
      );
      case 'bearish': return (
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
        </svg>
      );
      case 'overbought': return (
        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
          <circle cx="10" cy="10" r="8" />
        </svg>
      );
      case 'oversold': return (
        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
          <circle cx="10" cy="10" r="8" />
        </svg>
      );
      default: return (
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
      );
    }
  };

  const getRegimeColor = (regimeValue: string) => {
    switch (regimeValue?.toLowerCase()) {
      case 'bullish': return 'text-green-600 bg-green-100';
      case 'bearish': return 'text-red-600 bg-red-100';
      case 'overbought': return 'text-red-600 bg-red-100';
      case 'oversold': return 'text-green-600 bg-green-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getMACDIcon = (macdValue: number) => {
    if (macdValue > 0) return (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
      </svg>
    );
    if (macdValue < 0) return (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
      </svg>
    );
    return (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
      </svg>
    );
  };

  const getMACDColor = (macdValue: number) => {
    if (macdValue > 0) return 'text-green-600 bg-green-100';
    if (macdValue < 0) return 'text-red-600 bg-red-100';
    return 'text-gray-600 bg-gray-100';
  };

  const getVolatilityIcon = (volValue: number) => {
    if (volValue > 0.5) return (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    );
    if (volValue > 0.3) return (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    );
    return (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
      </svg>
    );
  };

  const getVolatilityColor = (volValue: number) => {
    if (volValue > 0.5) return 'text-orange-600 bg-orange-100';
    if (volValue > 0.3) return 'text-yellow-600 bg-yellow-100';
    return 'text-blue-600 bg-blue-100';
  };

  return (
    <div className={`flex items-center space-x-1 ${className}`}>
      {/* RSI Indicator */}
      {rsi !== undefined && (
        <div 
          className={`${sizeClasses[size]} rounded-full flex items-center justify-center ${getRSIColor(rsi)}`}
          title={`RSI: ${rsi.toFixed(1)}`}
        >
          {getRSIIcon(rsi)}
        </div>
      )}

      {/* Market Regime */}
      {regime && (
        <div 
          className={`${sizeClasses[size]} rounded-full flex items-center justify-center ${getRegimeColor(regime)}`}
          title={`Market Regime: ${regime}`}
        >
          {getRegimeIcon(regime)}
        </div>
      )}

      {/* MACD */}
      {macd !== undefined && (
        <div 
          className={`${sizeClasses[size]} rounded-full flex items-center justify-center ${getMACDColor(macd)}`}
          title={`MACD: ${macd.toFixed(2)}`}
        >
          {getMACDIcon(macd)}
        </div>
      )}

      {/* Volatility */}
      {volatility !== undefined && (
        <div 
          className={`${sizeClasses[size]} rounded-full flex items-center justify-center ${getVolatilityColor(volatility)}`}
          title={`Volatility: ${(volatility * 100).toFixed(1)}%`}
        >
          {getVolatilityIcon(volatility)}
        </div>
      )}

      {/* Bollinger Bands Position */}
      {bollingerPosition !== undefined && (
        <div 
          className={`${sizeClasses[size]} rounded-full flex items-center justify-center ${
            bollingerPosition > 0.8 ? 'text-red-600 bg-red-100 dark:text-red-400 dark:bg-red-900/30' :
            bollingerPosition < 0.2 ? 'text-green-600 bg-green-100 dark:text-green-400 dark:bg-green-900/30' :
            'text-gray-600 bg-gray-100 dark:text-gray-400 dark:bg-gray-800'
          }`}
          title={`Bollinger Position: ${(bollingerPosition * 100).toFixed(1)}%`}
        >
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <circle cx="10" cy="10" r="8" />
          </svg>
        </div>
      )}
    </div>
  );
};

export default TechnicalSignalIndicator;
