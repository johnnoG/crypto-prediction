import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType } from 'lightweight-charts';
import type {
  IChartApi,
  ISeriesApi,
  CandlestickData,
  HistogramData,
  LineData,
  Time,
} from 'lightweight-charts';
import { TrendingUp, TrendingDown, Maximize2, Download, Settings, Plus, Minus } from 'lucide-react';

interface OHLCData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface TechnicalIndicators {
  sma20?: number[];
  sma50?: number[];
  ema20?: number[];
  rsi?: number[];
  macd?: {
    macd: number[];
    signal: number[];
    histogram: number[];
  };
}

interface TradingViewChartProps {
  data: OHLCData[];
  symbol: string;
  timeframe?: string;
  height?: number;
  showVolume?: boolean;
  showIndicators?: boolean;
  indicators?: TechnicalIndicators;
  onTimeframeChange?: (timeframe: string) => void;
}

const TradingViewChart: React.FC<TradingViewChartProps> = ({
  data,
  symbol,
  timeframe = '1D',
  height = 500,
  showVolume = true,
  showIndicators = true,
  indicators,
  onTimeframeChange,
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const [activeIndicators, setActiveIndicators] = useState<string[]>(['SMA20', 'Volume']);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Timeframe options
  const timeframes = ['1m', '5m', '15m', '1h', '4h', '1D', '1W', '1M'];

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#0B1120' },
        textColor: '#9CA3AF',
      },
      grid: {
        vertLines: { color: '#1F2937' },
        horzLines: { color: '#1F2937' },
      },
      width: chartContainerRef.current.clientWidth,
      height: height,
      rightPriceScale: {
        borderColor: '#374151',
        scaleMargins: {
          top: 0.1,
          bottom: showVolume ? 0.3 : 0.1,
        },
      },
      timeScale: {
        borderColor: '#374151',
        timeVisible: true,
        secondsVisible: false,
      },
      crosshair: {
        mode: 1,
        vertLine: {
          width: 1,
          color: '#6B7280',
          style: 2,
          labelBackgroundColor: '#3B82F6',
        },
        horzLine: {
          width: 1,
          color: '#6B7280',
          style: 2,
          labelBackgroundColor: '#3B82F6',
        },
      },
    });

    chartRef.current = chart;

    // Candlestick series (v4 API)
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#10B981',
      downColor: '#EF4444',
      borderUpColor: '#10B981',
      borderDownColor: '#EF4444',
      wickUpColor: '#10B981',
      wickDownColor: '#EF4444',
    });

    candlestickSeriesRef.current = candlestickSeries;

    // Volume series (v4 API)
    if (showVolume) {
      const volumeSeries = chart.addHistogramSeries({
        color: '#6B7280',
        priceFormat: {
          type: 'volume',
        },
        priceScaleId: '',
      });

      volumeSeries.priceScale().applyOptions({
        scaleMargins: {
          top: 0.7,
          bottom: 0,
        },
      });

      volumeSeriesRef.current = volumeSeries;
    }

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [height, showVolume]);

  // Update data
  useEffect(() => {
    if (!candlestickSeriesRef.current || !data.length) return;

    // Convert data to candlestick format
    const candleData: CandlestickData[] = data.map((d) => ({
      time: (d.time / 1000) as Time, // Convert to seconds
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
    }));

    candlestickSeriesRef.current.setData(candleData);

    // Update volume
    if (volumeSeriesRef.current && data[0]?.volume !== undefined) {
      const volumeData: HistogramData[] = data.map((d, i) => ({
        time: (d.time / 1000) as Time,
        value: d.volume || 0,
        color:
          i > 0 && d.close >= data[i - 1].close
            ? 'rgba(16, 185, 129, 0.5)'
            : 'rgba(239, 68, 68, 0.5)',
      }));

      volumeSeriesRef.current.setData(volumeData);
    }

    // Fit content
    chartRef.current?.timeScale().fitContent();
  }, [data]);

  // Add technical indicators
  useEffect(() => {
    if (!chartRef.current || !indicators) return;

    // Add SMA20 (v4 API)
    if (activeIndicators.includes('SMA20') && indicators.sma20) {
      const sma20Series = chartRef.current.addLineSeries({
        color: '#3B82F6',
        lineWidth: 2,
        title: 'SMA 20',
      });

      const sma20Data: LineData[] = indicators.sma20.map((value, i) => ({
        time: (data[i].time / 1000) as Time,
        value,
      }));

      sma20Series.setData(sma20Data);
    }

    // Add SMA50 (v4 API)
    if (activeIndicators.includes('SMA50') && indicators.sma50) {
      const sma50Series = chartRef.current.addLineSeries({
        color: '#8B5CF6',
        lineWidth: 2,
        title: 'SMA 50',
      });

      const sma50Data: LineData[] = indicators.sma50.map((value, i) => ({
        time: (data[i].time / 1000) as Time,
        value,
      }));

      sma50Series.setData(sma50Data);
    }

    // Add EMA20 (v4 API)
    if (activeIndicators.includes('EMA20') && indicators.ema20) {
      const ema20Series = chartRef.current.addLineSeries({
        color: '#F59E0B',
        lineWidth: 2,
        title: 'EMA 20',
        lineStyle: 2, // Dashed
      });

      const ema20Data: LineData[] = indicators.ema20.map((value, i) => ({
        time: (data[i].time / 1000) as Time,
        value,
      }));

      ema20Series.setData(ema20Data);
    }
  }, [activeIndicators, indicators, data, chartRef.current]);

  // Calculate current price change
  const getCurrentPriceChange = () => {
    if (!data.length) return { change: 0, percent: 0, isPositive: true };

    const current = data[data.length - 1].close;
    const previous = data[0].open;
    const change = current - previous;
    const percent = (change / previous) * 100;

    return {
      change,
      percent,
      isPositive: change >= 0,
    };
  };

  const priceChange = getCurrentPriceChange();

  // Toggle indicator
  const toggleIndicator = (indicator: string) => {
    setActiveIndicators((prev) =>
      prev.includes(indicator)
        ? prev.filter((i) => i !== indicator)
        : [...prev, indicator]
    );
  };

  // Export chart as image
  const exportChart = () => {
    if (!chartContainerRef.current) return;

    const canvas = chartContainerRef.current.querySelector('canvas');
    if (canvas) {
      const url = canvas.toDataURL('image/png');
      const link = document.createElement('a');
      link.download = `${symbol}_${timeframe}_chart.png`;
      link.href = url;
      link.click();
    }
  };

  // Zoom controls
  const zoomIn = () => {
    chartRef.current?.timeScale().scrollToPosition(5, true);
  };

  const zoomOut = () => {
    chartRef.current?.timeScale().scrollToPosition(-5, true);
  };

  const resetZoom = () => {
    chartRef.current?.timeScale().fitContent();
  };

  return (
    <div className={`relative ${isFullscreen ? 'fixed inset-0 z-50 bg-gray-900' : ''}`}>
      {/* Header */}
      <div className="bg-gray-900 border-b border-gray-800 p-4">
        <div className="flex items-center justify-between">
          {/* Symbol and Price */}
          <div className="flex items-center space-x-4">
            <div>
              <h3 className="text-xl font-bold text-white">{symbol}</h3>
              <div className="flex items-center space-x-2">
                <span className="text-2xl font-bold text-white">
                  ${data.length ? data[data.length - 1].close.toFixed(2) : '0.00'}
                </span>
                <div
                  className={`flex items-center space-x-1 px-2 py-1 rounded ${
                    priceChange.isPositive ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'
                  }`}
                >
                  {priceChange.isPositive ? (
                    <TrendingUp className="w-4 h-4" />
                  ) : (
                    <TrendingDown className="w-4 h-4" />
                  )}
                  <span className="text-sm font-medium">
                    {priceChange.isPositive ? '+' : ''}
                    {priceChange.change.toFixed(2)} ({priceChange.percent.toFixed(2)}%)
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Timeframe Selector */}
          <div className="flex items-center space-x-2">
            {timeframes.map((tf) => (
              <button
                key={tf}
                onClick={() => onTimeframeChange?.(tf)}
                className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                  timeframe === tf
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-white'
                }`}
              >
                {tf}
              </button>
            ))}
          </div>

          {/* Controls */}
          <div className="flex items-center space-x-2">
            {/* Zoom Controls */}
            <button
              onClick={zoomIn}
              className="p-2 rounded bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-white transition-colors"
              title="Zoom In"
            >
              <Plus className="w-4 h-4" />
            </button>
            <button
              onClick={zoomOut}
              className="p-2 rounded bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-white transition-colors"
              title="Zoom Out"
            >
              <Minus className="w-4 h-4" />
            </button>
            <button
              onClick={resetZoom}
              className="px-3 py-2 rounded bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-white transition-colors text-sm"
              title="Reset Zoom"
            >
              Reset
            </button>

            {/* Export */}
            <button
              onClick={exportChart}
              className="p-2 rounded bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-white transition-colors"
              title="Export Chart"
            >
              <Download className="w-4 h-4" />
            </button>

            {/* Fullscreen */}
            <button
              onClick={() => setIsFullscreen(!isFullscreen)}
              className="p-2 rounded bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-white transition-colors"
              title="Fullscreen"
            >
              <Maximize2 className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Indicator Toggle */}
        {showIndicators && (
          <div className="flex items-center space-x-2 mt-3">
            <Settings className="w-4 h-4 text-gray-400" />
            <span className="text-sm text-gray-400">Indicators:</span>
            {['SMA20', 'SMA50', 'EMA20', 'Volume'].map((indicator) => (
              <button
                key={indicator}
                onClick={() => toggleIndicator(indicator)}
                className={`px-2 py-1 rounded text-xs font-medium transition-colors ${
                  activeIndicators.includes(indicator)
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                {indicator}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Chart Container */}
      <div ref={chartContainerRef} className="bg-gray-900" />

      {/* Footer - Market Info */}
      <div className="bg-gray-900 border-t border-gray-800 p-3">
        <div className="grid grid-cols-4 gap-4 text-sm">
          <div>
            <span className="text-gray-500">Open:</span>
            <span className="ml-2 text-white font-medium">
              ${data.length ? data[0].open.toFixed(2) : '0.00'}
            </span>
          </div>
          <div>
            <span className="text-gray-500">High:</span>
            <span className="ml-2 text-white font-medium">
              ${data.length ? Math.max(...data.map((d) => d.high)).toFixed(2) : '0.00'}
            </span>
          </div>
          <div>
            <span className="text-gray-500">Low:</span>
            <span className="ml-2 text-white font-medium">
              ${data.length ? Math.min(...data.map((d) => d.low)).toFixed(2) : '0.00'}
            </span>
          </div>
          <div>
            <span className="text-gray-500">Volume:</span>
            <span className="ml-2 text-white font-medium">
              {data.length && data[0].volume
                ? (data.reduce((sum, d) => sum + (d.volume || 0), 0) / 1e6).toFixed(2) + 'M'
                : 'N/A'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TradingViewChart;

