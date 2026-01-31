import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType } from 'lightweight-charts';
import type {
  IChartApi,
  ISeriesApi,
  CandlestickData,
  Time,
  MouseEventParams,
} from 'lightweight-charts';
import { 
  TrendingUp, 
  TrendingDown, 
  Maximize2, 
  Download, 
  Settings,
  Plus,
  Minus,
  Minus as TrendLine,
  Move,
  X,
  Trash2,
} from 'lucide-react';

interface OHLCData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface DrawingLine {
  id: string;
  type: 'trendline' | 'horizontal' | 'vertical';
  points: Array<{ time: number; price: number }>;
  color: string;
  width: number;
}

interface AdvancedTradingChartProps {
  data: OHLCData[];
  symbol: string;
  timeframe?: string;
  height?: number;
  onTimeframeChange?: (timeframe: string) => void;
}

const AdvancedTradingChart: React.FC<AdvancedTradingChartProps> = ({
  data,
  symbol,
  timeframe = '1D',
  height = 600,
  onTimeframeChange,
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);

  const [drawingMode, setDrawingMode] = useState<'none' | 'trendline' | 'horizontal' | 'support' | 'resistance'>('none');
  const [drawings, setDrawings] = useState<DrawingLine[]>([]);
  const [tempDrawing, setTempDrawing] = useState<{ time: number; price: number } | null>(null);
  const [activeIndicators, setActiveIndicators] = useState<string[]>(['SMA20', 'Volume']);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showDrawingTools, setShowDrawingTools] = useState(false);

  const timeframes = ['1m', '5m', '15m', '1h', '4h', '1D', '1W', '1M'];

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#0B1120' },
        textColor: '#9CA3AF',
      },
      grid: {
        vertLines: { color: '#1F2937', style: 1 },
        horzLines: { color: '#1F2937', style: 1 },
      },
      width: chartContainerRef.current.clientWidth,
      height: height,
      rightPriceScale: {
        borderColor: '#374151',
        scaleMargins: {
          top: 0.1,
          bottom: 0.3,
        },
      },
      timeScale: {
        borderColor: '#374151',
        timeVisible: true,
        secondsVisible: false,
        borderVisible: true,
      },
      crosshair: {
        mode: 1,
        vertLine: {
          width: 1,
          color: '#6B7280',
          style: 3,
          labelBackgroundColor: '#3B82F6',
        },
        horzLine: {
          width: 1,
          color: '#6B7280',
          style: 3,
          labelBackgroundColor: '#3B82F6',
        },
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: true,
      },
      handleScale: {
        axisPressedMouseMove: true,
        mouseWheel: true,
        pinch: true,
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

    // Handle chart clicks for drawing
    chart.subscribeClick((param: MouseEventParams) => {
      if (drawingMode === 'none') return;

      const price = candlestickSeries.coordinateToPrice(param.point?.y || 0);
      const time = chart.timeScale().coordinateToTime(param.point?.x || 0);

      if (price && time) {
        handleDrawingClick(Number(time), price as number);
      }
    });

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
  }, [height]);

  // Update data
  useEffect(() => {
    if (!candlestickSeriesRef.current || !data.length) return;

    const candleData: CandlestickData[] = data.map((d) => ({
      time: (d.time / 1000) as Time,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
    }));

    candlestickSeriesRef.current.setData(candleData);

    if (volumeSeriesRef.current && data[0]?.volume !== undefined) {
      const volumeData = data.map((d, i) => ({
        time: (d.time / 1000) as Time,
        value: d.volume || 0,
        color:
          i > 0 && d.close >= data[i - 1].close
            ? 'rgba(16, 185, 129, 0.4)'
            : 'rgba(239, 68, 68, 0.4)',
      }));

      volumeSeriesRef.current.setData(volumeData);
    }

    chartRef.current?.timeScale().fitContent();
  }, [data]);

  // Add moving averages
  useEffect(() => {
    if (!chartRef.current || !data.length) return;

    if (activeIndicators.includes('SMA20')) {
      const sma20 = calculateSMA(data.map(d => d.close), 20);
      const sma20Series = chartRef.current.addLineSeries({
        color: '#3B82F6',
        lineWidth: 2,
        title: 'SMA 20',
      });

      sma20Series.setData(
        sma20.map((value, i) => ({
          time: (data[i].time / 1000) as Time,
          value,
        })).filter(d => d.value !== null) as any
      );
    }

    if (activeIndicators.includes('SMA50')) {
      const sma50 = calculateSMA(data.map(d => d.close), 50);
      const sma50Series = chartRef.current.addLineSeries({
        color: '#8B5CF6',
        lineWidth: 2,
        title: 'SMA 50',
      });

      sma50Series.setData(
        sma50.map((value, i) => ({
          time: (data[i].time / 1000) as Time,
          value,
        })).filter(d => d.value !== null) as any
      );
    }

    if (activeIndicators.includes('EMA20')) {
      const ema20 = calculateEMA(data.map(d => d.close), 20);
      const ema20Series = chartRef.current.addLineSeries({
        color: '#F59E0B',
        lineWidth: 2,
        title: 'EMA 20',
        lineStyle: 2,
      });

      ema20Series.setData(
        ema20.map((value, i) => ({
          time: (data[i].time / 1000) as Time,
          value,
        })).filter(d => d.value !== null) as any
      );
    }
  }, [activeIndicators, data]);

  // Drawing tools logic
  const handleDrawingClick = (time: number, price: number) => {
    if (!tempDrawing) {
      // First point
      setTempDrawing({ time, price });
    } else {
      // Second point - complete drawing
      const newDrawing: DrawingLine = {
        id: `drawing_${Date.now()}`,
        type: drawingMode === 'horizontal' ? 'horizontal' : 'trendline',
        points: [tempDrawing, { time, price }],
        color: drawingMode === 'support' ? '#10B981' : drawingMode === 'resistance' ? '#EF4444' : '#3B82F6',
        width: 2,
      };

      setDrawings([...drawings, newDrawing]);
      setTempDrawing(null);
      setDrawingMode('none');
    }
  };

  const clearDrawings = () => {
    setDrawings([]);
    setTempDrawing(null);
    setDrawingMode('none');
  };

  const deleteDrawing = (id: string) => {
    setDrawings(drawings.filter(d => d.id !== id));
  };

  // Technical calculations
  const calculateSMA = (prices: number[], period: number): (number | null)[] => {
    const result: (number | null)[] = [];
    for (let i = 0; i < prices.length; i++) {
      if (i < period - 1) {
        result.push(null);
      } else {
        const sum = prices.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
        result.push(sum / period);
      }
    }
    return result;
  };

  const calculateEMA = (prices: number[], period: number): (number | null)[] => {
    const result: (number | null)[] = [];
    const multiplier = 2 / (period + 1);
    let ema = prices[0];

    for (let i = 0; i < prices.length; i++) {
      if (i === 0) {
        result.push(prices[i]);
        ema = prices[i];
      } else {
        ema = (prices[i] - ema) * multiplier + ema;
        result.push(ema);
      }
    }
    return result;
  };

  // Price change calculation
  const getCurrentPriceChange = () => {
    if (!data.length) return { change: 0, percent: 0, isPositive: true };

    const current = data[data.length - 1].close;
    const previous = data[0].open;
    const change = current - previous;
    const percent = (change / previous) * 100;

    return { change, percent, isPositive: change >= 0 };
  };

  const priceChange = getCurrentPriceChange();

  const toggleIndicator = (indicator: string) => {
    setActiveIndicators((prev) =>
      prev.includes(indicator) ? prev.filter((i) => i !== indicator) : [...prev, indicator]
    );
  };

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

  return (
    <div className={`relative ${isFullscreen ? 'fixed inset-0 z-50' : ''} bg-gray-900`}>
      {/* Header */}
      <div className="bg-gray-900 border-b border-gray-800 p-4">
        <div className="flex items-center justify-between mb-3">
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
                  {priceChange.isPositive ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
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
            <button
              onClick={exportChart}
              className="p-2 rounded bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-white transition-colors"
              title="Export Chart"
            >
              <Download className="w-4 h-4" />
            </button>

            <button
              onClick={() => setIsFullscreen(!isFullscreen)}
              className="p-2 rounded bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-white transition-colors"
              title="Fullscreen"
            >
              <Maximize2 className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Drawing Tools */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowDrawingTools(!showDrawingTools)}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                showDrawingTools ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              <Move className="w-4 h-4" />
              <span>Drawing Tools</span>
            </button>

            {showDrawingTools && (
              <>
                <button
                  onClick={() => setDrawingMode(drawingMode === 'trendline' ? 'none' : 'trendline')}
                  className={`flex items-center space-x-1 px-3 py-1.5 rounded text-sm transition-colors ${
                    drawingMode === 'trendline' ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                  }`}
                  title="Draw Trend Line"
                >
                  <TrendLine className="w-4 h-4" />
                  <span>Trend Line</span>
                </button>

                <button
                  onClick={() => setDrawingMode(drawingMode === 'support' ? 'none' : 'support')}
                  className={`px-3 py-1.5 rounded text-sm transition-colors ${
                    drawingMode === 'support' ? 'bg-green-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                  }`}
                  title="Draw Support Level"
                >
                  Support
                </button>

                <button
                  onClick={() => setDrawingMode(drawingMode === 'resistance' ? 'none' : 'resistance')}
                  className={`px-3 py-1.5 rounded text-sm transition-colors ${
                    drawingMode === 'resistance' ? 'bg-red-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                  }`}
                  title="Draw Resistance Level"
                >
                  Resistance
                </button>

                {drawings.length > 0 && (
                  <button
                    onClick={clearDrawings}
                    className="flex items-center space-x-1 px-3 py-1.5 rounded text-sm bg-red-900/30 text-red-400 hover:bg-red-900/50 transition-colors"
                    title="Clear All Drawings"
                  >
                    <Trash2 className="w-4 h-4" />
                    <span>Clear ({drawings.length})</span>
                  </button>
                )}
              </>
            )}
          </div>

          {/* Indicators */}
          <div className="flex items-center space-x-2">
            <Settings className="w-4 h-4 text-gray-400" />
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
        </div>

        {tempDrawing && (
          <div className="mt-2 bg-blue-900/20 border border-blue-800 rounded p-2 text-sm text-blue-400">
            📍 Click on the chart to complete the drawing
          </div>
        )}
      </div>

      {/* Chart Container */}
      <div ref={chartContainerRef} className="bg-gray-900" />

      {/* Footer - Market Info */}
      <div className="bg-gray-900 border-t border-gray-800 p-3">
        <div className="grid grid-cols-5 gap-4 text-sm">
          <div>
            <span className="text-gray-500">Open:</span>
            <span className="ml-2 text-white font-medium">
              ${data.length ? data[0].open.toFixed(2) : '0.00'}
            </span>
          </div>
          <div>
            <span className="text-gray-500">High:</span>
            <span className="ml-2 text-green-400 font-medium">
              ${data.length ? Math.max(...data.map((d) => d.high)).toFixed(2) : '0.00'}
            </span>
          </div>
          <div>
            <span className="text-gray-500">Low:</span>
            <span className="ml-2 text-red-400 font-medium">
              ${data.length ? Math.min(...data.map((d) => d.low)).toFixed(2) : '0.00'}
            </span>
          </div>
          <div>
            <span className="text-gray-500">Close:</span>
            <span className="ml-2 text-white font-medium">
              ${data.length ? data[data.length - 1].close.toFixed(2) : '0.00'}
            </span>
          </div>
          <div>
            <span className="text-gray-500">24h Volume:</span>
            <span className="ml-2 text-white font-medium">
              {data.length && data[0].volume
                ? '$' + (data.reduce((sum, d) => sum + (d.volume || 0), 0) / 1e6).toFixed(2) + 'M'
                : 'N/A'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdvancedTradingChart;

