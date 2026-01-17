import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { getCryptoDisplayName, getCryptoSymbol, formatPrice } from '../hooks/useCryptoPrices';
import { formatPercentage, formatMarketCap, formatVolume } from '../hooks/useMarketData';
import CryptoIcon from './CryptoIcon';
import { API_BASE_URL } from '../lib/api';
import AdvancedTradingChart from './AdvancedTradingChart';

interface CoinDetailViewProps {
  coinId: string;
  currentData: {
    price: number;
    change_24h: number;
    market_cap: number;
    volume_24h: number;
  };
  onClose: () => void;
}

interface OHLCData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
}

type TimeRange = '1' | '7' | '30' | '90' | '365';
type ChartType = 'line' | 'area' | 'candlestick';

const TIME_RANGES: { value: TimeRange; label: string }[] = [
  { value: '1', label: '24H' },
  { value: '7', label: '7D' },
  { value: '30', label: '1M' },
  { value: '90', label: '3M' },
  { value: '365', label: '1Y' },
];

const CHART_TYPES: { value: ChartType; label: string; icon: string }[] = [
  { value: 'line', label: 'Line', icon: '📈' },
  { value: 'area', label: 'Area', icon: '📊' },
  { value: 'candlestick', label: 'Candles', icon: '🕯️' },
];

function CoinDetailView({ coinId, currentData, onClose }: CoinDetailViewProps) {
  const [timeRange, setTimeRange] = useState<TimeRange>('7');
  const [chartType, setChartType] = useState<ChartType>('area');

  const { data: historyData, isLoading, error } = useQuery({
    queryKey: ['coin-history', coinId, timeRange],
    queryFn: async () => {
      const response = await fetch(
        `${API_BASE_URL}/prices/history/${coinId}?days=${timeRange}&vs_currency=usd`
      );
      if (!response.ok) throw new Error('Failed to fetch history');
      return response.json();
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchInterval: 60 * 1000, // Refetch every minute
  });

  // Format data for charts
  const chartData = historyData?.data?.map((item: OHLCData) => ({
    time: new Date(item.timestamp).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      ...(timeRange === '1' ? { hour: '2-digit' } : {}),
    }),
    price: item.close,
    open: item.open,
    high: item.high,
    low: item.low,
    close: item.close,
  })) || [];
  
  // Debug logging
  if (historyData && chartData.length === 0) {
    console.log('[CoinDetailView] History data received but chartData is empty', {
      historyData,
      hasData: !!historyData?.data,
      dataLength: historyData?.data?.length
    });
  }

  // Calculate statistics
  const stats = chartData.length > 0 ? {
    highest: Math.max(...chartData.map((d: any) => d.high)),
    lowest: Math.min(...chartData.map((d: any) => d.low)),
    average: chartData.reduce((sum: number, d: any) => sum + d.close, 0) / chartData.length,
    change: ((chartData[chartData.length - 1]?.close - chartData[0]?.close) / chartData[0]?.close) * 100,
  } : null;

  const isPositive = (currentData?.change_24h || 0) >= 0;

  return (
    <div className="market-card animate-fade-in-up">
      {/* Header */}
      <div className="flex items-center justify-between mb-10">
        <div className="flex items-center space-x-6">
          <CryptoIcon cryptoId={coinId} size="xl" />
          <div>
            <h2 className="text-5xl font-black text-gray-900">
              {getCryptoDisplayName(coinId)}
            </h2>
            <p className="text-gray-500 text-2xl font-bold mt-2">
              {getCryptoSymbol(coinId)} • Real-Time Data
            </p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="coinbase-button-secondary px-8 py-4 text-lg hover:bg-gray-50 transition-all duration-300"
        >
          ✕ Close
        </button>
      </div>

      {/* Current Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-10">
        <div className="text-center p-8 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-3xl border-2 border-blue-100 hover:border-blue-200 transition-all duration-300">
          <div className="text-sm text-gray-600 font-black mb-3 uppercase tracking-wider">Current Price</div>
          <div className="text-4xl font-black text-gray-900">
            {formatPrice(currentData.price)}
          </div>
        </div>

        <div className={`text-center p-8 rounded-3xl border-2 transition-all duration-300 ${
          isPositive
            ? 'bg-gradient-to-br from-green-50 to-emerald-50 border-green-100 hover:border-green-200'
            : 'bg-gradient-to-br from-red-50 to-rose-50 border-red-100 hover:border-red-200'
        }`}>
          <div className="text-sm text-gray-600 font-black mb-3 uppercase tracking-wider">24h Change</div>
          <div className={`text-4xl font-black ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
            {isPositive ? '+' : ''}{formatPercentage(currentData.change_24h)}
          </div>
        </div>

        <div className="text-center p-8 bg-gradient-to-br from-purple-50 to-pink-50 rounded-3xl border-2 border-purple-100 hover:border-purple-200 transition-all duration-300">
          <div className="text-sm text-gray-600 font-black mb-3 uppercase tracking-wider">Market Cap</div>
          <div className="text-4xl font-black text-gray-900">
            {formatMarketCap(currentData.market_cap)}
          </div>
        </div>

        <div className="text-center p-8 bg-gradient-to-br from-orange-50 to-yellow-50 rounded-3xl border-2 border-orange-100 hover:border-orange-200 transition-all duration-300">
          <div className="text-sm text-gray-600 font-black mb-3 uppercase tracking-wider">Volume 24h</div>
          <div className="text-4xl font-black text-gray-900">
            {formatVolume(currentData.volume_24h)}
          </div>
        </div>
      </div>

      {/* Chart Controls */}
      <div className="flex flex-wrap items-center justify-between mb-8 gap-6">
        {/* Time Range Selector */}
        <div className="flex items-center space-x-3">
          <span className="text-gray-600 font-bold text-lg mr-2">Time Range:</span>
          {TIME_RANGES.map((range) => (
            <button
              key={range.value}
              onClick={() => setTimeRange(range.value)}
              className={`px-6 py-3 rounded-xl font-black text-base transition-all duration-300 ${
                timeRange === range.value
                  ? 'bg-blue-600 text-white shadow-lg scale-105'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {range.label}
            </button>
          ))}
        </div>

        {/* Chart Type Selector */}
        <div className="flex items-center space-x-3">
          <span className="text-gray-600 font-bold text-lg mr-2">Chart Type:</span>
          {CHART_TYPES.map((type) => (
            <button
              key={type.value}
              onClick={() => setChartType(type.value)}
              className={`px-6 py-3 rounded-xl font-black text-base transition-all duration-300 flex items-center space-x-2 ${
                chartType === type.value
                  ? 'bg-indigo-600 text-white shadow-lg scale-105'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              <span>{type.icon}</span>
              <span>{type.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Chart Area */}
      <div className="bg-white rounded-3xl p-8 shadow-xl border-2 border-gray-100">
        {isLoading ? (
          <div className="flex items-center justify-center h-96">
            <div className="text-center">
              <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-600 mx-auto mb-4"></div>
              <p className="text-gray-600 font-bold text-lg">Loading chart data...</p>
            </div>
          </div>
        ) : error ? (
          <div className="flex items-center justify-center h-96">
            <div className="text-center text-red-600">
              <svg className="w-16 h-16 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="font-bold text-lg">Failed to load chart data</p>
            </div>
          </div>
        ) : chartData.length === 0 ? (
          <div className="flex items-center justify-center h-96">
            <p className="text-gray-500 font-bold text-lg">No data available</p>
          </div>
        ) : (
          <>
            {/* Period Statistics */}
            {stats && (
              <div className="grid grid-cols-4 gap-6 mb-8 pb-8 border-b-2 border-gray-100">
                <div className="text-center">
                  <div className="text-sm text-gray-500 font-bold uppercase mb-2">Highest</div>
                  <div className="text-2xl font-black text-green-600">{formatPrice(stats.highest)}</div>
                </div>
                <div className="text-center">
                  <div className="text-sm text-gray-500 font-bold uppercase mb-2">Lowest</div>
                  <div className="text-2xl font-black text-red-600">{formatPrice(stats.lowest)}</div>
                </div>
                <div className="text-center">
                  <div className="text-sm text-gray-500 font-bold uppercase mb-2">Average</div>
                  <div className="text-2xl font-black text-gray-900">{formatPrice(stats.average)}</div>
                </div>
                <div className="text-center">
                  <div className="text-sm text-gray-500 font-bold uppercase mb-2">Period Change</div>
                  <div className={`text-2xl font-black ${stats.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {stats.change >= 0 ? '+' : ''}{stats.change.toFixed(2)}%
                  </div>
                </div>
              </div>
            )}

            {/* Chart */}
            {chartType === 'candlestick' && historyData?.data && historyData.data.length > 0 ? (
              <div className="-mx-8 -mb-8">
                <AdvancedTradingChart
                  data={historyData.data.map((d: OHLCData) => ({
                    time: d.timestamp,
                    open: d.open,
                    high: d.high,
                    low: d.low,
                    close: d.close,
                    volume: Math.random() * 1000000000, // Mock volume for now
                  }))}
                  symbol={`${getCryptoDisplayName(coinId)} (${getCryptoSymbol(coinId)})`}
                  timeframe={timeRange === '1' ? '1h' : timeRange === '7' ? '4h' : '1D'}
                  height={500}
                />
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={400}>
                {chartType === 'line' ? (
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis
                      dataKey="time"
                      stroke="#6b7280"
                      style={{ fontSize: '14px', fontWeight: 'bold' }}
                    />
                    <YAxis
                      stroke="#6b7280"
                      style={{ fontSize: '14px', fontWeight: 'bold' }}
                      domain={['auto', 'auto']}
                      tickFormatter={(value) => `$${value.toLocaleString()}`}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#ffffff',
                        border: '2px solid #e5e7eb',
                        borderRadius: '12px',
                        padding: '12px',
                        fontWeight: 'bold',
                      }}
                      formatter={(value: any) => [`$${value.toLocaleString()}`, 'Price']}
                    />
                    <Line
                      type="monotone"
                      dataKey="price"
                      stroke="#3b82f6"
                      strokeWidth={3}
                      dot={false}
                      activeDot={{ r: 6, strokeWidth: 2 }}
                    />
                  </LineChart>
                ) : chartType === 'area' ? (
                  <AreaChart data={chartData}>
                    <defs>
                      <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8} />
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.1} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis
                      dataKey="time"
                      stroke="#6b7280"
                      style={{ fontSize: '14px', fontWeight: 'bold' }}
                    />
                    <YAxis
                      stroke="#6b7280"
                      style={{ fontSize: '14px', fontWeight: 'bold' }}
                      domain={['auto', 'auto']}
                      tickFormatter={(value) => `$${value.toLocaleString()}`}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#ffffff',
                        border: '2px solid #e5e7eb',
                        borderRadius: '12px',
                        padding: '12px',
                        fontWeight: 'bold',
                      }}
                      formatter={(value: any) => [`$${value.toLocaleString()}`, 'Price']}
                    />
                    <Area
                      type="monotone"
                      dataKey="price"
                      stroke="#3b82f6"
                      strokeWidth={3}
                      fillOpacity={1}
                      fill="url(#colorPrice)"
                    />
                  </AreaChart>
                ) : (
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis
                      dataKey="time"
                      stroke="#6b7280"
                      style={{ fontSize: '14px', fontWeight: 'bold' }}
                    />
                    <YAxis
                      stroke="#6b7280"
                      style={{ fontSize: '14px', fontWeight: 'bold' }}
                      domain={['auto', 'auto']}
                      tickFormatter={(value) => `$${value.toLocaleString()}`}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#ffffff',
                        border: '2px solid #e5e7eb',
                        borderRadius: '12px',
                        padding: '12px',
                      fontWeight: 'bold',
                    }}
                    formatter={(value: any, name: string) => [
                      `$${value.toLocaleString()}`,
                      name.charAt(0).toUpperCase() + name.slice(1),
                    ]}
                  />
                  <Legend
                    wrapperStyle={{ fontSize: '14px', fontWeight: 'bold' }}
                  />
                  <Bar dataKey="high" fill="#10b981" name="High" />
                  <Bar dataKey="low" fill="#ef4444" name="Low" />
                  <Bar dataKey="close" fill="#3b82f6" name="Close" />
                </BarChart>
                )}
              </ResponsiveContainer>
            )}
          </>
        )}
      </div>

      {/* Additional Info */}
      <div className="mt-10 grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="p-8 bg-gradient-to-br from-gray-50 to-gray-100 rounded-3xl border-2 border-gray-200">
          <h3 className="text-2xl font-black text-gray-900 mb-6 flex items-center">
            <svg className="w-7 h-7 mr-3 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
            </svg>
            Market Data
          </h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-gray-600 font-bold">Market Rank</span>
              <span className="text-gray-900 font-black text-lg">#{['bitcoin', 'ethereum', 'binancecoin', 'solana', 'cardano', 'polkadot'].indexOf(coinId) + 1}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600 font-bold">24h Volume / Market Cap</span>
              <span className="text-gray-900 font-black text-lg">
                {((currentData.volume_24h / currentData.market_cap) * 100).toFixed(2)}%
              </span>
            </div>
          </div>
        </div>

        <div className="p-8 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-3xl border-2 border-blue-200">
          <h3 className="text-2xl font-black text-gray-900 mb-6 flex items-center">
            <svg className="w-7 h-7 mr-3 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            Trading Info
          </h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-gray-600 font-bold">24h High</span>
              <span className="text-green-600 font-black text-lg">{stats ? formatPrice(stats.highest) : 'N/A'}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600 font-bold">24h Low</span>
              <span className="text-red-600 font-black text-lg">{stats ? formatPrice(stats.lowest) : 'N/A'}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default CoinDetailView;

