import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../lib/api';
import { useToast } from '../hooks/use-toast';
import { formatPrice, getCryptoDisplayName, getCryptoSymbol } from '../hooks/useCryptoPrices';
import { XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts';
import MiniSparkline from './MiniSparkline';
import ConfidenceIndicator from './ConfidenceIndicator';
import TechnicalSignalIndicator from './TechnicalSignalIndicator';
import CryptoIcon from './CryptoIcon';

interface ModelMetricsData {
  models: Record<string, {
    label: string;
    description: string;
    type: string;
    available_coins: string[];
    avg_directional_accuracy_1d: number | null;
    per_coin: Record<string, {
      val_rmse_1d?: number;
      directional_accuracy?: Record<string, number>;
      test_rmse?: number;
      test_mae?: number;
    }>;
  }>;
  ml_available: boolean;
}

interface ForecastData {
  forecasts: Record<string, {
    model: string;
    current_price: number;
    generated_at: string;
    forecast_horizon_days: number;
    forecasts: Array<{
      date: string;
      predicted_price: number;
      confidence_lower: number;
      confidence_upper: number;
      confidence: number;
      technical_signals?: {
        rsi: number;
        regime: string;
        macd?: number;
        bollinger_position?: number;
        volatility?: number;
      };
    }>;
    model_metrics: {
      mape: number;
      rmse: number;
      r_squared: number;
    };
    status: string;
    note: string;
    historical_data?: Array<{
      date: string;
      price: number;
      is_historical: boolean;
    }>;
  }>;
  metadata: {
    generated_at: string;
    model: string;
    forecast_horizon: number;
    total_assets: number;
  };
}

const FORECAST_MODELS = [
  {
    value: 'lightgbm',
    label: 'LightGBM',
    description: 'Gradient boosting with 150+ engineered features',
    accuracy: '53%',
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    )
  },
  {
    value: 'lstm',
    label: 'LSTM',
    description: 'Bidirectional LSTM with attention mechanism',
    accuracy: '52%',
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
      </svg>
    )
  },
  {
    value: 'transformer',
    label: 'Transformer',
    description: 'Multi-head self-attention with causal masking',
    accuracy: '54%',
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    )
  },
  {
    value: 'tcn',
    label: 'TCN',
    description: 'Temporal Convolutional Network with dilated causal convolutions',
    accuracy: '53%',
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
      </svg>
    )
  },
  {
    value: 'dlinear',
    label: 'DLinear',
    description: 'Trend/seasonal decomposition baseline model',
    accuracy: '51%',
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
      </svg>
    )
  },
  {
    value: 'ml_ensemble',
    label: 'ML Ensemble',
    description: 'Weighted average of all deep learning models',
    accuracy: '54%',
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
      </svg>
    )
  },
];

const ALL_CRYPTO_OPTIONS = [
  // Major Cryptocurrencies
  { id: 'bitcoin', name: 'Bitcoin', symbol: 'BTC', category: 'Major' },
  { id: 'ethereum', name: 'Ethereum', symbol: 'ETH', category: 'Major' },
  { id: 'tether', name: 'Tether', symbol: 'USDT', category: 'Major' },
  { id: 'binancecoin', name: 'BNB', symbol: 'BNB', category: 'Major' },
  { id: 'ripple', name: 'XRP', symbol: 'XRP', category: 'Major' },
  { id: 'usd-coin', name: 'USD Coin', symbol: 'USDC', category: 'Major' },
  
  // Layer 1 & Smart Contract Platforms
  { id: 'solana', name: 'Solana', symbol: 'SOL', category: 'L1' },
  { id: 'cardano', name: 'Cardano', symbol: 'ADA', category: 'L1' },
  { id: 'avalanche-2', name: 'Avalanche', symbol: 'AVAX', category: 'L1' },
  { id: 'polkadot', name: 'Polkadot', symbol: 'DOT', category: 'L1' },
  { id: 'near', name: 'NEAR Protocol', symbol: 'NEAR', category: 'L1' },
  { id: 'cosmos', name: 'Cosmos', symbol: 'ATOM', category: 'L1' },
  
  // DeFi Tokens
  { id: 'uniswap', name: 'Uniswap', symbol: 'UNI', category: 'DeFi' },
  { id: 'chainlink', name: 'Chainlink', symbol: 'LINK', category: 'DeFi' },
  { id: 'aave', name: 'Aave', symbol: 'AAVE', category: 'DeFi' },
  
  // Layer 2 & Scaling
  { id: 'matic-network', name: 'Polygon', symbol: 'MATIC', category: 'L2' },
  { id: 'optimism', name: 'Optimism', symbol: 'OP', category: 'L2' },
  { id: 'arbitrum', name: 'Arbitrum', symbol: 'ARB', category: 'L2' },
  
  // Meme & Community
  { id: 'dogecoin', name: 'Dogecoin', symbol: 'DOGE', category: 'Meme' },
  
  // Infrastructure & Web3
  { id: 'filecoin', name: 'Filecoin', symbol: 'FIL', category: 'Web3' },
  { id: 'litecoin', name: 'Litecoin', symbol: 'LTC', category: 'Major' },
  { id: 'bitcoin-cash', name: 'Bitcoin Cash', symbol: 'BCH', category: 'Major' },
  { id: 'stellar', name: 'Stellar', symbol: 'XLM', category: 'L1' },
  { id: 'ethereum-classic', name: 'Ethereum Classic', symbol: 'ETC', category: 'L1' },
  { id: 'monero', name: 'Monero', symbol: 'XMR', category: 'Major' },
  { id: 'algorand', name: 'Algorand', symbol: 'ALGO', category: 'L1' },
  { id: 'the-sandbox', name: 'The Sandbox', symbol: 'SAND', category: 'Web3' },
  { id: 'axie-infinity', name: 'Axie Infinity', symbol: 'AXS', category: 'Web3' },
  { id: 'decentraland', name: 'Decentraland', symbol: 'MANA', category: 'Web3' },
  { id: 'theta-token', name: 'Theta Network', symbol: 'THETA', category: 'Web3' },
  { id: 'fantom', name: 'Fantom', symbol: 'FTM', category: 'L1' },
];

// Default: bitcoin only for fast initial load (~16s vs 80s for all 5)
// Users can add more via "Customize" button
const DEFAULT_CRYPTOS = [
  'bitcoin',      // BTC
];

function ForecastPanel() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [selectedModel, setSelectedModel] = useState('lstm');
  const [forecastDays, setForecastDays] = useState(7);
  const [selectedCrypto, setSelectedCrypto] = useState<string | null>(null);
  const [selectedCryptos, setSelectedCryptos] = useState<string[]>(DEFAULT_CRYPTOS);
  const [showCryptoSelector, setShowCryptoSelector] = useState(false);
  const [filterCategory, setFilterCategory] = useState<string>('All');

  const { data: forecastData, isLoading, error, refetch } = useQuery<ForecastData>({
    queryKey: ['forecasts', selectedCryptos.join(','), forecastDays, selectedModel],
    queryFn: () => apiClient.getForecasts(selectedCryptos, forecastDays, selectedModel),
    refetchInterval: 900000, // 15 minutes
    staleTime: 900000, // 15 minutes
  });

  // Fetch real-time prices for live updates (like the market panel does)
  const { data: realTimePrices } = useQuery({
    queryKey: ['realtime-prices', selectedCryptos.join(',')],
    queryFn: async () => {
      const response = await apiClient.getMultiplePrices(selectedCryptos);
      return response;
    },
    refetchInterval: 60000, // Reduce API usage
    staleTime: 60000,
    enabled: selectedCryptos.length > 0,
  });

  // Fetch real training metrics for model accuracy display
  const { data: modelMetrics } = useQuery<ModelMetricsData>({
    queryKey: ['model-metrics'],
    queryFn: () => apiClient.getModelMetrics(),
    staleTime: 3600000, // 1 hour
  });

  // Helper functions
  const toggleCrypto = (cryptoId: string) => {
    setSelectedCryptos(prev => 
      prev.includes(cryptoId) 
        ? prev.filter(id => id !== cryptoId)
        : [...prev, cryptoId]
    );
  };

  const getModelAccuracy = (model: string) => {
    const dynamic = modelMetrics?.models?.[model]?.avg_directional_accuracy_1d;
    if (dynamic != null) {
      return `${Math.round(dynamic * 100)}% DA`;
    }
    return FORECAST_MODELS.find(m => m.value === model)?.accuracy || 'N/A';
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600 dark:text-green-400 bg-green-100 dark:bg-green-900/30';
    if (confidence >= 0.6) return 'text-yellow-600 dark:text-yellow-400 bg-yellow-100 dark:bg-yellow-900/30';
    return 'text-red-600 dark:text-red-400 bg-red-100 dark:bg-red-900/30';
  };

  // Handler for adding to watchlist
  const handleAddToWatchlist = async (e: React.MouseEvent, crypto: any) => {
    e.stopPropagation();

    try {
      const watchlistData = {
        crypto_symbol: getCryptoSymbol(crypto.id),
        crypto_name: getCryptoDisplayName(crypto.id),
        crypto_id: crypto.id,
        notification_enabled: true
      };
      console.log('Adding to watchlist:', watchlistData);
      console.log('Crypto object:', crypto);
      await apiClient.addToWatchlist(watchlistData);

      // Invalidate watchlist cache to refresh the page
      queryClient.invalidateQueries({ queryKey: ['watchlist'] });

      toast({
        title: "Added to Watchlist",
        description: `${getCryptoDisplayName(crypto.id)} has been added to your watchlist.`,
      });
    } catch (error: any) {
      console.error('Error adding to watchlist:', error);
      const errorMessage = error?.message || 'Failed to add to watchlist';
      if (errorMessage.includes('already in your watchlist') || errorMessage.includes('409')) {
        toast({
          title: "Already in Watchlist",
          description: `${getCryptoDisplayName(crypto.id)} is already in your watchlist.`,
          variant: "default",
        });
      } else {
        toast({
          title: "Error",
          description: "Failed to add to watchlist. Please try again.",
          variant: "destructive",
        });
      }
    }
  };

  // Handler for setting alert
  const handleSetAlert = async (e: React.MouseEvent, crypto: any) => {
    e.stopPropagation();

    console.log('Setting alert for crypto:', crypto);

    // Get current price for the alert - fallback to forecast price if real-time fails
    const currentPriceData = realTimePrices?.[crypto.id];
    const realTimePrice = currentPriceData?.usd;
    const forecastPrice = forecastData?.forecasts?.[crypto.id]?.current_price;
    const currentPrice = realTimePrice || forecastPrice || 0;

    console.log('Real time price:', realTimePrice);
    console.log('Forecast price:', forecastPrice);
    console.log('Using current price:', currentPrice);

    if (!currentPrice) {
      toast({
        title: "Error",
        description: "Unable to get current price. Please try again.",
        variant: "destructive",
      });
      return;
    }

    try {
      // Set a basic price alert 5% above current price
      const targetPrice = currentPrice * 1.05;

      await apiClient.createAlert({
        crypto_symbol: getCryptoSymbol(crypto.id),
        crypto_name: getCryptoDisplayName(crypto.id),
        alert_type: 'price_target',
        target_price: targetPrice,
        condition: 'above',
        message: `Alert when ${getCryptoDisplayName(crypto.id)} reaches $${targetPrice.toFixed(2)}`
      });

      // Invalidate alerts cache to refresh any alerts page
      queryClient.invalidateQueries({ queryKey: ['alerts'] });

      toast({
        title: "Alert Set",
        description: `Price alert set for ${getCryptoDisplayName(crypto.id)} at $${targetPrice.toFixed(2)}`,
      });
    } catch (error: any) {
      console.error('Error setting alert:', error);
      toast({
        title: "Error",
        description: "Failed to set alert. Please try again.",
        variant: "destructive",
      });
    }
  };

  if (isLoading) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="market-card">
          <div className="flex flex-col items-center justify-center py-20 space-y-6">
            {/* Animated brain/AI icon */}
            <div className="relative">
              <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-xl">
                <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              {/* Pulsing ring */}
              <div className="absolute inset-0 rounded-2xl border-4 border-blue-400 opacity-50 animate-ping"></div>
            </div>

            <div className="text-center space-y-2">
              <h3 className="text-2xl font-black text-gray-900 dark:text-gray-100">
                Running ML Inference...
              </h3>
              <p className="text-gray-500 dark:text-gray-400 max-w-sm">
                The {selectedModel.toUpperCase()} model is analyzing {selectedCryptos.length > 1 ? `${selectedCryptos.length} assets` : selectedCryptos[0] || 'market data'} in parallel. This typically takes 10–20 seconds.
              </p>
            </div>

            {/* Progress dots */}
            <div className="flex items-center space-x-2">
              {['Fetching prices', 'Running model', 'Building forecast'].map((step, i) => (
                <div key={step} className="flex items-center">
                  <div className="flex items-center space-x-1.5">
                    <div
                      className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"
                      style={{ animationDelay: `${i * 0.15}s` }}
                    ></div>
                    <span className="text-xs text-gray-500 dark:text-gray-400">{step}</span>
                  </div>
                  {i < 2 && <div className="w-4 h-px bg-gray-300 dark:bg-gray-600 mx-2"></div>}
                </div>
              ))}
            </div>

            {/* Model badge */}
            <div className="flex items-center space-x-2 px-4 py-2 bg-blue-50 dark:bg-blue-900/20 rounded-full border border-blue-200 dark:border-blue-700">
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
              <span className="text-sm font-semibold text-blue-700 dark:text-blue-300">
                {FORECAST_MODELS.find(m => m.value === selectedModel)?.label || selectedModel} · {forecastDays}-day forecast
              </span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="market-card border-red-200 dark:border-red-800">
          <div className="text-center py-12">
            <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-red-100 dark:bg-red-900/30 flex items-center justify-center">
              <svg className="w-8 h-8 text-red-600 dark:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-2">Failed to load forecasts</h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">Unable to generate price predictions. This might be due to API rate limiting.</p>
            <div className="flex justify-center space-x-4">
              <button 
                className="bg-blue-600 hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 text-white px-6 py-2 rounded-xl font-semibold text-sm transition-colors"
                onClick={() => refetch()}
              >
                Retry
              </button>
              <button 
                className="bg-gray-600 hover:bg-gray-700 dark:bg-gray-700 dark:hover:bg-gray-600 text-white px-6 py-2 rounded-xl font-semibold text-sm transition-colors"
                onClick={() => window.location.reload()}
              >
                Refresh Page
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const hasForecasts = forecastData?.forecasts && Object.keys(forecastData.forecasts).length > 0;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Controls */}
      <div className="market-card">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-4xl font-black text-gray-900 dark:text-gray-100 mb-3">AI Crypto Price Intelligence</h2>
            <p className="text-lg text-gray-600 dark:text-gray-400">Professional-grade forecasting with real-time technical analysis</p>
          </div>
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-3 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 px-4 py-3 rounded-2xl border border-blue-200 dark:border-blue-700">
              <svg className="w-5 h-5 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              <span className="font-bold text-blue-700 dark:text-blue-400">AI-Powered</span>
            </div>
            <div className="text-right">
              <div className="text-sm text-gray-500 dark:text-gray-400">Model Accuracy</div>
              <div className="text-2xl font-black text-green-600 dark:text-green-400">{getModelAccuracy(selectedModel)}</div>
            </div>
          </div>
        </div>

        {/* Crypto Selector */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100">Cryptocurrency Universe</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">All major cryptocurrencies, DeFi tokens, and altcoins</p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-right">
                <div className="text-sm text-gray-500 dark:text-gray-400">Tracking</div>
                <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">{selectedCryptos.length}</div>
              </div>
              <button
                onClick={() => setShowCryptoSelector(!showCryptoSelector)}
                className="bg-blue-600 hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 text-white px-4 py-2 rounded-xl font-semibold text-sm transition-colors"
              >
                {showCryptoSelector ? 'Hide' : 'Customize'}
              </button>
            </div>
          </div>
          
          {showCryptoSelector && (
            <div className="bg-gray-50 dark:bg-gray-800/50 rounded-2xl p-6 border border-gray-200 dark:border-gray-700">
              {/* Category Filter */}
              <div className="mb-6">
                <div className="flex flex-wrap gap-2">
                  {['All', 'Major', 'L1', 'DeFi', 'L2', 'Meme', 'Web3'].map((category) => (
                    <button
                      key={category}
                      onClick={() => setFilterCategory(category)}
                      className={`px-4 py-2 rounded-full text-sm font-semibold transition-all ${
                        filterCategory === category
                          ? 'bg-blue-600 dark:bg-blue-500 text-white shadow-md'
                          : 'bg-white dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-600 border border-gray-300 dark:border-gray-600'
                      }`}
                    >
                      {category} {category === 'All' ? `(${ALL_CRYPTO_OPTIONS.length})` : `(${ALL_CRYPTO_OPTIONS.filter(c => c.category === category).length})`}
                    </button>
                  ))}
                </div>
              </div>

              {/* Crypto Grid */}
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
                {ALL_CRYPTO_OPTIONS
                  .filter(crypto => filterCategory === 'All' || crypto.category === filterCategory)
                  .map((crypto) => (
                    <button
                      key={crypto.id}
                      onClick={() => toggleCrypto(crypto.id)}
                      className={`flex items-center space-x-2 p-3 rounded-xl border-2 transition-all duration-200 ${
                        selectedCryptos.includes(crypto.id)
                          ? 'border-blue-500 dark:border-blue-400 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 shadow-md'
                          : 'border-gray-200 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:border-gray-300 dark:hover:border-gray-500 hover:shadow-sm'
                      }`}
                    >
                      <CryptoIcon cryptoId={crypto.id} size="sm" />
                      <div className="text-left min-w-0 flex-1">
                        <div className="font-bold text-sm truncate">{crypto.symbol}</div>
                        <div className="text-xs opacity-75 truncate">{crypto.name}</div>
                      </div>
                    </button>
                  ))}
              </div>
              
              {/* Quick Actions */}
              <div className="mt-6 flex justify-between">
                <div className="flex space-x-2">
                  <button
                    onClick={() => setSelectedCryptos(ALL_CRYPTO_OPTIONS.map(c => c.id))}
                    className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-xl text-sm font-semibold"
                  >
                    Select All
                  </button>
                  <button
                    onClick={() => setSelectedCryptos([])}
                    className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-xl text-sm font-semibold"
                  >
                    Clear All
                  </button>
                </div>
                <button
                  onClick={() => setSelectedCryptos(ALL_CRYPTO_OPTIONS.filter(c => c.category === 'Major').map(c => c.id))}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-sm font-semibold"
                >
                  Major Only
                </button>
              </div>
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Model Selection */}
          <div>
            <label className="block text-lg font-bold text-gray-900 dark:text-gray-100 mb-4">
              AI Forecasting Engine
            </label>
            <div className="grid grid-cols-1 gap-3">
              {FORECAST_MODELS.map((model) => (
                <button
                  key={model.value}
                  onClick={() => setSelectedModel(model.value)}
                  className={`p-4 rounded-2xl border-2 transition-all duration-300 text-left ${
                    selectedModel === model.value
                      ? 'border-blue-500 dark:border-blue-400 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 shadow-lg transform scale-105'
                      : 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 hover:border-gray-300 dark:hover:border-gray-600 hover:shadow-md hover:scale-102'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-3">
                      <span className="text-2xl">{model.icon}</span>
                      <span className="font-bold text-gray-900 dark:text-gray-100 text-lg">{model.label}</span>
                    </div>
                    <div className="bg-gradient-to-r from-green-100 to-emerald-100 dark:from-green-900/30 dark:to-emerald-900/30 text-green-800 dark:text-green-300 px-3 py-1 rounded-full text-sm font-bold border border-green-200 dark:border-green-700">
                      {(() => {
                        const da = modelMetrics?.models?.[model.value]?.avg_directional_accuracy_1d;
                        return da != null ? `${Math.round(da * 100)}% DA` : model.accuracy;
                      })()}
                    </div>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">{model.description}</p>
                </button>
              ))}
            </div>
          </div>

          {/* Analysis Controls */}
          <div className="space-y-6">
            <div>
              <label className="block text-lg font-bold text-gray-900 dark:text-gray-100 mb-4">
                Analysis Settings
              </label>
              
              {/* Forecast Horizon */}
              <div className="mb-6">
                <label htmlFor="forecast-horizon" className="block text-sm font-bold text-gray-700 dark:text-gray-300 mb-3">
                  Forecast Horizon
                </label>
                <div className="grid grid-cols-2 gap-2">
                  {[
                    { value: 1, label: '1 Day', subtitle: 'Short-term' },
                    { value: 7, label: '7 Days', subtitle: 'Medium-term' },
                    { value: 30, label: '30 Days', subtitle: 'Long-term' }
                  ].map((option) => (
                    <button
                      key={option.value}
                      onClick={() => setForecastDays(option.value)}
                      className={`p-3 rounded-xl border-2 text-center transition-all duration-200 ${
                        forecastDays === option.value
                          ? 'border-blue-500 dark:border-blue-400 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
                          : 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:border-gray-300 dark:hover:border-gray-600'
                      }`}
                    >
                      <div className="font-bold">{option.label}</div>
                      <div className="text-xs opacity-75">{option.subtitle}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Real-time Analysis */}
              <div className="bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-700 p-4 rounded-xl border border-gray-200 dark:border-gray-700">
                <h4 className="font-bold text-gray-900 dark:text-gray-100 mb-3">Market Regime</h4>
                <div className="grid grid-cols-2 gap-3">
                  <div className="text-center">
                    <div className="text-sm text-gray-600 dark:text-gray-400">Volatility</div>
                    <div className="text-lg font-bold text-orange-600 dark:text-orange-400">Medium</div>
                  </div>
                  <div className="text-center">
                    <div className="text-sm text-gray-600 dark:text-gray-400">Trend</div>
                    <div className="text-lg font-bold text-green-600 dark:text-green-400">Bullish</div>
                  </div>
                </div>
              </div>

              {/* Refresh Button */}
              <button
                onClick={() => refetch()}
                className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-500 dark:to-indigo-500 hover:from-blue-700 hover:to-indigo-700 dark:hover:from-blue-600 dark:hover:to-indigo-600 text-white font-bold py-4 px-6 rounded-xl transition-all duration-300 shadow-lg hover:shadow-xl disabled:opacity-50"
                disabled={isLoading}
              >
                {isLoading ? (
                  <div className="flex items-center justify-center space-x-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    <span>Analyzing...</span>
                  </div>
                ) : (
                  <div className="flex items-center justify-center space-x-2">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    <span>Refresh Analysis</span>
                  </div>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* No data message */}
      {!isLoading && !hasForecasts && (
        <div className="market-card">
          <div className="text-center py-12">
            <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gray-100 dark:bg-gray-700 flex items-center justify-center">
              <svg className="w-8 h-8 text-gray-600 dark:text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-2">No forecast data available</h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">Prices may be temporarily unavailable. Try again in a moment.</p>
            <button
              className="bg-blue-600 hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 text-white px-6 py-2 rounded-xl font-semibold text-sm transition-colors"
              onClick={() => refetch()}
            >
              Retry
            </button>
          </div>
        </div>
      )}

      {/* AI Market Intelligence Dashboard */}
      {hasForecasts && forecastData && (
        <div className="market-card">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-2xl font-black text-gray-900 dark:text-gray-100 flex items-center">
                <svg className="w-6 h-6 mr-3 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                AI Market Intelligence
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">Advanced technical analysis and ML predictions</p>
            </div>
            <div className="text-right">
              <div className="text-sm text-gray-500 dark:text-gray-400">Last updated</div>
              <div className="text-sm font-bold text-gray-700 dark:text-gray-300">
                {new Date(forecastData.metadata.generated_at).toLocaleTimeString()}
              </div>
            </div>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="text-center p-6 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-2xl border-2 border-blue-200 dark:border-blue-700 shadow-sm hover:shadow-md transition-shadow">
              <div className="text-3xl font-black text-blue-600 dark:text-blue-400 mb-2">{Object.keys(forecastData.forecasts).length}</div>
              <div className="text-sm font-semibold text-gray-700 dark:text-gray-300">Assets Analyzed</div>
              <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">Real-time tracking</div>
            </div>
            
            <div className="text-center p-6 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-2xl border-2 border-green-200 dark:border-green-700 shadow-sm hover:shadow-md transition-shadow">
              <div className="text-3xl font-black text-green-600 dark:text-green-400 mb-2">
                {Object.values(forecastData.forecasts).filter(f => {
                  const latest = f.forecasts[f.forecasts.length - 1];
                  return ((latest.predicted_price - f.current_price) / f.current_price) > 0;
                }).length}
              </div>
              <div className="text-sm font-semibold text-gray-700 dark:text-gray-300">Bullish Signals</div>
              <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">Positive outlook</div>
            </div>
            
            <div className="text-center p-6 bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20 rounded-2xl border-2 border-red-200 dark:border-red-700 shadow-sm hover:shadow-md transition-shadow">
              <div className="text-3xl font-black text-red-600 dark:text-red-400 mb-2">
                {Object.values(forecastData.forecasts).filter(f => {
                  const latest = f.forecasts[f.forecasts.length - 1];
                  return ((latest.predicted_price - f.current_price) / f.current_price) <= 0;
                }).length}
              </div>
              <div className="text-sm font-semibold text-gray-700 dark:text-gray-300">Bearish Signals</div>
              <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">Negative outlook</div>
            </div>
            
            <div className="text-center p-6 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-2xl border-2 border-purple-200 dark:border-purple-700 shadow-sm hover:shadow-md transition-shadow">
              <div className="text-3xl font-black text-purple-600 dark:text-purple-400 mb-2">
                {Math.round(Object.values(forecastData.forecasts).reduce((acc, f) => {
                  const latest = f.forecasts[f.forecasts.length - 1];
                  return acc + (latest.confidence * 100);
                }, 0) / Object.keys(forecastData.forecasts).length)}%
              </div>
              <div className="text-sm font-semibold text-gray-700 dark:text-gray-300">Avg Confidence</div>
              <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">Model certainty</div>
            </div>
            
            <div className="text-center p-6 bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 rounded-2xl border-2 border-orange-200 dark:border-orange-700 shadow-sm hover:shadow-md transition-shadow">
              <div className="text-3xl font-black text-orange-600 dark:text-orange-400 mb-2">
                {getModelAccuracy(selectedModel)}
              </div>
              <div className="text-sm font-semibold text-gray-700 dark:text-gray-300">Directional Accuracy</div>
              <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">Avg across all coins</div>
            </div>
          </div>
        </div>
      )}

      {/* Enhanced Crypto Forecast Cards */}
      {hasForecasts && forecastData && <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {Object.entries(forecastData.forecasts).map(([cryptoId, forecast]) => {
          const isSelected = selectedCrypto === cryptoId;
          const latestForecast = forecast.forecasts[forecast.forecasts.length - 1];
          
          // Use real-time price if available, otherwise fall back to forecast's current_price
          const currentPrice = realTimePrices?.[cryptoId]?.usd ?? forecast.current_price;
          const priceChange = ((latestForecast.predicted_price - currentPrice) / currentPrice) * 100;
          const isPositive = priceChange >= 0;

          // Generate sparkline data from historical prices
          const sparklineData = forecast.historical_data?.slice(-7).map((point: { date: string; price: number; is_historical: boolean }) => ({
            value: point.price,
            timestamp: point.date
          })) || [];

          return (
            <div
              key={cryptoId}
              className={`bg-white dark:bg-gray-800 border-2 rounded-3xl p-6 cursor-pointer transition-all duration-300 transform hover:scale-105 hover:shadow-xl group ${
                isSelected ? 'border-blue-500 dark:border-blue-400 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 shadow-xl' : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600 shadow-lg'
              }`}
              onClick={() => setSelectedCrypto(isSelected ? null : cryptoId)}
            >
              {/* Enhanced Header with Sparkline */}
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center space-x-4">
                  <div className={`w-12 h-12 rounded-2xl bg-gradient-to-br ${getCryptoIconStyle(cryptoId)} flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform duration-200`}>
                    <span className="text-white text-lg font-black">
                      {getCryptoSymbol(cryptoId).charAt(0)}
                    </span>
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl font-black text-gray-900 dark:text-gray-100 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                      {getCryptoDisplayName(cryptoId)}
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400 font-semibold">
                      {getCryptoSymbol(cryptoId)}
                    </p>
                    {/* Mini Sparkline */}
                    <div className="mt-2 h-8">
                      <MiniSparkline 
                        data={sparklineData} 
                        height={32}
                        className="opacity-70 group-hover:opacity-100 transition-opacity"
                      />
                    </div>
                  </div>
                </div>
                <div className={`price-change ${isPositive ? 'price-positive' : 'price-negative'}`}>
                  {isPositive ? '+' : ''}{priceChange.toFixed(1)}%
                </div>
              </div>

              {/* Enhanced Price Display */}
              <div className="space-y-4">
                <div className="text-center p-4 bg-gradient-to-br from-emerald-50 to-green-50 dark:from-emerald-900/20 dark:to-green-900/20 rounded-2xl border-2 border-emerald-200 dark:border-emerald-700 group-hover:from-emerald-100 group-hover:to-green-100 dark:group-hover:from-emerald-900/30 dark:group-hover:to-green-900/30 transition-all">
                  <div className="flex items-center justify-center gap-2 mb-2">
                    <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></div>
                    <div className="text-xs text-emerald-600 dark:text-emerald-400 font-bold uppercase tracking-wider">Live Price</div>
                  </div>
                  <div className="text-3xl font-black text-emerald-700 dark:text-emerald-300">
                    {formatPrice(currentPrice)}
                  </div>
                  {realTimePrices?.[cryptoId]?.usd && (
                    <div className="text-xs text-emerald-500 dark:text-emerald-400 mt-1">
                      Real-time update
                    </div>
                  )}
                </div>
                
                <div className="text-center p-4 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-2xl border-2 border-blue-200 dark:border-blue-700 group-hover:from-blue-100 group-hover:to-indigo-100 dark:group-hover:from-blue-900/30 dark:group-hover:to-indigo-900/30 transition-all">
                  <div className="text-xs text-blue-600 dark:text-blue-400 font-bold mb-2 uppercase tracking-wider">{forecastDays}d Forecast</div>
                  <div className="text-3xl font-black text-blue-700 dark:text-blue-300">
                    {formatPrice(latestForecast.predicted_price)}
                  </div>
                  <div className="text-xs text-blue-500 dark:text-blue-400 mt-1">
                    {isPositive ? 'Expected gain' : 'Expected loss'}
                  </div>
                </div>
              </div>

              {/* Enhanced Technical Analysis Section */}
              <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700 space-y-4">
                {/* Confidence Indicator */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-600 dark:text-gray-400 font-semibold">Model Confidence</span>
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      {Math.round(latestForecast.confidence * 100)}%
                    </span>
                  </div>
                  <ConfidenceIndicator 
                    confidence={latestForecast.confidence} 
                    size="md"
                    showPercentage={false}
                  />
                </div>

                {/* Technical Signal Indicators */}
                {forecast.forecasts[0]?.technical_signals && (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-600 dark:text-gray-400 font-semibold">Technical Signals</span>
                      <TechnicalSignalIndicator
                        rsi={forecast.forecasts[0].technical_signals.rsi}
                        regime={forecast.forecasts[0].technical_signals.regime}
                        macd={forecast.forecasts[0].technical_signals.macd}
                        bollingerPosition={forecast.forecasts[0].technical_signals.bollinger_position}
                        volatility={forecast.forecasts[0].technical_signals.volatility}
                        size="sm"
                      />
                    </div>
                  </div>
                )}

                {/* Model Metrics */}
                <div className="grid grid-cols-2 gap-3 pt-2">
                  <div className="text-center p-2 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                    <div className="text-xs text-gray-500 dark:text-gray-400 font-semibold">R² Score</div>
                    <div className={`text-sm font-black ${
                      (forecast.model_metrics.r_squared ?? 0) > 0.9 ? 'text-green-600 dark:text-green-400' :
                      (forecast.model_metrics.r_squared ?? 0) > 0.8 ? 'text-yellow-600 dark:text-yellow-400' : 'text-red-600 dark:text-red-400'
                    }`}>
                      {(forecast.model_metrics.r_squared ?? 0).toFixed(2)}
                    </div>
                  </div>
                  <div className="text-center p-2 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                    <div className="text-xs text-gray-500 dark:text-gray-400 font-semibold">MAPE</div>
                    <div className="text-sm font-black text-green-600 dark:text-green-400">{(forecast.model_metrics.mape ?? 0).toFixed(1)}%</div>
                  </div>
                </div>

                {/* Quick Actions */}
                <div className="flex space-x-2 pt-2">
                  <button
                    className="flex-1 px-3 py-2 text-xs font-semibold text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/30 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-900/50 transition-colors"
                    onClick={(e) => handleAddToWatchlist(e, { id: cryptoId })}
                  >
                    + Watchlist
                  </button>
                  <button
                    className="flex-1 px-3 py-2 text-xs font-semibold text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-700/50 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                    onClick={(e) => handleSetAlert(e, { id: cryptoId })}
                  >
                    Set Alert
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>}

      {/* Detailed View */}
      {hasForecasts && forecastData && selectedCrypto && forecastData.forecasts[selectedCrypto] && (
        <div className="market-card">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-4">
              <div className={`crypto-icon ${getCryptoIconStyle(selectedCrypto)}`}>
                <span className="text-white text-lg">
                  {getCryptoSymbol(selectedCrypto).charAt(0)}
                </span>
              </div>
              <div>
                <h3 className="text-2xl font-black text-gray-900 dark:text-gray-100">
                  {getCryptoDisplayName(selectedCrypto)} Detailed Forecast
                </h3>
                <p className="text-gray-600 dark:text-gray-400">In-depth analysis and predictions</p>
              </div>
            </div>
            <button
              onClick={() => setSelectedCrypto(null)}
              className="coinbase-button-secondary"
            >
              Close
            </button>
          </div>

          <div className="space-y-6">
            {/* Live Price Banner */}
            {realTimePrices?.[selectedCrypto]?.usd && (
              <div className="bg-gradient-to-r from-emerald-50 to-green-50 dark:from-emerald-900/20 dark:to-green-900/20 rounded-2xl p-6 border-2 border-emerald-200 dark:border-emerald-700">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="flex items-center gap-3 mb-2">
                      <div className="w-3 h-3 bg-emerald-500 rounded-full animate-pulse"></div>
                      <h4 className="text-lg font-black text-gray-900 dark:text-gray-100">Live Market Price</h4>
                    </div>
                    <div className="text-4xl font-black text-emerald-700 dark:text-emerald-300">
                      {formatPrice(realTimePrices[selectedCrypto].usd)}
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                      Real-time data • Updated every 15 seconds
                    </p>
                  </div>
                  <div className="text-right">
                    <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">Forecast was based on</div>
                    <div className="text-2xl font-bold text-gray-700 dark:text-gray-300">
                      {formatPrice(forecastData.forecasts[selectedCrypto].current_price)}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      at {new Date(forecastData.forecasts[selectedCrypto].generated_at).toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {/* Technical Analysis Dashboard */}
            {(forecastData.forecasts[selectedCrypto] as any).technical_analysis && (
              <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-3xl p-8 border-2 border-blue-200 dark:border-blue-700 mb-6">
                <h4 className="text-2xl font-black text-gray-900 dark:text-gray-100 mb-6 flex items-center">
                  <svg className="w-7 h-7 mr-3 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                  Professional Technical Analysis
                </h4>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  {/* RSI */}
                  <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
                    <div className="text-xs text-gray-600 dark:text-gray-400 font-bold mb-2">RSI (14)</div>
                    <div className={`text-2xl font-black ${
                      (forecastData.forecasts[selectedCrypto] as any).technical_analysis.rsi > 70 ? 'text-red-600 dark:text-red-400' :
                      (forecastData.forecasts[selectedCrypto] as any).technical_analysis.rsi < 30 ? 'text-green-600 dark:text-green-400' : 'text-gray-700 dark:text-gray-300'
                    }`}>
                      {(forecastData.forecasts[selectedCrypto] as any).technical_analysis.rsi.toFixed(1)}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      {(forecastData.forecasts[selectedCrypto] as any).technical_analysis.rsi > 70 ? 'Overbought' :
                       (forecastData.forecasts[selectedCrypto] as any).technical_analysis.rsi < 30 ? 'Oversold' : 'Neutral'}
                    </div>
                  </div>
                  
                  {/* MACD */}
                  <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
                    <div className="text-xs text-gray-600 dark:text-gray-400 font-bold mb-2">MACD</div>
                    <div className={`text-2xl font-black ${
                      (forecastData.forecasts[selectedCrypto] as any).technical_analysis.macd_histogram > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                    }`}>
                      {(forecastData.forecasts[selectedCrypto] as any).technical_analysis.macd.toFixed(2)}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      {(forecastData.forecasts[selectedCrypto] as any).technical_analysis.macd_histogram > 0 ? 'Bullish' : 'Bearish'}
                    </div>
                  </div>
                  
                  {/* Bollinger Position */}
                  <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
                    <div className="text-xs text-gray-600 dark:text-gray-400 font-bold mb-2">BB Position</div>
                    <div className={`text-2xl font-black ${
                      (forecastData.forecasts[selectedCrypto] as any).technical_analysis.bollinger_position > 0.8 ? 'text-red-600 dark:text-red-400' :
                      (forecastData.forecasts[selectedCrypto] as any).technical_analysis.bollinger_position < 0.2 ? 'text-green-600 dark:text-green-400' : 'text-gray-700 dark:text-gray-300'
                    }`}>
                      {((forecastData.forecasts[selectedCrypto] as any).technical_analysis.bollinger_position * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      {(forecastData.forecasts[selectedCrypto] as any).technical_analysis.bollinger_position > 0.8 ? 'Near Top' :
                       (forecastData.forecasts[selectedCrypto] as any).technical_analysis.bollinger_position < 0.2 ? 'Near Bottom' : 'Middle'}
                    </div>
                  </div>
                  
                  {/* Volatility */}
                  <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
                    <div className="text-xs text-gray-600 dark:text-gray-400 font-bold mb-2">Volatility</div>
                    <div className="text-2xl font-black text-orange-600 dark:text-orange-400">
                      {(forecastData.forecasts[selectedCrypto] as any).technical_analysis.volatility_annual.toFixed(1)}%
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">Annual</div>
                  </div>
                </div>
                
                {/* Trading Signal */}
                <div className={`p-6 rounded-2xl border-2 ${
                  (forecastData.forecasts[selectedCrypto] as any).technical_analysis.trading_signal === 'BUY' 
                    ? 'bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 border-green-300 dark:border-green-700'
                    : (forecastData.forecasts[selectedCrypto] as any).technical_analysis.trading_signal === 'SELL'
                    ? 'bg-gradient-to-r from-red-50 to-rose-50 dark:from-red-900/20 dark:to-rose-900/20 border-red-300 dark:border-red-700'
                    : 'bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-700 border-gray-300 dark:border-gray-600'
                }`}>
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm text-gray-600 dark:text-gray-400 font-bold mb-2">AI Trading Signal</div>
                      <div className={`text-4xl font-black ${
                        (forecastData.forecasts[selectedCrypto] as any).technical_analysis.trading_signal === 'BUY' ? 'text-green-700 dark:text-green-400' :
                        (forecastData.forecasts[selectedCrypto] as any).technical_analysis.trading_signal === 'SELL' ? 'text-red-700 dark:text-red-400' :
                        'text-gray-700 dark:text-gray-300'
                      }`}>
                        {(forecastData.forecasts[selectedCrypto] as any).technical_analysis.trading_signal}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm text-gray-600 dark:text-gray-400 font-bold mb-2">Signal Strength</div>
                      <div className="text-3xl font-black text-blue-600 dark:text-blue-400">
                        {((forecastData.forecasts[selectedCrypto] as any).technical_analysis.signal_strength * 100).toFixed(0)}%
                      </div>
                    </div>
                  </div>
                  <div className="mt-4 text-sm text-gray-600 dark:text-gray-400">
                    Market Regime: <span className="font-bold text-gray-900 dark:text-gray-100">{(forecastData.forecasts[selectedCrypto] as any).technical_analysis.market_regime.toUpperCase()}</span>
                    {' • '}
                    7-Day Trend: <span className={`font-bold ${
                      (forecastData.forecasts[selectedCrypto] as any).technical_analysis.trend_7d > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                    }`}>
                      {(forecastData.forecasts[selectedCrypto] as any).technical_analysis.trend_7d > 0 ? '+' : ''}
                      {(forecastData.forecasts[selectedCrypto] as any).technical_analysis.trend_7d.toFixed(2)}%
                    </span>
                  </div>
                </div>
              </div>
            )}
            
            {/* Model Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="text-center p-6 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-xl border border-blue-200 dark:border-blue-700">
                <div className="text-sm text-blue-600 dark:text-blue-400 font-bold mb-2 uppercase tracking-wider">
                  MAPE
                </div>
                <div className="text-3xl font-black text-blue-700 dark:text-blue-300 mb-1">
                  {(forecastData.forecasts[selectedCrypto].model_metrics.mape ?? 0).toFixed(1)}%
                </div>
                <div className="text-xs text-blue-500 dark:text-blue-400">Model accuracy</div>
              </div>
              <div className="text-center p-6 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-xl border border-green-200 dark:border-green-700">
                <div className="text-sm text-green-600 dark:text-green-400 font-bold mb-2 uppercase tracking-wider">
                  RMSE
                </div>
                <div className="text-3xl font-black text-green-700 dark:text-green-300 mb-1">
                  {(forecastData.forecasts[selectedCrypto].model_metrics.rmse ?? 0).toFixed(3)}
                </div>
                <div className="text-xs text-green-500 dark:text-green-400">Prediction error</div>
              </div>
              <div className="text-center p-6 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-xl border border-purple-200 dark:border-purple-700">
                <div className="text-sm text-purple-600 dark:text-purple-400 font-bold mb-2 uppercase tracking-wider">
                  R²
                </div>
                <div className="text-3xl font-black text-purple-700 dark:text-purple-300 mb-1">
                  {(forecastData.forecasts[selectedCrypto].model_metrics.r_squared ?? 0).toFixed(3)}
                </div>
                <div className="text-xs text-purple-500 dark:text-purple-400">Model fit quality</div>
              </div>
              <div className="text-center p-6 bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 rounded-xl border border-orange-200 dark:border-orange-700">
                <div className="text-sm text-orange-600 dark:text-orange-400 font-bold mb-2 uppercase tracking-wider">
                  Backtest Samples
                </div>
                <div className="text-3xl font-black text-orange-700 dark:text-orange-300 mb-1">
                  {(forecastData.forecasts[selectedCrypto].model_metrics as any).backtest_samples || 0}
                </div>
                <div className="text-xs text-orange-500 dark:text-orange-400">Validation points</div>
              </div>
            </div>

            {/* Enhanced Price Chart with Historical Data */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h4 className="text-xl font-black text-gray-900 dark:text-gray-100">Historical Prices + AI Forecast</h4>
                <div className="flex items-center space-x-4">
                  {(forecastData.forecasts[selectedCrypto] as any).technical_analysis && (
                    <>
                      <div className="text-sm">
                        <span className="text-gray-600 dark:text-gray-400">Support:</span>
                        <span className="ml-2 font-bold text-green-600 dark:text-green-400">
                          {formatPrice((forecastData.forecasts[selectedCrypto] as any).technical_analysis.support_level)}
                        </span>
                      </div>
                      <div className="text-sm">
                        <span className="text-gray-600 dark:text-gray-400">Resistance:</span>
                        <span className="ml-2 font-bold text-red-600 dark:text-red-400">
                          {formatPrice((forecastData.forecasts[selectedCrypto] as any).technical_analysis.resistance_level)}
                        </span>
                      </div>
                    </>
                  )}
                </div>
              </div>
              <div className="bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-700 p-6 rounded-xl border border-gray-200 dark:border-gray-700">
                <ResponsiveContainer width="100%" height={400}>
                  <AreaChart
                    data={[
                      // Combine historical data with forecasts
                      ...((forecastData.forecasts[selectedCrypto] as any).historical_data || []).slice(-15).map((point: any) => ({
                        date: new Date(point.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
                        historical: point.price,
                        forecast: null,
                        upper: null,
                        lower: null,
                      })),
                      // Add current price as transition point
                      {
                        date: 'Now',
                        historical: forecastData.forecasts[selectedCrypto].current_price,
                        forecast: forecastData.forecasts[selectedCrypto].current_price,
                        upper: forecastData.forecasts[selectedCrypto].current_price,
                        lower: forecastData.forecasts[selectedCrypto].current_price,
                      },
                      // Add forecast data
                      ...forecastData.forecasts[selectedCrypto].forecasts.map((point) => ({
                        date: new Date(point.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
                        historical: null,
                        forecast: point.predicted_price,
                        upper: point.confidence_upper,
                        lower: point.confidence_lower,
                      }))
                    ]}
                    margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                  >
                    <defs>
                      <linearGradient id="colorHistorical" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#6b7280" stopOpacity={0.6}/>
                        <stop offset="95%" stopColor="#6b7280" stopOpacity={0.05}/>
                      </linearGradient>
                      <linearGradient id="colorForecast" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.1}/>
                      </linearGradient>
                      <linearGradient id="colorConfidence" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#10b981" stopOpacity={0.2}/>
                        <stop offset="95%" stopColor="#10b981" stopOpacity={0.02}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis 
                      dataKey="date" 
                      stroke="#6b7280"
                      style={{ fontSize: '11px', fontWeight: 600 }}
                      angle={-45}
                      textAnchor="end"
                      height={60}
                    />
                    <YAxis 
                      stroke="#6b7280"
                      style={{ fontSize: '12px', fontWeight: 600 }}
                      tickFormatter={(value) => `$${value.toLocaleString()}`}
                      domain={['auto', 'auto']}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#ffffff',
                        border: '2px solid #e5e7eb',
                        borderRadius: '12px',
                        padding: '16px',
                        boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)'
                      }}
                      formatter={(value: number, name: string) => {
                        if (!value) return ['', ''];
                        return [
                          `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
                          name === 'historical' ? 'Historical Price' :
                          name === 'forecast' ? 'Predicted Price' :
                          name === 'upper' ? 'Upper Confidence' :
                          name === 'lower' ? 'Lower Confidence' : name
                        ];
                      }}
                      labelStyle={{ fontWeight: 'bold', marginBottom: '8px', color: '#1f2937' }}
                    />
                    <Legend 
                      wrapperStyle={{ paddingTop: '20px', fontSize: '13px', fontWeight: 600 }}
                      iconType="circle"
                    />
                    
                    {/* Confidence Interval Band */}
                    <Area
                      type="monotone"
                      dataKey="upper"
                      stroke="none"
                      fill="url(#colorConfidence)"
                      name="Confidence Range"
                      isAnimationActive={true}
                    />
                    <Area
                      type="monotone"
                      dataKey="lower"
                      stroke="none"
                      fill="url(#colorConfidence)"
                      name=""
                      isAnimationActive={true}
                    />
                    
                    {/* Historical Price Line */}
                    <Area
                      type="monotone"
                      dataKey="historical"
                      stroke="#6b7280"
                      strokeWidth={3}
                      fill="url(#colorHistorical)"
                      name="Historical Price"
                      dot={false}
                      isAnimationActive={true}
                      connectNulls={false}
                    />
                    
                    {/* Forecast Price Line */}
                    <Area
                      type="monotone"
                      dataKey="forecast"
                      stroke="#3b82f6"
                      strokeWidth={4}
                      fill="url(#colorForecast)"
                      name="AI Forecast"
                      dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
                      activeDot={{ r: 6, strokeWidth: 2 }}
                      isAnimationActive={true}
                      connectNulls={false}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Key Trading Levels */}
            {(forecastData.forecasts[selectedCrypto] as any).technical_analysis && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                {/* Price Action Summary */}
                <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 border-2 border-gray-200 dark:border-gray-700 shadow-sm">
                  <h4 className="text-lg font-black text-gray-900 dark:text-gray-100 mb-4 flex items-center">
                    <svg className="w-5 h-5 mr-2 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                    </svg>
                    Price Action Summary
                  </h4>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Live Price</span>
                      <span className="text-lg font-black text-emerald-600 dark:text-emerald-400 flex items-center gap-2">
                        <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></span>
                        {formatPrice(realTimePrices?.[selectedCrypto]?.usd ?? forecastData.forecasts[selectedCrypto].current_price)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Support Level</span>
                      <span className="text-lg font-black text-green-600 dark:text-green-400">
                        {formatPrice((forecastData.forecasts[selectedCrypto] as any).technical_analysis.support_level)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Resistance Level</span>
                      <span className="text-lg font-black text-red-600 dark:text-red-400">
                        {formatPrice((forecastData.forecasts[selectedCrypto] as any).technical_analysis.resistance_level)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Bollinger Middle</span>
                      <span className="text-lg font-black text-blue-600 dark:text-blue-400">
                        {formatPrice((forecastData.forecasts[selectedCrypto] as any).technical_analysis.bollinger_middle)}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Forecast Summary */}
                <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 border-2 border-gray-200 dark:border-gray-700 shadow-sm">
                  <h4 className="text-lg font-black text-gray-900 dark:text-gray-100 mb-4 flex items-center">
                    <svg className="w-5 h-5 mr-2 text-purple-600 dark:text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    Forecast Summary
                  </h4>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Target Price ({forecastDays}d)</span>
                      <span className="text-lg font-black text-blue-600 dark:text-blue-400">
                        {formatPrice(forecastData.forecasts[selectedCrypto].forecasts[forecastData.forecasts[selectedCrypto].forecasts.length - 1].predicted_price)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Expected Change</span>
                      <span className={`text-lg font-black ${
                        ((forecastData.forecasts[selectedCrypto].forecasts[forecastData.forecasts[selectedCrypto].forecasts.length - 1].predicted_price - forecastData.forecasts[selectedCrypto].current_price) / forecastData.forecasts[selectedCrypto].current_price) >= 0
                          ? 'text-green-600 dark:text-green-400'
                          : 'text-red-600 dark:text-red-400'
                      }`}>
                        {((forecastData.forecasts[selectedCrypto].forecasts[forecastData.forecasts[selectedCrypto].forecasts.length - 1].predicted_price - forecastData.forecasts[selectedCrypto].current_price) / forecastData.forecasts[selectedCrypto].current_price * 100) >= 0 ? '+' : ''}
                        {(((forecastData.forecasts[selectedCrypto].forecasts[forecastData.forecasts[selectedCrypto].forecasts.length - 1].predicted_price - forecastData.forecasts[selectedCrypto].current_price) / forecastData.forecasts[selectedCrypto].current_price) * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Best Case</span>
                      <span className="text-lg font-black text-green-600 dark:text-green-400">
                        {formatPrice(forecastData.forecasts[selectedCrypto].forecasts[forecastData.forecasts[selectedCrypto].forecasts.length - 1].confidence_upper)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Worst Case</span>
                      <span className="text-lg font-black text-red-600 dark:text-red-400">
                        {formatPrice(forecastData.forecasts[selectedCrypto].forecasts[forecastData.forecasts[selectedCrypto].forecasts.length - 1].confidence_lower)}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Forecast Timeline */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h4 className="text-xl font-black text-gray-900 dark:text-gray-100">Daily Price Predictions Timeline</h4>
                <div className="text-sm text-gray-500 dark:text-gray-400">
                  {forecastData.forecasts[selectedCrypto].forecasts.length} predictions
                </div>
              </div>
              <div className="space-y-3 max-h-80 overflow-y-auto">
                {forecastData.forecasts[selectedCrypto].forecasts.map((point, index) => {
                  const date = new Date(point.date);
                  const isToday = index === 0;
                  const isLast = index === forecastData.forecasts[selectedCrypto].forecasts.length - 1;
                  
                  return (
                    <div
                      key={point.date}
                      className={`flex items-center justify-between p-4 rounded-xl border-2 transition-all duration-200 ${
                        isToday 
                          ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-300 dark:border-blue-700 shadow-md' 
                          : isLast
                          ? 'bg-green-50 dark:bg-green-900/20 border-green-300 dark:border-green-700 shadow-md'
                          : 'bg-gray-50 dark:bg-gray-700/50 border-gray-200 dark:border-gray-600 hover:border-gray-300 dark:hover:border-gray-500'
                      }`}
                    >
                      <div className="flex items-center space-x-4">
                        <div className={`w-3 h-3 rounded-full ${
                          isToday ? 'bg-blue-500 dark:bg-blue-400' : isLast ? 'bg-green-500 dark:bg-green-400' : 'bg-gray-400 dark:bg-gray-500'
                        }`}></div>
                        <div>
                          <div className="font-bold text-gray-900 dark:text-gray-100 text-lg">
                            {date.toLocaleDateString('en-US', { 
                              weekday: 'long', 
                              month: 'short', 
                              day: 'numeric' 
                            })}
                          </div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">
                            {isToday ? 'Current' : isLast ? 'Final Prediction' : `Day ${index + 1}`}
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-black text-gray-900 dark:text-gray-100 text-xl">
                          {formatPrice(point.predicted_price)}
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          Range: {formatPrice(point.confidence_lower)} - {formatPrice(point.confidence_upper)}
                        </div>
                        <div className={`text-xs font-semibold mt-1 px-2 py-1 rounded-full ${getConfidenceColor(point.confidence)}`}>
                          {Math.round(point.confidence * 100)}% confidence
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Helper function to get crypto icon styling
function getCryptoIconStyle(cryptoId: string): string {
  const styles: Record<string, string> = {
    // Major Cryptocurrencies
    'bitcoin': 'from-orange-400 to-orange-600',
    'ethereum': 'from-blue-400 to-blue-600',
    'ripple': 'from-blue-500 to-blue-700',
    'binancecoin': 'from-yellow-400 to-yellow-600',
    
    // Layer 1 & Smart Contract Platforms
    'solana': 'from-purple-400 to-purple-600',
    'cardano': 'from-blue-500 to-indigo-600',
    'avalanche-2': 'from-red-400 to-red-600',
    'polkadot': 'from-pink-400 to-pink-600',
    'near': 'from-green-400 to-green-600',
    
    // DeFi Tokens
    'uniswap': 'from-pink-300 to-pink-500',
    'chainlink': 'from-blue-300 to-blue-500',
    'aave': 'from-purple-500 to-purple-700',
    'compound-governance-token': 'from-green-500 to-green-700',
    'maker': 'from-teal-400 to-teal-600',
    'curve-dao-token': 'from-yellow-500 to-yellow-700',
    
    // Layer 2 & Scaling
    'matic-network': 'from-purple-300 to-purple-500',
    'optimism': 'from-red-500 to-red-700',
    'arbitrum': 'from-blue-500 to-blue-700',
    
    // Meme & Community
    'dogecoin': 'from-yellow-300 to-yellow-500',
    'shiba-inu': 'from-orange-300 to-orange-500',
    
    // Infrastructure & Web3
    'filecoin': 'from-blue-600 to-blue-800',
    'the-graph': 'from-indigo-400 to-indigo-600',
    'internet-computer': 'from-purple-600 to-purple-800',
  };
  
  return styles[cryptoId] || 'from-gray-400 to-gray-600';
}

export default ForecastPanel;
