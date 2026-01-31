import { useState, useMemo, useRef } from 'react';
import { 
  formatPrice, 
  getCryptoDisplayName, 
  getCryptoSymbol,
  useCryptoPrices
} from '../hooks/useCryptoPrices';
import { formatPercentage, formatMarketCap, formatVolume, useMarketData } from '../hooks/useMarketData';
import { CryptoTableSkeleton } from './LoadingStates';
import { WebSocketError } from './ErrorStates';
import CryptoIcon from './CryptoIcon';
import CoinDetailView from './CoinDetailView';

// Use the same 5 cryptos as PriceTicker for fast loading and consistency
const TICKER_CRYPTOS = ['bitcoin', 'ethereum', 'solana', 'cardano', 'polkadot'];

interface RealTimeCryptoGridProps {
  selectedCrypto: string | null;
  onSelectCrypto: (cryptoId: string | null) => void;
}

function RealTimeCryptoGrid({ selectedCrypto, onSelectCrypto }: RealTimeCryptoGridProps) {
  // Use the SAME 5 cryptos as PriceTicker - this ensures fast loading and consistency!
  const { data: prices, isLoading: pricesLoading, isError: pricesError } = useCryptoPrices(
    TICKER_CRYPTOS,
    ['usd'],
    { refetchInterval: 60000 } // Reduced interval to lower API usage
  );
  
  const { data: marketData, isLoading: marketLoading, isError: marketError } = useMarketData(
    TICKER_CRYPTOS
  );
  
  // Combine prices and market data into the format expected by the component
  const streamData = useMemo(() => {
    // Handle empty objects from API (when cache fallback returns {})
    if (!prices || !marketData || 
        Object.keys(prices).length === 0 || 
        Object.keys(marketData).length === 0) {
      return null;
    }
    
    const combined: Record<string, {
      price: number;
      change_24h: number | null;
      market_cap: number;
      volume_24h: number;
    }> = {};
    
    TICKER_CRYPTOS.forEach(cryptoId => {
      const price = prices[cryptoId]?.usd;
      const market = marketData[cryptoId];
      
      if (price && price > 0) {
        combined[cryptoId] = {
          price,
          change_24h: market?.price_change_24h ?? null,
          market_cap: market?.market_cap ?? 0,
          volume_24h: market?.volume_24h ?? 0,
        };
      }
    });
    
    // Return null if no valid data was found
    if (Object.keys(combined).length === 0) {
      return null;
    }
    
    return {
      data: combined,
      timestamp: Date.now() / 1000,
      type: 'snapshot',
    };
  }, [prices, marketData]);
  
  const isLoading = pricesLoading || marketLoading;
  const error = pricesError || marketError;
  const isConnected = !isLoading && !error && streamData !== null;
  const isFallbackMode = true; // Using REST API (same as PriceTicker)
  const reconnectCount = 0;
  
  const reconnect = () => {
    // Force refetch by reloading
    window.location.reload();
  };

  const [priceChanges, setPriceChanges] = useState<Record<string, 'up' | 'down' | 'none'>>({});
  const previousDataRef = useRef<Record<string, { price: number }>>({});

  // Track price changes for visual feedback
  const cryptoData = useMemo(() => {
    if (!streamData?.data) return {};

    const newPriceChanges: Record<string, 'up' | 'down' | 'none'> = {};
    const currentData = streamData.data;
    const previousData = previousDataRef.current;
    
    Object.keys(currentData).forEach(cryptoId => {
      const currentPrice = currentData[cryptoId]?.price;
      const previousPriceData = previousData[cryptoId];
      
      if (currentPrice !== undefined && currentPrice !== null) {
        if (previousPriceData && previousPriceData.price !== undefined) {
          const priceDiff = currentPrice - previousPriceData.price;
          // Use a small threshold (0.01% of price) to avoid noise from tiny fluctuations
          const threshold = Math.abs(currentPrice * 0.0001);
          
          if (Math.abs(priceDiff) > threshold) {
            newPriceChanges[cryptoId] = priceDiff > 0 ? 'up' : 'down';
          } else {
            newPriceChanges[cryptoId] = 'none';
          }
        } else {
          // First time seeing this crypto, no change detected yet
          newPriceChanges[cryptoId] = 'none';
        }
        
        // Update previous data for next comparison
        previousData[cryptoId] = { price: currentPrice };
      } else {
        newPriceChanges[cryptoId] = 'none';
      }
    });

    setPriceChanges(newPriceChanges);
    return currentData;
  }, [streamData]);

  if (isLoading) {
    return (
      <div className="animate-fade-in-down">
        <CryptoTableSkeleton rows={6} />
      </div>
    );
  }

  if (error && !streamData) {
    return (
      <div className="animate-fade-in-up">
        <WebSocketError 
          onReconnect={reconnect}
          className="animate-shake"
        />
      </div>
    );
  }

  if (!streamData || !streamData.data || Object.keys(streamData.data).length === 0) {
    return (
      <div className="animate-fade-in-down">
        <CryptoTableSkeleton rows={6} />
      </div>
    );
  }

  return (
    <div className="space-y-10">
      {/* Enhanced Connection Status */}
      <div className="flex items-center justify-between animate-fade-in-down">
        <div className="flex items-center space-x-4">
          <div className={`flex items-center space-x-2 px-4 py-2 rounded-2xl text-sm font-bold transition-all duration-300 shadow-md ${
            isConnected && !isFallbackMode 
              ? 'bg-green-100 text-green-800 connection-online border border-green-300' 
              : isFallbackMode 
              ? 'bg-yellow-100 text-yellow-800 connection-warning border border-yellow-300'
              : 'bg-red-100 text-red-800 connection-offline border border-red-300'
          }`}>
            <div className={`w-3 h-3 rounded-full transition-all duration-300 ${
              isConnected && !isFallbackMode 
                ? 'bg-green-500 animate-pulse-glow' 
                : isFallbackMode 
                ? 'bg-yellow-500 animate-pulse'
                : 'bg-red-500'
            }`}></div>
            <span>
              {isConnected && !isFallbackMode 
                ? 'Live Stream' 
                : isFallbackMode 
                ? 'Polling Mode'
                : 'Disconnected'}
            </span>
          </div>
          {streamData && (
            <span className="text-sm text-gray-500 animate-fade-in-up">
              Last updated: {new Date(streamData.timestamp * 1000).toLocaleTimeString()}
            </span>
          )}
          {reconnectCount > 0 && (
            <span className="text-xs text-orange-600 bg-orange-50 px-2 py-1 rounded-full animate-fade-in-up">
              Reconnect attempts: {reconnectCount}
            </span>
          )}
        </div>
        {error && (
          <div className="flex items-center space-x-2">
            <button 
              className="coinbase-button-secondary text-sm px-4 py-2 hover:bg-red-50 hover:border-red-200 transition-all duration-300 rounded-2xl"
              onClick={reconnect}
              disabled={isConnected && !error}
            >
              {isConnected ? 'Reconnecting...' : 'Reconnect'}
            </button>
          </div>
        )}
      </div>

      {/* Market Table */}
      <div className="crypto-table">
        {/* Table Header */}
        <div className="crypto-table-header">
          <div className="grid grid-cols-12 gap-6">
            <div className="col-span-4 text-sm font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider">Name</div>
            <div className="col-span-2 text-sm font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider text-right">Price</div>
            <div className="col-span-2 text-sm font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider text-right">24h Change</div>
            <div className="col-span-2 text-sm font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider text-right">Market Cap</div>
            <div className="col-span-2 text-sm font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider text-right">Volume (24h)</div>
          </div>
        </div>
        
        {/* Table Body */}
        <div className="divide-y divide-gray-100">
          {Object.keys(cryptoData).length > 0 ? (
            Object.keys(cryptoData)
              .filter(cryptoId => cryptoData[cryptoId]) // Only show cryptos with data
              .map((cryptoId, index) => {
                const data = cryptoData[cryptoId];
                if (!data) return null;

            const isSelected = selectedCrypto === cryptoId;
            const percentageChange = data.change_24h ?? 0;
            const isPositive = percentageChange >= 0;
            const priceChange = priceChanges[cryptoId] || 'none';
            
            // Ensure we have valid percentage data
            const hasValidChange = data.change_24h !== null && data.change_24h !== undefined;
            
            return (
              <div
                key={cryptoId}
                className={`crypto-table-row group transition-all duration-300 ${
                  isSelected ? 'bg-gradient-to-r from-blue-50 via-indigo-50 to-blue-50 dark:from-blue-900/20 dark:via-indigo-900/20 dark:to-blue-900/20 border-l-4 border-blue-500 dark:border-blue-400 shadow-md' : ''
                } ${
                  priceChange === 'up' ? 'bg-green-50 dark:bg-green-900/10' : priceChange === 'down' ? 'bg-red-50 dark:bg-red-900/10' : ''
                } hover:bg-gray-50 dark:hover:bg-gray-800/50`}
                onClick={() => onSelectCrypto(isSelected ? null : cryptoId)}
              >
                <div className="grid grid-cols-12 gap-6 items-center">
                  {/* Name Column */}
                  <div className="col-span-4 flex items-center space-x-4">
                    <div className="flex items-center space-x-3 text-gray-500 text-sm font-bold">
                      <span className="w-8 text-center">{index + 1}</span>
                    </div>
                    
                    <CryptoIcon cryptoId={cryptoId} size="md" />
                    
                    <div>
                      <div className="font-bold text-gray-900 dark:text-gray-100 text-xl">
                        {getCryptoDisplayName(cryptoId)}
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400 font-semibold">
                        {getCryptoSymbol(cryptoId)}
                      </div>
                    </div>
                  </div>

                  {/* Price Column */}
                  <div className="col-span-2 text-right">
                    <div className={`font-bold text-2xl transition-colors duration-300 ${
                      priceChange === 'up' ? 'text-green-600 dark:text-green-400' : 
                      priceChange === 'down' ? 'text-red-600 dark:text-red-400' : 
                      'text-gray-900 dark:text-gray-100'
                    }`}>
                      {formatPrice(data.price)}
                    </div>
                  </div>

                  {/* 24h Change Column */}
                  <div className="col-span-2 text-right">
                    {hasValidChange ? (
                      <div className={`price-change ${
                        isPositive ? 'price-positive' : 'price-negative'
                      }`}>
                        {formatPercentage(percentageChange)}
                      </div>
                    ) : (
                      <div className="text-gray-400 dark:text-gray-500 text-sm">
                        --
                      </div>
                    )}
                  </div>

                  {/* Market Cap Column */}
                  <div className="col-span-2 text-right">
                    <div className="text-xl text-gray-900 dark:text-gray-100 font-bold">
                      {formatMarketCap(data.market_cap)}
                    </div>
                  </div>

                  {/* Volume Column */}
                  <div className="col-span-2 text-right">
                    <div className="text-xl text-gray-900 dark:text-gray-100 font-bold">
                      {formatVolume(data.volume_24h)}
                    </div>
                  </div>
                </div>
              </div>
            );
          })) : (
            <div className="text-center py-12">
              <div className="inline-flex flex-col items-center space-y-4 p-6 rounded-xl bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700">
                {error ? (
                  <>
                    <div className="text-red-500 dark:text-red-400 text-lg font-semibold">
                      Error Loading Prices
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400 max-w-md">
                      {error}
                    </div>
                    <button
                      onClick={reconnect}
                      className="mt-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors text-sm font-semibold"
                    >
                      Retry
                    </button>
                  </>
                ) : (
                  <>
                    <div className="text-gray-500 dark:text-gray-400 text-lg font-semibold">
                      Loading Prices...
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      Please wait while we fetch the latest cryptocurrency prices
                    </div>
                  </>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Selected Crypto Details with Charts */}
      {selectedCrypto && cryptoData[selectedCrypto] && (
        <CoinDetailView
          coinId={selectedCrypto}
          currentData={{
            price: cryptoData[selectedCrypto].price,
            change_24h: cryptoData[selectedCrypto].change_24h || 0,
            market_cap: cryptoData[selectedCrypto].market_cap || 0,
            volume_24h: cryptoData[selectedCrypto].volume_24h || 0,
          }}
          onClose={() => onSelectCrypto(null)}
        />
      )}
    </div>
  );
}

export default RealTimeCryptoGrid;
