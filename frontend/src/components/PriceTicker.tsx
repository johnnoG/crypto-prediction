import { useQuery } from '@tanstack/react-query';
import { useState, useEffect } from 'react';
import { useCryptoPrices, formatPrice, getCryptoSymbol } from '../hooks/useCryptoPrices';
import { apiClient } from '../lib/api';
import CryptoIcon from './CryptoIcon';

const TICKER_CRYPTOS = ['bitcoin', 'ethereum', 'solana', 'cardano', 'polkadot'];

function PriceTicker() {
  const [isVisible, setIsVisible] = useState(true);
  const [lastScrollY, setLastScrollY] = useState(0);

  const { data: prices, isLoading: pricesLoading, isError: pricesError } = useCryptoPrices(
    TICKER_CRYPTOS,
    ['usd'],
    { refetchInterval: 15000 }
  );

  const { data: health, isError: healthError } = useQuery({
    queryKey: ['health'],
    queryFn: () => apiClient.getHealth(),
    refetchInterval: 120000,
    retry: 1,
  });

  const { data: cacheStatus } = useQuery({
    queryKey: ['cache-status'],
    queryFn: () => apiClient.getCacheStatus(),
    refetchInterval: 300000,
    retry: 1,
  });

  const isHealthy = health?.status === 'ok' && !healthError;
  const isCacheHealthy = cacheStatus?.healthy;

  // Scroll detection
  useEffect(() => {
    const handleScroll = () => {
      const currentScrollY = window.scrollY;
      
      // Show ticker when at top or scrolling up
      if (currentScrollY < 100) {
        setIsVisible(true);
      } else if (currentScrollY < lastScrollY) {
        // Scrolling up
        setIsVisible(true);
      } else if (currentScrollY > lastScrollY && currentScrollY > 100) {
        // Scrolling down and past threshold
        setIsVisible(false);
      }
      
      setLastScrollY(currentScrollY);
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, [lastScrollY]);

  if (pricesLoading) {
    return (
      <div className={`sticky top-16 z-40 transition-transform duration-300 ${isVisible ? 'translate-y-0' : '-translate-y-full'}`}>
        <div className="relative overflow-hidden">
          <div className="absolute inset-0 bg-slate-900/95 backdrop-blur-md"></div>
          <div className="relative py-2.5">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex items-center justify-center gap-3">
                <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
                <span className="text-slate-400 font-medium text-sm">Loading live market data...</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (pricesError || !prices) {
    return (
      <div className={`sticky top-16 z-40 transition-transform duration-300 ${isVisible ? 'translate-y-0' : '-translate-y-full'}`}>
        <div className="relative overflow-hidden">
          <div className="absolute inset-0 bg-slate-900/95 backdrop-blur-md"></div>
          <div className="relative py-2.5">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex items-center justify-center gap-3">
                <div className="w-2 h-2 bg-rose-400 rounded-full"></div>
                <span className="text-rose-400 font-medium text-sm">Unable to load market data</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`sticky top-16 z-40 transition-transform duration-300 ${isVisible ? 'translate-y-0' : '-translate-y-full'}`}>
      <div className="relative overflow-hidden">
        {/* Background */}
        <div className="absolute inset-0 bg-slate-900/95 backdrop-blur-md"></div>
        
        {/* Subtle gradient */}
        <div className="absolute inset-0 opacity-30" style={{
          backgroundImage: `
            radial-gradient(at 0% 50%, rgba(34, 197, 94, 0.15) 0px, transparent 50%),
            radial-gradient(at 100% 50%, rgba(59, 130, 246, 0.15) 0px, transparent 50%)
          `
        }}></div>
        
        {/* Bottom border */}
        <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-slate-700 to-transparent"></div>
        
        <div className="relative py-2">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between gap-4">
              
              {/* Left: Status indicators */}
              <div className="flex items-center gap-4 flex-shrink-0">
                {/* API Status */}
                <div className="flex items-center gap-1.5">
                  <div className="relative">
                    <div className={`w-1.5 h-1.5 rounded-full ${isHealthy ? 'bg-emerald-400' : 'bg-rose-400'}`}></div>
                    {isHealthy && (
                      <div className="absolute inset-0 w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping opacity-75"></div>
                    )}
                  </div>
                  <span className="text-[10px] font-medium text-slate-500 uppercase tracking-wide hidden sm:block">
                    {isHealthy ? 'Live' : 'Offline'}
                  </span>
                </div>

                {/* Cache Status - only show on larger screens */}
                <div className="hidden lg:flex items-center gap-1.5">
                  <div className={`w-1.5 h-1.5 rounded-full ${isCacheHealthy ? 'bg-emerald-400' : 'bg-amber-400'}`}></div>
                  <span className="text-[10px] font-medium text-slate-500 uppercase tracking-wide">
                    {cacheStatus?.backend || 'cache'}
                  </span>
                </div>

                {/* Divider */}
                <div className="w-px h-4 bg-slate-700/50"></div>
              </div>
              
              {/* Center: Price ticker */}
              <div className="flex-1 overflow-hidden">
                <div className="flex items-center gap-5 overflow-x-auto scrollbar-hide">
                  {TICKER_CRYPTOS.map((cryptoId, index) => {
                    const price = prices[cryptoId]?.usd;
                    if (!price) return null;

                    return (
                      <div
                        key={cryptoId}
                        className="flex items-center gap-2.5 group cursor-pointer hover:opacity-80 transition-opacity"
                      >
                        {/* Divider */}
                        {index > 0 && (
                          <div className="w-px h-4 bg-slate-700/30 -ml-2.5"></div>
                        )}
                        
                        {/* Crypto icon */}
                        <CryptoIcon cryptoId={cryptoId} size="xs" />
                        
                        {/* Symbol & Price */}
                        <div className="flex items-center gap-2">
                          <span className="text-slate-400 font-semibold text-xs">
                            {getCryptoSymbol(cryptoId)}
                          </span>
                          <span className="font-mono font-bold text-white text-xs tabular-nums">
                            {formatPrice(price)}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
              
              {/* Right: Timestamp */}
              <div className="flex items-center gap-2 flex-shrink-0">
                <div className="w-px h-4 bg-slate-700/50"></div>
                <svg className="w-3 h-3 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="text-[10px] font-mono text-slate-500">
                  {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default PriceTicker;
