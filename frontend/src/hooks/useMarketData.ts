import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../lib/api';

// Default crypto IDs to fetch - Expanded list
const DEFAULT_CRYPTO_IDS = [
  'bitcoin',
  'ethereum',
  'tether',
  'binancecoin',
  'solana',
  'ripple',
  'usd-coin',
  'cardano',
  'avalanche-2',
  'dogecoin',
  'polkadot',
  'matic-network',
  'chainlink',
  'litecoin',
  'uniswap',
  'bitcoin-cash',
  'cosmos',
  'stellar',
  'ethereum-classic',
  'filecoin',
  'monero',
  'aave',
  'algorand',
  'the-sandbox',
  'axie-infinity',
  'decentraland',
  'theta-token',
  'fantom',
  'near',
  'optimism',
  'arbitrum'
];

// Display name and symbol mapping
const CRYPTO_INFO: Record<string, { name: string; symbol: string }> = {
  bitcoin: { name: 'Bitcoin', symbol: 'BTC' },
  ethereum: { name: 'Ethereum', symbol: 'ETH' },
  tether: { name: 'Tether', symbol: 'USDT' },
  binancecoin: { name: 'BNB', symbol: 'BNB' },
  solana: { name: 'Solana', symbol: 'SOL' },
  ripple: { name: 'XRP', symbol: 'XRP' },
  'usd-coin': { name: 'USD Coin', symbol: 'USDC' },
  cardano: { name: 'Cardano', symbol: 'ADA' },
  'avalanche-2': { name: 'Avalanche', symbol: 'AVAX' },
  dogecoin: { name: 'Dogecoin', symbol: 'DOGE' },
  polkadot: { name: 'Polkadot', symbol: 'DOT' },
  'matic-network': { name: 'Polygon', symbol: 'MATIC' },
  chainlink: { name: 'Chainlink', symbol: 'LINK' },
  litecoin: { name: 'Litecoin', symbol: 'LTC' },
  uniswap: { name: 'Uniswap', symbol: 'UNI' },
  'bitcoin-cash': { name: 'Bitcoin Cash', symbol: 'BCH' },
  cosmos: { name: 'Cosmos', symbol: 'ATOM' },
  stellar: { name: 'Stellar', symbol: 'XLM' },
  'ethereum-classic': { name: 'Ethereum Classic', symbol: 'ETC' },
  filecoin: { name: 'Filecoin', symbol: 'FIL' },
  monero: { name: 'Monero', symbol: 'XMR' },
  aave: { name: 'Aave', symbol: 'AAVE' },
  algorand: { name: 'Algorand', symbol: 'ALGO' },
  'the-sandbox': { name: 'The Sandbox', symbol: 'SAND' },
  'axie-infinity': { name: 'Axie Infinity', symbol: 'AXS' },
  decentraland: { name: 'Decentraland', symbol: 'MANA' },
  'theta-token': { name: 'Theta Network', symbol: 'THETA' },
  fantom: { name: 'Fantom', symbol: 'FTM' },
  near: { name: 'NEAR Protocol', symbol: 'NEAR' },
  optimism: { name: 'Optimism', symbol: 'OP' },
  arbitrum: { name: 'Arbitrum', symbol: 'ARB' },
};

export function useMarketData(cryptoIds: string[] = DEFAULT_CRYPTO_IDS) {
  return useQuery({
    queryKey: ['marketData', cryptoIds],
    queryFn: async () => {
      try {
        const data = await apiClient.getMarketData(cryptoIds);
        // If we got empty data (from cache fallback), that's okay - return it
        return data || {};
      } catch (error) {
        console.error('Failed to fetch market data:', error);
        // Return empty object - UI will show loading/empty state gracefully
        return {};
      }
    },
    staleTime: 10 * 60 * 1000, // 10 minutes
    refetchInterval: 10 * 60 * 1000, // Refetch every 10 minutes to reduce rate limiting
    retry: 2, // Reduced retries since we have cache fallback
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
  });
}

// Utility functions for formatting
export function formatPrice(price: number): string {
  if (price === 0) return '--';
  if (price < 0.01) return `$${price.toFixed(6)}`;
  if (price < 1) return `$${price.toFixed(4)}`;
  if (price < 100) return `$${price.toFixed(2)}`;
  if (price < 10000) return `$${price.toFixed(2)}`;
  return `$${(price / 1000).toFixed(1)}K`;
}

export function formatPercentage(percentage: number | null | undefined): string {
  if (percentage === null || percentage === undefined) return '--';
  if (percentage === 0) return '0.00%';
  const sign = percentage >= 0 ? '+' : '';
  return `${sign}${percentage.toFixed(2)}%`;
}

export function formatMarketCap(marketCap: number): string {
  if (marketCap === 0) return '--';
  if (marketCap < 1e9) return `$${(marketCap / 1e6).toFixed(1)}M`;
  if (marketCap < 1e12) return `$${(marketCap / 1e9).toFixed(1)}B`;
  return `$${(marketCap / 1e12).toFixed(1)}T`;
}

export function formatVolume(volume: number): string {
  if (volume === 0) return '--';
  if (volume < 1e9) return `$${(volume / 1e6).toFixed(1)}M`;
  if (volume < 1e12) return `$${(volume / 1e9).toFixed(1)}B`;
  return `$${(volume / 1e12).toFixed(1)}T`;
}

export function getCryptoInfo(cryptoId: string) {
  return CRYPTO_INFO[cryptoId] || { name: cryptoId, symbol: cryptoId.toUpperCase() };
}

export function getCryptoIcon(cryptoId: string): string {
  // Return symbol as fallback - actual icons are handled by CryptoIcon component
  return CRYPTO_INFO[cryptoId]?.symbol || cryptoId.toUpperCase().replace(/-/g, '').substring(0, 3);
}
