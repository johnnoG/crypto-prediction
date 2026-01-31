import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../lib/api';

export const DEFAULT_CRYPTO_IDS = [
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

export function useCryptoPrices(
  cryptoIds: string[] = DEFAULT_CRYPTO_IDS,
  vsCurrencies: string[] = ['usd'],
  options?: {
    enabled?: boolean;
    refetchInterval?: number;
  }
) {
  return useQuery({
    queryKey: ['crypto-prices', cryptoIds, vsCurrencies],
    queryFn: async () => {
      try {
        const data = await apiClient.getPrices(cryptoIds, vsCurrencies);
        // If we got empty data (from cache fallback), that's okay - return it
        return data || {};
      } catch (error) {
        console.error('Failed to fetch crypto prices:', error);
        // Return empty object - UI will show loading/empty state gracefully
        return {};
      }
    },
    enabled: options?.enabled ?? true,
    refetchInterval: options?.refetchInterval ?? 120000, // 2 minutes to reduce API calls and rate limiting
    staleTime: 60000, // 1 minute
    retry: 2, // Reduced retries since we have cache fallback
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
  });
}

export function useSingleCryptoPrice(
  cryptoId: string,
  vsCurrency: string = 'usd',
  options?: {
    enabled?: boolean;
    refetchInterval?: number;
  }
) {
  return useQuery({
    queryKey: ['crypto-price', cryptoId, vsCurrency],
    queryFn: () => apiClient.getCryptoPrice(cryptoId, vsCurrency),
    enabled: options?.enabled ?? true,
    refetchInterval: options?.refetchInterval ?? 120000, // 2 minutes
    staleTime: 60000, // 1 minute
  });
}

// Helper function to format price
export function formatPrice(price: number, currency: string = 'USD'): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    minimumFractionDigits: price < 1 ? 4 : 2,
    maximumFractionDigits: price < 1 ? 6 : 2,
  }).format(price);
}

// Helper function to format percentage change
export function formatPercentage(value: number): string {
  const formatted = new Intl.NumberFormat('en-US', {
    style: 'percent',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value / 100);
  
  return value >= 0 ? `+${formatted}` : formatted;
}

// Helper to get crypto display name
export function getCryptoDisplayName(cryptoId: string): string {
  const names: Record<string, string> = {
    bitcoin: 'Bitcoin',
    ethereum: 'Ethereum',
    tether: 'Tether',
    binancecoin: 'BNB',
    solana: 'Solana',
    ripple: 'XRP',
    'usd-coin': 'USD Coin',
    cardano: 'Cardano',
    'avalanche-2': 'Avalanche',
    dogecoin: 'Dogecoin',
    polkadot: 'Polkadot',
    'matic-network': 'Polygon',
    chainlink: 'Chainlink',
    litecoin: 'Litecoin',
    uniswap: 'Uniswap',
    'bitcoin-cash': 'Bitcoin Cash',
    cosmos: 'Cosmos',
    stellar: 'Stellar',
    'ethereum-classic': 'Ethereum Classic',
    filecoin: 'Filecoin',
    monero: 'Monero',
    aave: 'Aave',
    algorand: 'Algorand',
    'the-sandbox': 'The Sandbox',
    'axie-infinity': 'Axie Infinity',
    decentraland: 'Decentraland',
    'theta-token': 'Theta Network',
    fantom: 'Fantom',
    near: 'NEAR Protocol',
    optimism: 'Optimism',
    arbitrum: 'Arbitrum',
  };
  
  return names[cryptoId] || (cryptoId ? cryptoId.charAt(0).toUpperCase() + cryptoId.slice(1).replace(/-/g, ' ') : 'Unknown');
}

// Helper to get crypto symbol
export function getCryptoSymbol(cryptoId: string): string {
  const symbols: Record<string, string> = {
    bitcoin: 'BTC',
    ethereum: 'ETH',
    tether: 'USDT',
    binancecoin: 'BNB',
    solana: 'SOL',
    ripple: 'XRP',
    'usd-coin': 'USDC',
    cardano: 'ADA',
    'avalanche-2': 'AVAX',
    dogecoin: 'DOGE',
    polkadot: 'DOT',
    'matic-network': 'MATIC',
    chainlink: 'LINK',
    litecoin: 'LTC',
    uniswap: 'UNI',
    'bitcoin-cash': 'BCH',
    cosmos: 'ATOM',
    stellar: 'XLM',
    'ethereum-classic': 'ETC',
    filecoin: 'FIL',
    monero: 'XMR',
    aave: 'AAVE',
    algorand: 'ALGO',
    'the-sandbox': 'SAND',
    'axie-infinity': 'AXS',
    decentraland: 'MANA',
    'theta-token': 'THETA',
    fantom: 'FTM',
    near: 'NEAR',
    optimism: 'OP',
    arbitrum: 'ARB',
  };
  
  return symbols[cryptoId] || (cryptoId ? cryptoId.toUpperCase().replace(/-/g, '') : 'UNKNOWN');
}
