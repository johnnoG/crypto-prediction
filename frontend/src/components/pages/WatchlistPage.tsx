import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../lib/api';
import { useToast } from '../../hooks/use-toast';
import { Button } from '../ui/button';
import { Card } from '../ui/card';
import { ConfirmationDialog } from '../ui/confirmation-dialog';
import CryptoIcon from '../CryptoIcon';
import { formatPrice } from '../../hooks/useCryptoPrices';

interface WatchlistItem {
  id: number;
  crypto_symbol: string;
  crypto_name: string;
  crypto_id: string;
  notes?: string;
  is_favorite: boolean;
  notification_enabled: boolean;
  created_at: string;
  updated_at: string;
}

const WatchlistPage: React.FC = () => {
  const { toast } = useToast();
  const [removeDialogOpen, setRemoveDialogOpen] = useState(false);
  const [itemToRemove, setItemToRemove] = useState<WatchlistItem | null>(null);
  const [isRemoving, setIsRemoving] = useState(false);

  // Fetch watchlist
  const { data: watchlist, isLoading, error, refetch } = useQuery<WatchlistItem[]>({
    queryKey: ['watchlist'],
    queryFn: async () => {
      const result = await apiClient.getWatchlist();
      console.log('Watchlist API response:', result);
      return result;
    },
    refetchInterval: 300000, // 5 minutes
  });

  // Fetch current prices for watchlist items
  const watchlistSymbols = watchlist?.map(item => item.crypto_id) || [];
  const { data: priceData } = useQuery({
    queryKey: ['watchlist-prices', watchlistSymbols.join(',')],
    queryFn: async () => {
      if (watchlistSymbols.length === 0) return {};
      return apiClient.getMultiplePrices(watchlistSymbols);
    },
    refetchInterval: 60000, // 1 minute
    enabled: watchlistSymbols.length > 0,
  });

  const handleRemoveItem = (item: WatchlistItem) => {
    setItemToRemove(item);
    setRemoveDialogOpen(true);
  };

  const confirmRemove = async () => {
    if (!itemToRemove) return;

    setIsRemoving(true);
    try {
      await apiClient.removeFromWatchlist(itemToRemove.id);
      toast({
        title: "Removed from Watchlist",
        description: `${itemToRemove.crypto_name} has been removed from your watchlist.`,
      });
      refetch(); // Refresh the list
      setRemoveDialogOpen(false);
      setItemToRemove(null);
    } catch {
      toast({
        title: "Error",
        description: "Failed to remove item from watchlist. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsRemoving(false);
    }
  };

  const handleToggleFavorite = async (item: WatchlistItem) => {
    try {
      await apiClient.updateWatchlistItem(item.id, {
        is_favorite: !item.is_favorite
      });
      toast({
        title: item.is_favorite ? "Removed from Favorites" : "Added to Favorites",
        description: `${item.crypto_name} has been ${item.is_favorite ? 'removed from' : 'added to'} your favorites.`,
      });
      refetch(); // Refresh the list
    } catch {
      toast({
        title: "Error",
        description: "Failed to update favorite status. Please try again.",
        variant: "destructive",
      });
    }
  };

  const handleToggleNotifications = async (item: WatchlistItem) => {
    try {
      await apiClient.updateWatchlistItem(item.id, {
        notification_enabled: !item.notification_enabled
      });
      toast({
        title: "Notifications Updated",
        description: `Notifications ${item.notification_enabled ? 'disabled' : 'enabled'} for ${item.crypto_name}.`,
      });
      refetch(); // Refresh the list
    } catch {
      toast({
        title: "Error",
        description: "Failed to update notification settings. Please try again.",
        variant: "destructive",
      });
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-950 text-white p-6">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-3xl font-bold text-white mb-8">My Watchlist</h1>
          <div className="grid gap-4">
            {[...Array(3)].map((_, i) => (
              <Card key={i} className="bg-gray-900 border-gray-800 animate-pulse">
                <div className="p-6">
                  <div className="h-4 bg-gray-700 rounded w-1/4 mb-4"></div>
                  <div className="h-6 bg-gray-700 rounded w-1/2"></div>
                </div>
              </Card>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-950 text-white p-6">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-3xl font-bold text-white mb-8">My Watchlist</h1>
          <Card className="bg-gray-900 border-gray-800">
            <div className="p-6 text-center">
              <p className="text-red-400 mb-4">Failed to load watchlist</p>
              <Button onClick={() => refetch()} className="bg-blue-600 hover:bg-blue-700">
                Try Again
              </Button>
            </div>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">My Watchlist</h1>
          <p className="text-gray-400">
            Track your favorite cryptocurrencies and their latest prices
          </p>
        </div>

        {/* Debug Info */}
        {error && <div className="mb-4 p-4 bg-red-100 text-red-800 rounded">Error: {error.message}</div>}

        {/* Watchlist Items */}
        {!watchlist || watchlist.length === 0 ? (
          <Card className="bg-gray-900 border-gray-800">
            <div className="p-8 text-center">
              <div className="w-16 h-16 bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">Your watchlist is empty</h3>
              <p className="text-gray-400 mb-6">
                Start adding cryptocurrencies to your watchlist from the forecasts page
              </p>
            </div>
          </Card>
        ) : (
          <div className="grid gap-4">
            {watchlist.map((item) => {
              const priceInfo = priceData?.[item.crypto_id];
              const currentPrice = priceInfo?.usd || 0;
              const priceChange24h = priceInfo?.usd_24h_change || 0;
              const isPositive = priceChange24h >= 0;

              return (
                <Card key={item.id} className="bg-gray-900 border-gray-800 hover:border-gray-700 transition-colors">
                  <div className="p-6">
                    <div className="flex items-center justify-between">
                      {/* Crypto Info */}
                      <div className="flex items-center space-x-4">
                        <CryptoIcon cryptoId={item.crypto_id} className="w-12 h-12" />
                        <div>
                          <div className="flex items-center space-x-2">
                            <h3 className="text-lg font-semibold text-white">
                              {item.crypto_name}
                            </h3>
                            <span className="text-sm text-gray-400 uppercase">
                              {item.crypto_symbol}
                            </span>
                            {item.is_favorite && (
                              <svg className="w-4 h-4 text-yellow-400 fill-current" viewBox="0 0 20 20">
                                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                              </svg>
                            )}
                          </div>
                          {item.notes && (
                            <p className="text-sm text-gray-400 mt-1">{item.notes}</p>
                          )}
                          <p className="text-xs text-gray-500 mt-1">
                            Added {new Date(item.created_at).toLocaleDateString()}
                          </p>
                        </div>
                      </div>

                      {/* Price Info */}
                      <div className="text-right">
                        {currentPrice > 0 ? (
                          <>
                            <div className="text-xl font-bold text-white">
                              {formatPrice(currentPrice)}
                            </div>
                            <div className={`text-sm font-medium ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                              {isPositive ? '+' : ''}{priceChange24h.toFixed(2)}%
                            </div>
                          </>
                        ) : (
                          <div className="text-gray-400">Price unavailable</div>
                        )}
                      </div>

                      {/* Actions */}
                      <div className="flex items-center space-x-2 ml-6">
                        {/* Favorite Toggle */}
                        <button
                          onClick={() => handleToggleFavorite(item)}
                          className={`p-2 rounded-lg transition-colors ${
                            item.is_favorite
                              ? 'text-yellow-400 hover:bg-yellow-400/10'
                              : 'text-gray-400 hover:text-yellow-400 hover:bg-gray-800'
                          }`}
                          title={item.is_favorite ? 'Remove from favorites' : 'Add to favorites'}
                        >
                          <svg className="w-5 h-5" fill={item.is_favorite ? 'currentColor' : 'none'} stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
                          </svg>
                        </button>

                        {/* Notifications Toggle */}
                        <button
                          onClick={() => handleToggleNotifications(item)}
                          className={`p-2 rounded-lg transition-colors ${
                            item.notification_enabled
                              ? 'text-blue-400 hover:bg-blue-400/10'
                              : 'text-gray-400 hover:text-blue-400 hover:bg-gray-800'
                          }`}
                          title={item.notification_enabled ? 'Disable notifications' : 'Enable notifications'}
                        >
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            {item.notification_enabled ? (
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-5 5v-5zm0-8l-5-5v5h5zm-8 8v2a2 2 0 01-2-2h2zm-4-8a2 2 0 012-2v2H3z" />
                            ) : (
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5.586 15H4a1 1 0 01-.707-1.707l8-8a1 1 0 011.414 0l8 8A1 1 0 0120 15h-1.586l-8-8-8 8z" />
                            )}
                          </svg>
                        </button>

                        {/* Remove Button */}
                        <button
                          onClick={() => handleRemoveItem(item)}
                          className="p-2 rounded-lg text-red-400 hover:text-red-300 hover:bg-red-400/10 transition-colors"
                          title="Remove from watchlist"
                        >
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                        </button>
                      </div>
                    </div>
                  </div>
                </Card>
              );
            })}
          </div>
        )}

        {/* Confirmation Dialog */}
        <ConfirmationDialog
          isOpen={removeDialogOpen}
          onClose={() => setRemoveDialogOpen(false)}
          onConfirm={confirmRemove}
          title="Remove from Watchlist"
          description={`Are you sure you want to remove ${itemToRemove?.crypto_name} from your watchlist? This action cannot be undone.`}
          confirmText="Remove"
          cancelText="Cancel"
          isLoading={isRemoving}
          variant="destructive"
        />
      </div>
    </div>
  );
};

export default WatchlistPage;