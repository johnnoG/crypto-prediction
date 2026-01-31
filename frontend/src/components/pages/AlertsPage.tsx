import React, { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../../lib/api';
import { useToast } from '../../hooks/use-toast';
import { Button } from '../ui/button';
import { Card } from '../ui/card';
import { ConfirmationDialog } from '../ui/confirmation-dialog';
import CryptoIcon from '../CryptoIcon';
import { formatPrice } from '../../hooks/useCryptoPrices';

interface AlertItem {
  id: number;
  crypto_symbol: string;
  crypto_name: string;
  alert_type: string;
  target_price?: number;
  condition?: string;
  status: string;
  message?: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  triggered_at?: string;
  expires_at?: string;
}

const AlertsPage: React.FC = () => {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [itemToDelete, setItemToDelete] = useState<AlertItem | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  // Fetch alerts
  const { data: alerts, isLoading, error, refetch } = useQuery<AlertItem[]>({
    queryKey: ['alerts'],
    queryFn: async () => {
      const result = await apiClient.getAlerts();
      console.log('Alerts API response:', result);
      return result;
    },
    refetchInterval: 300000, // 5 minutes
  });

  const handleDeleteAlert = (item: AlertItem) => {
    setItemToDelete(item);
    setDeleteDialogOpen(true);
  };

  const confirmDelete = async () => {
    if (!itemToDelete) return;

    setIsDeleting(true);
    try {
      await apiClient.deleteAlert(itemToDelete.id);
      toast({
        title: "Alert Deleted",
        description: `Alert for ${itemToDelete.crypto_name} has been deleted.`,
      });
      queryClient.invalidateQueries({ queryKey: ['alerts'] });
      setDeleteDialogOpen(false);
      setItemToDelete(null);
    } catch {
      toast({
        title: "Error",
        description: "Failed to delete alert. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsDeleting(false);
    }
  };

  const handleToggleAlert = async (item: AlertItem) => {
    try {
      // For now, we'll delete the alert when toggling off, and you'd need to recreate to toggle on
      if (item.is_active) {
        await apiClient.deleteAlert(item.id);
        toast({
          title: "Alert Disabled",
          description: `Alert for ${item.crypto_name} has been disabled.`,
        });
      }
      queryClient.invalidateQueries({ queryKey: ['alerts'] });
    } catch {
      toast({
        title: "Error",
        description: "Failed to update alert. Please try again.",
        variant: "destructive",
      });
    }
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'active':
        return 'text-green-400 bg-green-900/30';
      case 'triggered':
        return 'text-yellow-400 bg-yellow-900/30';
      case 'cancelled':
        return 'text-red-400 bg-red-900/30';
      default:
        return 'text-gray-400 bg-gray-900/30';
    }
  };

  const getAlertTypeDisplay = (alertType: string) => {
    switch (alertType.toLowerCase()) {
      case 'price_target':
        return 'Price Alert';
      case 'forecast_change':
        return 'Forecast Alert';
      case 'volatility':
        return 'Volatility Alert';
      default:
        return alertType;
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-950 text-white p-6">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-3xl font-bold text-white mb-8">My Alerts</h1>
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
          <h1 className="text-3xl font-bold text-white mb-8">My Alerts</h1>
          <Card className="bg-gray-900 border-gray-800">
            <div className="p-6 text-center">
              <p className="text-red-400 mb-4">Failed to load alerts</p>
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
          <h1 className="text-3xl font-bold text-white mb-2">My Alerts</h1>
          <p className="text-gray-400">
            Manage your price alerts and notifications
          </p>
        </div>

        {/* Debug Info */}
        {error && <div className="mb-4 p-4 bg-red-100 text-red-800 rounded">Error: {error.message}</div>}

        {/* Alert Items */}
        {!alerts || alerts.length === 0 ? (
          <Card className="bg-gray-900 border-gray-800">
            <div className="p-8 text-center">
              <div className="w-16 h-16 bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-5 5v-5zm0-8l-5-5v5h5z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">No alerts set</h3>
              <p className="text-gray-400 mb-6">
                Start setting alerts from the forecasts page to track price changes
              </p>
            </div>
          </Card>
        ) : (
          <div className="grid gap-4">
            {alerts.map((item) => {
              return (
                <Card key={item.id} className="bg-gray-900 border-gray-800 hover:border-gray-700 transition-colors">
                  <div className="p-6">
                    <div className="flex items-center justify-between">
                      {/* Alert Info */}
                      <div className="flex items-center space-x-4">
                        <CryptoIcon cryptoId={item.crypto_symbol.toLowerCase()} className="w-12 h-12" />
                        <div>
                          <div className="flex items-center space-x-3">
                            <h3 className="text-lg font-semibold text-white">
                              {item.crypto_name}
                            </h3>
                            <span className="text-sm text-gray-400 uppercase">
                              {item.crypto_symbol}
                            </span>
                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(item.status)}`}>
                              {item.status}
                            </span>
                            {!item.is_active && (
                              <span className="px-2 py-1 rounded-full text-xs font-medium text-gray-400 bg-gray-700">
                                Inactive
                              </span>
                            )}
                          </div>
                          <div className="mt-1">
                            <span className="text-sm text-blue-400 font-medium">
                              {getAlertTypeDisplay(item.alert_type)}
                            </span>
                            {item.target_price && (
                              <span className="text-sm text-gray-400 ml-2">
                                • Target: {formatPrice(item.target_price)}
                              </span>
                            )}
                            {item.condition && (
                              <span className="text-sm text-gray-400 ml-2">
                                • {item.condition}
                              </span>
                            )}
                          </div>
                          {item.message && (
                            <p className="text-sm text-gray-400 mt-1">{item.message}</p>
                          )}
                          <p className="text-xs text-gray-500 mt-1">
                            Created {new Date(item.created_at).toLocaleDateString()}
                          </p>
                          {item.triggered_at && (
                            <p className="text-xs text-yellow-400 mt-1">
                              Triggered {new Date(item.triggered_at).toLocaleDateString()}
                            </p>
                          )}
                        </div>
                      </div>

                      {/* Actions */}
                      <div className="flex items-center space-x-2 ml-6">
                        {/* Toggle Active/Inactive */}
                        <button
                          onClick={() => handleToggleAlert(item)}
                          className={`p-2 rounded-lg transition-colors ${
                            item.is_active
                              ? 'text-green-400 hover:bg-green-400/10'
                              : 'text-gray-400 hover:text-green-400 hover:bg-gray-800'
                          }`}
                          title={item.is_active ? 'Disable alert' : 'Enable alert'}
                        >
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            {item.is_active ? (
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-5 5v-5zm0-8l-5-5v5h5z" />
                            ) : (
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728L5.636 5.636m12.728 12.728L18.364 5.636" />
                            )}
                          </svg>
                        </button>

                        {/* Delete Button */}
                        <button
                          onClick={() => handleDeleteAlert(item)}
                          className="p-2 rounded-lg text-red-400 hover:text-red-300 hover:bg-red-400/10 transition-colors"
                          title="Delete alert"
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
          isOpen={deleteDialogOpen}
          onClose={() => setDeleteDialogOpen(false)}
          onConfirm={confirmDelete}
          title="Delete Alert"
          description={`Are you sure you want to delete the alert for ${itemToDelete?.crypto_name}? This action cannot be undone.`}
          confirmText="Delete"
          cancelText="Cancel"
          isLoading={isDeleting}
          variant="destructive"
        />
      </div>
    </div>
  );
};

export default AlertsPage;