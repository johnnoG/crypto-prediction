import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../lib/api';

export interface PortfolioHolding {
  id: number;
  crypto_id: string;
  crypto_symbol: string;
  crypto_name: string;
  amount: number;
  avg_buy_price: number;
  created_at: string;
  updated_at: string;
}

export function usePortfolio() {
  return useQuery<PortfolioHolding[]>({
    queryKey: ['portfolio'],
    queryFn: () => apiClient.getPortfolio(),
    staleTime: 30000,
    retry: 2,
  });
}

export function useAddHolding() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: {
      crypto_id: string;
      crypto_symbol: string;
      crypto_name: string;
      amount: number;
      avg_buy_price: number;
    }) => apiClient.addPortfolioHolding(data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['portfolio'] }),
  });
}

export function useUpdateHolding() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, data }: { id: number; data: { amount?: number; avg_buy_price?: number } }) =>
      apiClient.updatePortfolioHolding(id, data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['portfolio'] }),
  });
}

export function useDeleteHolding() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => apiClient.deletePortfolioHolding(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['portfolio'] }),
  });
}
