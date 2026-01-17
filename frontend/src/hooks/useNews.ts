import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../lib/api';

export function useNews(
  page: number = 1,
  limit: number = 20,
  options?: {
    enabled?: boolean;
  }
) {
  return useQuery({
    queryKey: ['news', page, limit],
    queryFn: async () => {
      try {
        return await apiClient.getNews({ page, limit });
      } catch (error) {
        console.error('Failed to fetch news:', error);
        // Return empty news response instead of failing
        return {
          items: [],
          page: 1,
          limit: 20,
          total: 0,
          total_pages: 0,
          realtime: false
        };
      }
    },
    enabled: options?.enabled ?? true,
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchInterval: 5 * 60 * 1000, // Refetch every 5 minutes
    retry: 2,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
  });
}

// Helper to format news date
export function formatNewsDate(dateString?: string): string {
  if (!dateString) return 'Unknown date';
  
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMinutes = Math.floor(diffMs / (1000 * 60));
  const diffHours = Math.floor(diffMinutes / 60);
  const diffDays = Math.floor(diffHours / 24);
  
  if (diffMinutes < 1) return 'Just now';
  if (diffMinutes < 60) return `${diffMinutes}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined,
  });
}

// Helper to truncate news content
export function truncateContent(content: string, maxLength: number = 150): string {
  if (content.length <= maxLength) return content;
  return content.slice(0, maxLength).trim() + '...';
}
