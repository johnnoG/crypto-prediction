import { useQuery, UseQueryOptions, QueryKey } from '@tanstack/react-query';
import { useState, useCallback } from 'react';

// Enhanced query hook with better error handling and retry logic
export function useEnhancedQuery<TData = unknown, TError = Error>(
  options: UseQueryOptions<TData, TError> & {
    queryKey: QueryKey;
    queryFn: () => Promise<TData>;
  }
) {
  const [retryAttempts, setRetryAttempts] = useState(0);
  const [lastError, setLastError] = useState<TError | null>(null);

  const enhancedOptions: UseQueryOptions<TData, TError> = {
    ...options,
    retry: (failureCount, error) => {
      setRetryAttempts(failureCount);
      setLastError(error);
      
      // Custom retry logic based on error type
      if (error instanceof Error) {
        // Don't retry on 404s or authentication errors
        if (error.message.includes('404') || error.message.includes('401')) {
          return false;
        }
        
        // Retry network errors up to 3 times
        if (error.message.includes('NetworkError') || error.message.includes('fetch')) {
          return failureCount < 3;
        }
        
        // Retry server errors up to 2 times
        if (error.message.includes('500') || error.message.includes('502') || error.message.includes('503')) {
          return failureCount < 2;
        }
      }
      
      // Default retry behavior
      return failureCount < (options.retry as number || 1);
    },
    retryDelay: (attemptIndex) => {
      // Exponential backoff with jitter
      const baseDelay = Math.min(1000 * Math.pow(2, attemptIndex), 30000);
      const jitter = Math.random() * 1000;
      return baseDelay + jitter;
    },
    onError: (error) => {
      setLastError(error);
      console.error(`Query error for ${JSON.stringify(options.queryKey)}:`, error);
      options.onError?.(error);
    },
    onSuccess: (data) => {
      setRetryAttempts(0);
      setLastError(null);
      options.onSuccess?.(data);
    }
  };

  const query = useQuery(enhancedOptions);

  const manualRetry = useCallback(() => {
    setRetryAttempts(0);
    setLastError(null);
    query.refetch();
  }, [query]);

  const getErrorType = useCallback((error: TError | null): string => {
    if (!error || !(error instanceof Error)) return 'unknown';
    
    if (error.message.includes('NetworkError') || error.message.includes('fetch')) {
      return 'network';
    }
    
    if (error.message.includes('404')) return 'not_found';
    if (error.message.includes('401') || error.message.includes('403')) return 'auth';
    if (error.message.includes('500') || error.message.includes('502') || error.message.includes('503')) return 'server';
    if (error.message.includes('timeout')) return 'timeout';
    
    return 'generic';
  }, []);

  return {
    ...query,
    retryAttempts,
    lastError,
    errorType: getErrorType(lastError),
    manualRetry,
    isNetworkError: getErrorType(lastError) === 'network',
    isServerError: getErrorType(lastError) === 'server',
    isAuthError: getErrorType(lastError) === 'auth',
    isTimeoutError: getErrorType(lastError) === 'timeout',
  };
}

// Hook for handling WebSocket connection with enhanced error handling
export function useWebSocketConnection(url: string, options?: {
  maxReconnectAttempts?: number;
  reconnectInterval?: number;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Event) => void;
  onMessage?: (data: any) => void;
}) {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [connectionState, setConnectionState] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const [lastError, setLastError] = useState<string | null>(null);

  const connect = useCallback(() => {
    if (socket?.readyState === WebSocket.CONNECTING) return;
    
    try {
      setConnectionState('connecting');
      const ws = new WebSocket(url);
      
      ws.onopen = () => {
        setConnectionState('connected');
        setReconnectAttempts(0);
        setLastError(null);
        options?.onConnect?.();
      };
      
      ws.onclose = (event) => {
        setConnectionState('disconnected');
        setSocket(null);
        options?.onDisconnect?.();
        
        // Auto-reconnect logic
        const maxAttempts = options?.maxReconnectAttempts || 5;
        if (reconnectAttempts < maxAttempts && event.code !== 1000) { // 1000 = normal closure
          setTimeout(() => {
            setReconnectAttempts(prev => prev + 1);
            connect();
          }, options?.reconnectInterval || 5000);
        }
      };
      
      ws.onerror = (error) => {
        setConnectionState('error');
        setLastError('WebSocket connection error');
        options?.onError?.(error);
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          options?.onMessage?.(data);
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };
      
      setSocket(ws);
    } catch (error) {
      setConnectionState('error');
      setLastError(error instanceof Error ? error.message : 'Unknown WebSocket error');
    }
  }, [url, socket, reconnectAttempts, options]);

  const disconnect = useCallback(() => {
    if (socket) {
      socket.close(1000, 'Manual disconnect');
      setSocket(null);
      setConnectionState('disconnected');
    }
  }, [socket]);

  const send = useCallback((data: any) => {
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(data));
      return true;
    }
    return false;
  }, [socket]);

  return {
    socket,
    connectionState,
    reconnectAttempts,
    lastError,
    connect,
    disconnect,
    send,
    isConnected: connectionState === 'connected',
    isConnecting: connectionState === 'connecting',
    isDisconnected: connectionState === 'disconnected',
    hasError: connectionState === 'error',
  };
}

