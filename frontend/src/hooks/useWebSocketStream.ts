import { useState, useEffect, useRef, useCallback } from 'react';
import { API_BASE_URL } from '../lib/api';

interface StreamData {
  type: string;
  data: Record<string, {
    price: number;
    change_24h: number;
    market_cap: number;
    volume_24h: number;
  }>;
  timestamp: number;
}

interface UseWebSocketStreamOptions {
  enabled?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  fallbackToPolling?: boolean;
  pollingInterval?: number;
}

export function useWebSocketStream(options: UseWebSocketStreamOptions = {}) {
  const {
    enabled = true,
    reconnectInterval = 5000,
    maxReconnectAttempts = 5,
    fallbackToPolling = true,
    pollingInterval = 15000, // 15 seconds for faster updates
  } = options;

  const [data, setData] = useState<StreamData | null>(null);
  const [isConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reconnectCount, setReconnectCount] = useState(0);
  const [isFallbackMode, setIsFallbackMode] = useState(false);
  const [lastAttemptTime, setLastAttemptTime] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const fallbackTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const cleanup = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (fallbackTimeoutRef.current) {
      clearTimeout(fallbackTimeoutRef.current);
      fallbackTimeoutRef.current = null;
    }
  }, []);

  const snapshotUrl = `${API_BASE_URL.replace(/\/$/, '')}/stream/snapshot`;
  const healthUrl = `${API_BASE_URL.replace(/\/$/, '')}/health/quick`;

  const startFallbackPolling = useCallback(() => {
    console.log('Starting fallback polling mode');
    
    const poll = async () => {
      try {
        const response = await fetch(snapshotUrl, {
          signal: AbortSignal.timeout(60000) // 60 second timeout - backend may be slow on first load
        });
        if (response.ok) {
          const streamData: StreamData = await response.json();
          
          // Always set data, even if it's empty - this allows UI to render
          setData(streamData);
          
          // Only set error if there's an error field in the response
          if (streamData.type === 'error_snapshot' || (streamData as any).error) {
            setError((streamData as any).error || 'Failed to fetch price data');
            console.error('Stream snapshot returned error:', (streamData as any).error);
          } else {
            setError(null);
            const dataCount = streamData.data ? Object.keys(streamData.data).length : 0;
            if (dataCount > 0) {
              console.log(`✅ Fallback polling: Data updated successfully (${dataCount} cryptocurrencies)`);
              console.log('Sample data:', Object.keys(streamData.data).slice(0, 3).map(id => ({
                id,
                price: streamData.data[id]?.price
              })));
            } else {
              console.warn(`⚠️ Fallback polling: Received response but no cryptocurrency data (0 items)`);
              console.warn('Response structure:', { type: streamData.type, hasData: !!streamData.data });
            }
          }
        } else {
          const errorText = await response.text().catch(() => 'Unknown error');
          throw new Error(`HTTP ${response.status}: ${errorText}`);
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        console.error('❌ Fallback polling error:', errorMessage);
        
        // Provide more helpful error messages
        let userFriendlyError = errorMessage;
        if (errorMessage.includes('timeout') || errorMessage.includes('timed out')) {
          userFriendlyError = 'Backend is taking too long to respond. Please check if the backend server is running on port 8000.';
        } else if (errorMessage.includes('Failed to fetch') || errorMessage.includes('NetworkError')) {
          userFriendlyError = 'Cannot connect to backend server. Make sure it\'s running on http://127.0.0.1:8000';
        }
        
        setError(userFriendlyError);
        console.error('📡 Backend URL:', snapshotUrl);
        console.error('💡 Tip: Check if the backend is running by visiting http://127.0.0.1:8000/health in your browser');
        
        // Set empty data structure so UI can still render
        setData({
          type: 'error',
          data: {},
          timestamp: Date.now() / 1000,
        });
      }

      if (enabled && isFallbackMode) {
        fallbackTimeoutRef.current = setTimeout(poll, pollingInterval);
      }
    };

    // Start polling immediately
    poll();
  }, [enabled, isFallbackMode, pollingInterval, snapshotUrl]);

  const connectWebSocket = useCallback(() => {
    if (!enabled || wsRef.current?.readyState === WebSocket.CONNECTING) {
      return;
    }

    // Rate limiting: don't attempt too frequently
    const now = Date.now();
    if (now - lastAttemptTime < 10000) { // Wait at least 10 seconds between attempts
      return;
    }
    setLastAttemptTime(now);

    cleanup();
    
    // First check if backend is available before attempting WebSocket
    // Use longer timeout and don't block - just go straight to polling
    fetch(healthUrl, {
      signal: AbortSignal.timeout(30000) // 30 second timeout for health check
    })
      .then(response => {
        if (!response.ok) {
          console.log('Backend not ready, switching to fallback mode');
          setIsFallbackMode(true);
          startFallbackPolling();
          return;
        }
        
        // For now, skip WebSocket and go directly to fallback mode
        // WebSocket has connection issues, fallback polling works perfectly
        console.log('Using fallback polling mode for reliable data streaming');
        setIsFallbackMode(true);
        startFallbackPolling();
        
      })
      .catch(() => {
        console.log('Backend not ready, using fallback polling mode');
        setIsFallbackMode(true);
        startFallbackPolling();
      });
  }, [enabled, reconnectCount, maxReconnectAttempts, reconnectInterval, fallbackToPolling, isFallbackMode, cleanup, startFallbackPolling, healthUrl]);

  useEffect(() => {
    if (!enabled) {
      return cleanup;
    }

    // Always start with fallback polling for immediate data
    // WebSocket can be added later if needed
    if (!isFallbackMode) {
      setIsFallbackMode(true);
    }
    
    // Start polling immediately with a small delay to avoid race conditions
    const timer = setTimeout(() => {
      startFallbackPolling();
    }, 100);

    return () => {
      cleanup();
      clearTimeout(timer);
    };
  }, [enabled, isFallbackMode, startFallbackPolling, cleanup]);

  const reconnect = useCallback(() => {
    setReconnectCount(0);
    setIsFallbackMode(false);
    connectWebSocket();
  }, [connectWebSocket]);

  return {
    data,
    isConnected,
    error,
    reconnectCount,
    isFallbackMode,
    reconnect,
  };
}

export default useWebSocketStream;