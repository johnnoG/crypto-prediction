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
    pollingInterval = 15000,
  } = options;

  const [data, setData] = useState<StreamData | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reconnectCount, setReconnectCount] = useState(0);
  const [isFallbackMode, setIsFallbackMode] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fallbackTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Refs so callbacks always see current values without re-creating
  const reconnectCountRef = useRef(0);
  const isFallbackRef = useRef(false);
  const enabledRef = useRef(enabled);
  // Forward-ref so onclose can call connect() recursively
  const connectRef = useRef<(() => void) | null>(null);

  useEffect(() => { enabledRef.current = enabled; }, [enabled]);

  const base = API_BASE_URL.replace(/\/$/, '');
  const snapshotUrl = `${base}/api/stream/snapshot`;
  const wsUrl = base
    .replace(/^https:/, 'wss:')
    .replace(/^http:/, 'ws:')
    + '/api/stream/ws';

  const cleanup = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.onclose = null; // prevent onclose from firing reconnect after intentional close
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

  const startFallbackPolling = useCallback(() => {
    if (!fallbackToPolling) return;
    isFallbackRef.current = true;
    setIsFallbackMode(true);
    console.log('[Stream] WebSocket unavailable — starting REST fallback polling');

    const poll = async () => {
      if (!enabledRef.current || !isFallbackRef.current) return;
      try {
        const response = await fetch(snapshotUrl, {
          signal: AbortSignal.timeout(60000),
        });
        if (response.ok) {
          const streamData: StreamData = await response.json();
          setData(streamData);
          if (streamData.type === 'error_snapshot' || (streamData as any).error) {
            setError((streamData as any).error || 'Failed to fetch price data');
          } else {
            setError(null);
            const count = streamData.data ? Object.keys(streamData.data).length : 0;
            console.log(`[Poll] Updated ${count} coins`);
          }
        } else {
          throw new Error(`HTTP ${response.status}`);
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Unknown error';
        console.error('[Poll] Error:', msg);
        setError(msg);
        setData({ type: 'error', data: {}, timestamp: Date.now() / 1000 });
      }
      if (enabledRef.current && isFallbackRef.current) {
        fallbackTimeoutRef.current = setTimeout(poll, pollingInterval);
      }
    };

    poll();
  }, [fallbackToPolling, pollingInterval, snapshotUrl]);

  const connect = useCallback(() => {
    if (!enabledRef.current) return;
    if (
      wsRef.current?.readyState === WebSocket.OPEN ||
      wsRef.current?.readyState === WebSocket.CONNECTING
    ) return;

    console.log(`[WS] Connecting to ${wsUrl}`);

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('[WS] Connected');
        setIsConnected(true);
        setError(null);
        // If fallback was running, stop it
        isFallbackRef.current = false;
        setIsFallbackMode(false);
        if (fallbackTimeoutRef.current) {
          clearTimeout(fallbackTimeoutRef.current);
          fallbackTimeoutRef.current = null;
        }
        reconnectCountRef.current = 0;
        setReconnectCount(0);
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data as string);

          if (msg.type === 'price_update') {
            // Format: { bitcoin: { price, change_24h, market_cap, volume_24h } }
            setData({ type: msg.type, data: msg.data, timestamp: msg.timestamp });
            setError(null);
          } else if (msg.type === 'initial_prices') {
            // Backend sends CoinGecko simple-price format: { bitcoin: { usd: 67000 } }
            // Convert to StreamData format
            const converted: StreamData['data'] = {};
            for (const [id, val] of Object.entries(
              msg.data as Record<string, { usd: number }>
            )) {
              converted[id] = {
                price: val.usd ?? 0,
                change_24h: 0,
                market_cap: 0,
                volume_24h: 0,
              };
            }
            setData({ type: 'initial_prices', data: converted, timestamp: msg.timestamp });
          } else if (msg.type === 'error') {
            setError(msg.message);
          }
          // 'connection_established' is informational only
        } catch (e) {
          console.error('[WS] Failed to parse message:', e);
        }
      };

      ws.onerror = () => {
        console.warn('[WS] Connection error');
        setIsConnected(false);
      };

      ws.onclose = (event) => {
        console.log(`[WS] Disconnected (code ${event.code})`);
        setIsConnected(false);
        wsRef.current = null;

        if (!enabledRef.current) return;

        const attempts = reconnectCountRef.current + 1;
        reconnectCountRef.current = attempts;
        setReconnectCount(attempts);

        if (attempts <= maxReconnectAttempts) {
          console.log(
            `[WS] Reconnect attempt ${attempts}/${maxReconnectAttempts} in ${reconnectInterval}ms`
          );
          reconnectTimeoutRef.current = setTimeout(
            () => connectRef.current?.(),
            reconnectInterval
          );
        } else {
          console.log('[WS] Max reconnect attempts reached — falling back to polling');
          startFallbackPolling();
        }
      };
    } catch (e) {
      console.error('[WS] Failed to create WebSocket:', e);
      startFallbackPolling();
    }
  }, [wsUrl, maxReconnectAttempts, reconnectInterval, startFallbackPolling]);

  // Keep forward-ref current so onclose can reconnect
  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  // Start on mount / when enabled changes
  useEffect(() => {
    if (!enabled) {
      return cleanup;
    }
    connect();
    return cleanup;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled]);

  const reconnect = useCallback(() => {
    cleanup();
    reconnectCountRef.current = 0;
    setReconnectCount(0);
    isFallbackRef.current = false;
    setIsFallbackMode(false);
    setIsConnected(false);
    connect();
  }, [cleanup, connect]);

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
