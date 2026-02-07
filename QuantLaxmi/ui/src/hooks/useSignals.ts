"use client";

import { useState, useCallback } from "react";
import { useWebSocket } from "./useWebSocket";
import { getWSUrl, WS_ENDPOINTS } from "@/lib/api";
import type { Signal, WSMessage } from "@/lib/types";

// ============================================================
// Real-time Signals WebSocket Hook
// ============================================================

export interface UseSignalsReturn {
  /** Live signals buffer (most recent first) */
  signals: Signal[];
  /** WebSocket connection status */
  status: "connecting" | "connected" | "disconnected" | "error";
  /** Clear signal buffer */
  clearSignals: () => void;
  /** Reconnect attempts */
  reconnectCount: number;
}

const MAX_BUFFER_SIZE = 200;

export function useSignals(): UseSignalsReturn {
  const [signals, setSignals] = useState<Signal[]>([]);

  const handleMessage = useCallback((message: WSMessage) => {
    if (message.type === "signal" && message.data) {
      const signal = message.data as Signal;
      setSignals((prev) => {
        const updated = [signal, ...prev];
        // Cap buffer size to prevent memory growth
        if (updated.length > MAX_BUFFER_SIZE) {
          return updated.slice(0, MAX_BUFFER_SIZE);
        }
        return updated;
      });
    }

    if (message.type === "signals_batch" && Array.isArray(message.data)) {
      const batch = message.data as Signal[];
      setSignals((prev) => {
        const updated = [...batch.reverse(), ...prev];
        if (updated.length > MAX_BUFFER_SIZE) {
          return updated.slice(0, MAX_BUFFER_SIZE);
        }
        return updated;
      });
    }
  }, []);

  const { status, reconnectCount } = useWebSocket({
    url: getWSUrl(WS_ENDPOINTS.signals),
    onMessage: handleMessage,
    reconnect: true,
    reconnectInterval: 3000,
    maxReconnectAttempts: 20,
  });

  const clearSignals = useCallback(() => {
    setSignals([]);
  }, []);

  return {
    signals,
    status,
    clearSignals,
    reconnectCount,
  };
}
