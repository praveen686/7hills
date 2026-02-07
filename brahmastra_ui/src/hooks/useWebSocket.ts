"use client";

import { useEffect, useRef, useCallback, useState } from "react";
import type { WSMessage } from "@/lib/types";

// ============================================================
// Generic WebSocket Hook with Auto-Reconnection
// ============================================================

export interface UseWebSocketOptions {
  /** WebSocket URL */
  url: string;
  /** Auto-connect on mount (default: true) */
  autoConnect?: boolean;
  /** Reconnect on disconnect (default: true) */
  reconnect?: boolean;
  /** Reconnect interval in ms (default: 3000) */
  reconnectInterval?: number;
  /** Max reconnect attempts (default: 10) */
  maxReconnectAttempts?: number;
  /** Callback when message received */
  onMessage?: (message: WSMessage) => void;
  /** Callback on open */
  onOpen?: () => void;
  /** Callback on close */
  onClose?: () => void;
  /** Callback on error */
  onError?: (error: Event) => void;
}

export interface UseWebSocketReturn {
  /** Current connection status */
  status: "connecting" | "connected" | "disconnected" | "error";
  /** Last received message */
  lastMessage: WSMessage | null;
  /** Send a message */
  send: (data: unknown) => void;
  /** Manually connect */
  connect: () => void;
  /** Manually disconnect */
  disconnect: () => void;
  /** Number of reconnect attempts */
  reconnectCount: number;
}

export function useWebSocket(options: UseWebSocketOptions): UseWebSocketReturn {
  const {
    url,
    autoConnect = true,
    reconnect = true,
    reconnectInterval = 3000,
    maxReconnectAttempts = 10,
    onMessage,
    onOpen,
    onClose,
    onError,
  } = options;

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout>>();
  const reconnectCountRef = useRef(0);
  const mountedRef = useRef(true);

  const [status, setStatus] = useState<UseWebSocketReturn["status"]>("disconnected");
  const [lastMessage, setLastMessage] = useState<WSMessage | null>(null);
  const [reconnectCount, setReconnectCount] = useState(0);

  // Stable refs for callbacks
  const onMessageRef = useRef(onMessage);
  const onOpenRef = useRef(onOpen);
  const onCloseRef = useRef(onClose);
  const onErrorRef = useRef(onError);

  useEffect(() => {
    onMessageRef.current = onMessage;
    onOpenRef.current = onOpen;
    onCloseRef.current = onClose;
    onErrorRef.current = onError;
  }, [onMessage, onOpen, onClose, onError]);

  const disconnect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
    }
    if (wsRef.current) {
      wsRef.current.onopen = null;
      wsRef.current.onmessage = null;
      wsRef.current.onclose = null;
      wsRef.current.onerror = null;
      wsRef.current.close();
      wsRef.current = null;
    }
    if (mountedRef.current) {
      setStatus("disconnected");
    }
  }, []);

  const connect = useCallback(() => {
    disconnect();

    if (!mountedRef.current) return;

    setStatus("connecting");

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) return;
        setStatus("connected");
        reconnectCountRef.current = 0;
        setReconnectCount(0);
        onOpenRef.current?.();
      };

      ws.onmessage = (event: MessageEvent) => {
        if (!mountedRef.current) return;
        try {
          const message = JSON.parse(event.data) as WSMessage;
          setLastMessage(message);
          onMessageRef.current?.(message);
        } catch {
          // Non-JSON message, wrap it
          const wrapped: WSMessage = {
            type: "raw",
            data: event.data,
            timestamp: new Date().toISOString(),
          };
          setLastMessage(wrapped);
          onMessageRef.current?.(wrapped);
        }
      };

      ws.onclose = () => {
        if (!mountedRef.current) return;
        setStatus("disconnected");
        onCloseRef.current?.();

        // Auto-reconnect
        if (
          reconnect &&
          reconnectCountRef.current < maxReconnectAttempts &&
          mountedRef.current
        ) {
          reconnectCountRef.current += 1;
          setReconnectCount(reconnectCountRef.current);
          reconnectTimerRef.current = setTimeout(() => {
            if (mountedRef.current) {
              connect();
            }
          }, reconnectInterval);
        }
      };

      ws.onerror = (error) => {
        if (!mountedRef.current) return;
        setStatus("error");
        onErrorRef.current?.(error);
      };
    } catch {
      if (mountedRef.current) {
        setStatus("error");
      }
    }
  }, [url, reconnect, reconnectInterval, maxReconnectAttempts, disconnect]);

  const send = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  // Auto-connect on mount
  useEffect(() => {
    mountedRef.current = true;
    if (autoConnect) {
      connect();
    }
    return () => {
      mountedRef.current = false;
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  return {
    status,
    lastMessage,
    send,
    connect,
    disconnect,
    reconnectCount,
  };
}
