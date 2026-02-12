import { useEffect, useRef } from "react";
import { useSetAtom } from "jotai";
import { connectionAtom, type ConnectionState } from "@/stores/workspace";
import { wsManager } from "@/lib/ws";
import { apiFetch } from "@/lib/api";

/**
 * Polls FastAPI /health every 5s, tracks WebSocket connection states,
 * and updates connectionAtom in the workspace store.
 */
export function useConnection(): void {
  const setConnection = useSetAtom(connectionAtom);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    async function ping() {
      const state: ConnectionState = {
        zerodha: "disconnected",
        binance: "disconnected",
        fastapi: "disconnected",
        latencyMs: 0,
      };

      // Ping FastAPI
      const start = performance.now();
      try {
        await apiFetch<{ status: string }>("/health");
        state.fastapi = "connected";
        state.latencyMs = Math.round(performance.now() - start);
      } catch {
        state.fastapi = "disconnected";
        state.latencyMs = 0;
      }

      // Check WebSocket channels
      if (wsManager.isConnected("/ws/ticks")) {
        state.zerodha = "connected";
        state.binance = "connected";
      } else if (wsManager.anyConnected) {
        state.zerodha = "connecting";
        state.binance = "connecting";
      }

      setConnection(state);
    }

    // Initial ping
    ping();

    // Poll every 5 seconds
    timerRef.current = setInterval(ping, 5000);

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [setConnection]);
}
