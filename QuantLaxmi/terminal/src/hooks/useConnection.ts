import { useEffect, useRef } from "react";
import { useSetAtom } from "jotai";
import { connectionAtom, type ConnectionState } from "@/stores/workspace";
import { wsManager } from "@/lib/ws";
import { apiFetch } from "@/lib/api";

interface ConnectionStatusResponse {
  zerodha: string;
  binance: string;
  mode: string;
  engine_running: boolean;
  ticks_received: number;
  bars_completed: number;
  signals_emitted: number;
  uptime_seconds: number;
  strategies_registered: number;
  tokens_subscribed: number;
}

/**
 * Polls FastAPI /health and /api/status/connections every 5s,
 * tracks WebSocket connection states, and updates connectionAtom.
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
        mode: "offline",
        engineRunning: false,
        ticksReceived: 0,
        barsCompleted: 0,
        signalsEmitted: 0,
      };

      // Ping FastAPI health
      const start = performance.now();
      try {
        await apiFetch<{ status: string }>("/health");
        state.fastapi = "connected";
        state.latencyMs = Math.round(performance.now() - start);
      } catch {
        state.fastapi = "disconnected";
        state.latencyMs = 0;
      }

      // Fetch detailed connection status from live engine
      if (state.fastapi === "connected") {
        try {
          const status = await apiFetch<ConnectionStatusResponse>(
            "/api/status/connections",
          );
          state.zerodha = status.zerodha as ConnectionState["zerodha"];
          state.binance = status.binance as ConnectionState["binance"];
          state.mode = status.mode;
          state.engineRunning = status.engine_running;
          state.ticksReceived = status.ticks_received;
          state.barsCompleted = status.bars_completed;
          state.signalsEmitted = status.signals_emitted;
        } catch {
          // Fallback: check WebSocket channels
          if (wsManager.isConnected("/ws/ticks")) {
            state.zerodha = "connected";
            state.binance = "connected";
          } else if (wsManager.anyConnected) {
            state.zerodha = "connecting";
            state.binance = "connecting";
          }
        }
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
