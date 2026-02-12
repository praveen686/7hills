import { useCallback, useRef, useState } from "react";
import { apiFetch } from "@/lib/api";

interface TauriCommandState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

interface UseTauriCommandReturn<T> extends TauriCommandState<T> {
  /** Execute the command with optional arguments object. */
  execute: (args?: Record<string, unknown>) => Promise<T>;
  /** Reset state back to initial values. */
  reset: () => void;
}

// ---------------------------------------------------------------------------
// Command → FastAPI endpoint mapping
// ---------------------------------------------------------------------------

function resolveEndpoint(
  command: string,
  args?: Record<string, unknown>,
): { url: string; method: string; body?: string } {
  switch (command) {
    case "get_portfolio":
    case "get_positions":
      return { url: "/api/portfolio", method: "GET" };
    case "get_strategies":
      return { url: "/api/strategies", method: "GET" };
    case "get_strategy_detail":
      return { url: `/api/strategies/${args?.id ?? args?.strategy_id ?? ""}`, method: "GET" };
    case "get_risk_state":
      return { url: "/api/risk", method: "GET" };
    case "get_orderbook":
      return { url: `/api/market/orderbook/${args?.symbol ?? "NIFTY"}`, method: "GET" };
    case "search_symbols": {
      const q = args?.query || args?.q || "";
      return { url: `/api/market/symbols?q=${encodeURIComponent(String(q))}`, method: "GET" };
    }
    case "place_order":
      return {
        url: "/api/trading/order",
        method: "POST",
        body: JSON.stringify(args),
      };
    case "cancel_order":
      return {
        url: `/api/trading/order/${args?.orderId ?? args?.order_id ?? ""}`,
        method: "DELETE",
      };
    case "estimate_margin":
      return {
        url: "/api/trading/margin",
        method: "POST",
        body: JSON.stringify(args),
      };
    case "get_status":
      return { url: "/health", method: "GET" };
    case "get_bars": {
      const params = new URLSearchParams();
      if (args?.interval) params.set("interval", String(args.interval));
      if (args?.start_date) params.set("start_date", String(args.start_date));
      if (args?.end_date) params.set("end_date", String(args.end_date));
      const qs = params.toString();
      return {
        url: `/api/market/bars/${args?.symbol ?? "NIFTY"}${qs ? `?${qs}` : ""}`,
        method: "GET",
      };
    }
    default:
      return { url: `/api/${command}`, method: "GET" };
  }
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

/**
 * Drop-in replacement for the Tauri invoke() hook.
 * Maps command names to FastAPI REST endpoints.
 *
 * @param command - The command name (same snake_case names used with Tauri)
 * @returns Object with { data, loading, error, execute, reset }
 */
export function useTauriCommand<T>(command: string): UseTauriCommandReturn<T> {
  const [state, setState] = useState<TauriCommandState<T>>({
    data: null,
    loading: false,
    error: null,
  });

  const callIdRef = useRef(0);

  const execute = useCallback(
    async (args?: Record<string, unknown>): Promise<T> => {
      const callId = ++callIdRef.current;
      setState({ data: null, loading: true, error: null });

      try {
        const { url, method, body } = resolveEndpoint(command, args);
        const result = await apiFetch<T>(url, {
          method,
          ...(body ? { body } : {}),
        });

        if (callId === callIdRef.current) {
          setState({ data: result, loading: false, error: null });
        }
        return result;
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err);
        if (callId === callIdRef.current) {
          setState({ data: null, loading: false, error: message });
        }
        throw err;
      }
    },
    [command],
  );

  const reset = useCallback(() => {
    callIdRef.current++;
    setState({ data: null, loading: false, error: null });
  }, []);

  return { ...state, execute, reset };
}
