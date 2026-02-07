"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchPortfolio, fetchStrategies, fetchTodaySignals, fetchVIX } from "@/lib/api";
import type { PortfolioSummary, StrategySummary, Signal, VIXData } from "@/lib/types";

// ============================================================
// Portfolio & Dashboard React Query Hooks
// ============================================================

/** Fetch portfolio summary, refreshes every 5s */
export function usePortfolio() {
  return useQuery<PortfolioSummary>({
    queryKey: ["portfolio"],
    queryFn: fetchPortfolio,
    refetchInterval: 5000,
    staleTime: 3000,
    retry: 3,
    retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 10000),
  });
}

/** Fetch all strategies, refreshes every 10s */
export function useStrategies() {
  return useQuery<StrategySummary[]>({
    queryKey: ["strategies"],
    queryFn: fetchStrategies,
    refetchInterval: 10000,
    staleTime: 5000,
  });
}

/** Fetch today's signals, refreshes every 5s */
export function useTodaySignals() {
  return useQuery<Signal[]>({
    queryKey: ["signals", "today"],
    queryFn: fetchTodaySignals,
    refetchInterval: 5000,
    staleTime: 3000,
  });
}

/** Fetch VIX data, refreshes every 10s */
export function useVIX() {
  return useQuery<VIXData>({
    queryKey: ["vix"],
    queryFn: fetchVIX,
    refetchInterval: 10000,
    staleTime: 5000,
  });
}
