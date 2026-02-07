"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import {
  fetchTradeAnalytics,
  fetchTradeAnalyticsSummary,
  fetchMissedOpportunities,
  fetchARSHeatmap,
} from "@/lib/api";

export type DiagnosticsTab = "analytics" | "missed" | "ars";

export function useDiagnosticsTab() {
  const [tab, setTab] = useState<DiagnosticsTab>("analytics");
  return { tab, setTab };
}

export function useTradeAnalytics(strategyId?: string) {
  return useQuery({
    queryKey: ["diagnostics", "trade-analytics", strategyId],
    queryFn: () => fetchTradeAnalytics(strategyId),
    staleTime: 30_000,
  });
}

export function useTradeAnalyticsSummary() {
  return useQuery({
    queryKey: ["diagnostics", "trade-analytics-summary"],
    queryFn: () => fetchTradeAnalyticsSummary(),
    staleTime: 30_000,
  });
}

export function useMissedOpportunities(
  date?: string,
  startDate?: string,
  endDate?: string,
  strategyId?: string,
) {
  return useQuery({
    queryKey: ["diagnostics", "missed", date, startDate, endDate, strategyId],
    queryFn: () => fetchMissedOpportunities(date, startDate, endDate, strategyId),
    enabled: !!(date || (startDate && endDate)),
    staleTime: 30_000,
  });
}

export function useARSSurface(startDate?: string, endDate?: string) {
  return useQuery({
    queryKey: ["diagnostics", "ars-surface", startDate, endDate],
    queryFn: () => fetchARSHeatmap(startDate, endDate),
    staleTime: 60_000,
  });
}
