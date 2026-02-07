"use client";

import { useState, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchTradeDecisions } from "@/lib/api";
import type { TradeDecisionChain } from "@/lib/types";

// ============================================================
// Why Panel State Hook
// ============================================================

export interface WhyPanelTarget {
  strategy_id: string;
  symbol: string;
  date: string;
}

export interface UseWhyPanelReturn {
  /** Whether the panel is open */
  isOpen: boolean;
  /** Current target (what trade are we explaining?) */
  target: WhyPanelTarget | null;
  /** Full decision chain from WAL */
  chain: TradeDecisionChain | undefined;
  /** Loading state */
  isLoading: boolean;
  /** Error state */
  error: Error | null;
  /** Open the panel for a specific trade */
  open: (target: WhyPanelTarget) => void;
  /** Close the panel */
  close: () => void;
}

export function useWhyPanel(): UseWhyPanelReturn {
  const [isOpen, setIsOpen] = useState(false);
  const [target, setTarget] = useState<WhyPanelTarget | null>(null);

  const { data: chain, isLoading, error } = useQuery<TradeDecisionChain>({
    queryKey: ["why-panel", target?.strategy_id, target?.symbol, target?.date],
    queryFn: () =>
      fetchTradeDecisions(target!.strategy_id, target!.symbol, target!.date),
    enabled: !!target && isOpen,
    staleTime: 30000,
    retry: 1,
  });

  const open = useCallback((t: WhyPanelTarget) => {
    setTarget(t);
    setIsOpen(true);
  }, []);

  const close = useCallback(() => {
    setIsOpen(false);
  }, []);

  return {
    isOpen,
    target,
    chain,
    isLoading,
    error: error as Error | null,
    open,
    close,
  };
}
