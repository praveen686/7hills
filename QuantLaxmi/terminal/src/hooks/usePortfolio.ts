import { useCallback } from "react";
import { useAtom, useAtomValue } from "jotai";
import {
  portfolioAtom,
  positionsAtom,
  equityAtom,
  dayPnlAtom,
  drawdownAtom,
  type PortfolioState,
  type Position,
} from "@/stores/portfolio";
import { useTauriStream } from "@/hooks/useTauriStream";

interface UsePortfolioReturn {
  portfolio: PortfolioState;
  positions: Position[];
  equity: number;
  dayPnl: number;
  drawdown: number;
}

/**
 * Subscribe to the portfolio stream from the Tauri backend and expose
 * derived portfolio state.
 *
 * The backend emits "portfolio_update" events with a full PortfolioState payload.
 */
export function usePortfolio(): UsePortfolioReturn {
  const [portfolio, setPortfolio] = useAtom(portfolioAtom);
  const positions = useAtomValue(positionsAtom);
  const equity = useAtomValue(equityAtom);
  const dayPnl = useAtomValue(dayPnlAtom);
  const drawdown = useAtomValue(drawdownAtom);

  const handleUpdate = useCallback(
    (data: PortfolioState) => {
      setPortfolio(data);
    },
    [setPortfolio],
  );

  useTauriStream<PortfolioState>("portfolio_update", handleUpdate);

  return { portfolio, positions, equity, dayPnl, drawdown };
}
