import { atom } from "jotai";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Position {
  symbol: string;
  side: "LONG" | "SHORT";
  quantity: number;
  avgPrice: number;
  ltp: number;
  pnl: number;
  pnlPct: number;
  strategyId: string;
}

export interface PortfolioState {
  equity: number;
  peakEquity: number;
  cash: number;
  drawdownPct: number;
  totalExposure: number;
  returnPct: number;
  dayPnl: number;
  positions: Position[];
}

// ---------------------------------------------------------------------------
// Atoms
// ---------------------------------------------------------------------------

/** Full portfolio state — updated by useTauriStream("portfolio_update") */
export const portfolioAtom = atom<PortfolioState>({
  equity: 0,
  peakEquity: 0,
  cash: 0,
  drawdownPct: 0,
  totalExposure: 0,
  returnPct: 0,
  dayPnl: 0,
  positions: [],
});

// ---------------------------------------------------------------------------
// Derived atoms
// ---------------------------------------------------------------------------

/** All open positions */
export const positionsAtom = atom<Position[]>((get) => get(portfolioAtom).positions);

/** Current equity */
export const equityAtom = atom<number>((get) => get(portfolioAtom).equity);

/** Day P&L */
export const dayPnlAtom = atom<number>((get) => get(portfolioAtom).dayPnl);

/** Drawdown from peak as a percentage (negative value) */
export const drawdownAtom = atom<number>((get) => get(portfolioAtom).drawdownPct);

/** Total exposure across all positions */
export const totalExposureAtom = atom<number>((get) => get(portfolioAtom).totalExposure);

/** Cash available */
export const cashAtom = atom<number>((get) => get(portfolioAtom).cash);

/** Return percentage since inception */
export const returnPctAtom = atom<number>((get) => get(portfolioAtom).returnPct);

/** Peak equity */
export const peakEquityAtom = atom<number>((get) => get(portfolioAtom).peakEquity);

/** Number of open positions */
export const positionCountAtom = atom<number>((get) => get(portfolioAtom).positions.length);

/** Aggregate unrealized P&L across all positions */
export const unrealizedPnlAtom = atom<number>((get) =>
  get(portfolioAtom).positions.reduce((sum, p) => sum + p.pnl, 0),
);
