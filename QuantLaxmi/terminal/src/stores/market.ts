import { atom } from "jotai";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface TickData {
  symbol: string;
  ltp: number;
  change: number;
  changePct: number;
  volume: number;
  timestamp: string;
}

export interface OrderbookData {
  bids: [number, number][];
  asks: [number, number][];
  spread: number;
  midPrice: number;
}

export interface BarData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// ---------------------------------------------------------------------------
// Atoms
// ---------------------------------------------------------------------------

/** Currently selected symbol (drives chart, orderbook, order entry, etc.) */
export const selectedSymbolAtom = atom<string>("NIFTY");

/** Real-time tick data keyed by symbol */
export const ticksAtom = atom<Map<string, TickData>>(new Map());

/** Live orderbook snapshots keyed by symbol */
export const orderbookAtom = atom<Map<string, OrderbookData>>(new Map());

/** OHLCV bar history keyed by symbol */
export const barsAtom = atom<Map<string, BarData[]>>(new Map());

/** VPIN (Volume-Synchronized Probability of Informed Trading) per symbol */
export const vpinAtom = atom<Map<string, number>>(new Map());

/** User's watchlist of tracked symbols */
export const watchlistAtom = atom<string[]>([
  "NIFTY",
  "BANKNIFTY",
  "FINNIFTY",
  "MIDCPNIFTY",
  "RELIANCE",
  "HDFCBANK",
  "TCS",
  "INFY",
]);

// ---------------------------------------------------------------------------
// Derived atoms
// ---------------------------------------------------------------------------

/** Tick data for the currently selected symbol */
export const selectedTickAtom = atom<TickData | null>((get) => {
  const symbol = get(selectedSymbolAtom);
  return get(ticksAtom).get(symbol) ?? null;
});

/** Orderbook for the currently selected symbol */
export const selectedOrderbookAtom = atom<OrderbookData | null>((get) => {
  const symbol = get(selectedSymbolAtom);
  return get(orderbookAtom).get(symbol) ?? null;
});

/** Bars for the currently selected symbol */
export const selectedBarsAtom = atom<BarData[]>((get) => {
  const symbol = get(selectedSymbolAtom);
  return get(barsAtom).get(symbol) ?? [];
});

/** VPIN for the currently selected symbol */
export const selectedVpinAtom = atom<number | null>((get) => {
  const symbol = get(selectedSymbolAtom);
  return get(vpinAtom).get(symbol) ?? null;
});
