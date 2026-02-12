import { useEffect, useCallback, useMemo } from "react";
import { useAtomValue } from "jotai";
import {
  selectedSymbolAtom,
  selectedOrderbookAtom,
  orderbookAtom,
  type OrderbookData,
} from "@/stores/market";
import { useTauriStream } from "@/hooks/useTauriStream";
import { useTauriCommand } from "@/hooks/useTauriCommand";
import { cn } from "@/lib/utils";
import { useAtom } from "jotai";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_LEVELS = 15;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface BookLevel {
  price: number;
  size: number;
  cumulative: number;
  pctOfMax: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function buildLevels(
  raw: [number, number][],
  maxCumulative: number,
): BookLevel[] {
  let cumulative = 0;
  return raw.slice(0, MAX_LEVELS).map(([price, size]) => {
    cumulative += size;
    return {
      price,
      size,
      cumulative,
      pctOfMax: maxCumulative > 0 ? (cumulative / maxCumulative) * 100 : 0,
    };
  });
}

function formatPrice(p: number): string {
  return p.toLocaleString("en-IN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function formatSize(s: number): string {
  if (s >= 100_000) return `${(s / 1000).toFixed(0)}K`;
  if (s >= 10_000) return `${(s / 1000).toFixed(1)}K`;
  return s.toLocaleString("en-IN");
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function AskRow({ level }: { level: BookLevel }) {
  return (
    <div className="relative flex items-center h-6 font-mono text-xs tabular-nums group">
      {/* Background bar (right-aligned for asks) */}
      <div
        className="absolute inset-y-0 right-0 bg-terminal-loss/10 transition-all duration-100"
        style={{ width: `${level.pctOfMax}%` }}
      />
      <span className="relative z-10 w-[72px] text-right pr-2 text-terminal-muted">
        {formatSize(level.cumulative)}
      </span>
      <span className="relative z-10 w-[56px] text-right pr-2 text-terminal-muted">
        {formatSize(level.size)}
      </span>
      <span className="relative z-10 flex-1 text-right pr-2 text-terminal-loss">
        {formatPrice(level.price)}
      </span>
    </div>
  );
}

function BidRow({ level }: { level: BookLevel }) {
  return (
    <div className="relative flex items-center h-6 font-mono text-xs tabular-nums group">
      {/* Background bar (left-aligned for bids) */}
      <div
        className="absolute inset-y-0 left-0 bg-terminal-profit/10 transition-all duration-100"
        style={{ width: `${level.pctOfMax}%` }}
      />
      <span className="relative z-10 flex-1 pl-2 text-terminal-profit">
        {formatPrice(level.price)}
      </span>
      <span className="relative z-10 w-[56px] text-left pl-2 text-terminal-muted">
        {formatSize(level.size)}
      </span>
      <span className="relative z-10 w-[72px] text-left pl-2 text-terminal-muted">
        {formatSize(level.cumulative)}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function OrderbookPanel() {
  const symbol = useAtomValue(selectedSymbolAtom);
  const book = useAtomValue(selectedOrderbookAtom);
  const [, setOrderbook] = useAtom(orderbookAtom);

  // ---- Fetch initial snapshot ----
  const { execute: fetchBook } = useTauriCommand<OrderbookData>("get_orderbook");

  useEffect(() => {
    fetchBook({ symbol }).then((data) => {
      setOrderbook((prev) => {
        const next = new Map(prev);
        next.set(symbol, data);
        return next;
      });
    }).catch(() => {
      // Backend may not be available yet; stream will populate
    });
  }, [symbol, fetchBook, setOrderbook]);

  // ---- Subscribe to live updates ----
  const handleBookUpdate = useCallback(
    (data: OrderbookData) => {
      setOrderbook((prev) => {
        const next = new Map(prev);
        next.set(symbol, data);
        return next;
      });
    },
    [symbol, setOrderbook],
  );

  useTauriStream<OrderbookData>(`book:${symbol}`, handleBookUpdate);

  // ---- Build level arrays ----
  const { askLevels, bidLevels, spread, midPrice } = useMemo(() => {
    if (!book) {
      return { askLevels: [], bidLevels: [], spread: 0, midPrice: 0 };
    }

    // Cumulative totals for sizing bars
    const askCum = book.asks.slice(0, MAX_LEVELS).reduce((s, [, sz]) => s + sz, 0);
    const bidCum = book.bids.slice(0, MAX_LEVELS).reduce((s, [, sz]) => s + sz, 0);
    const maxCum = Math.max(askCum, bidCum);

    // Asks are displayed top-down (highest first → reversed)
    const asksTopDown = [...book.asks.slice(0, MAX_LEVELS)].reverse();

    return {
      askLevels: buildLevels(asksTopDown, maxCum),
      bidLevels: buildLevels(book.bids.slice(0, MAX_LEVELS), maxCum),
      spread: book.spread,
      midPrice: book.midPrice,
    };
  }, [book]);

  return (
    <div className="flex flex-col h-full w-full bg-terminal-surface">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-terminal-border">
        <span className="text-xs font-mono font-semibold text-terminal-accent">
          Orderbook
        </span>
        <span className="text-2xs text-terminal-muted">{symbol}</span>
      </div>

      {/* Column headers */}
      <div className="flex items-center px-1 py-1 border-b border-terminal-border text-2xs text-terminal-muted font-mono">
        <span className="w-[72px] text-right pr-2">Total</span>
        <span className="w-[56px] text-right pr-2">Size</span>
        <span className="flex-1 text-right pr-2">Price</span>
      </div>

      {/* Scrollable body */}
      <div className="flex-1 min-h-0 overflow-y-auto scrollbar-thin">
        {/* Asks (highest price at top, lowest near spread) */}
        <div className="flex flex-col justify-end min-h-0">
          {askLevels.map((level) => (
            <AskRow key={`a-${level.price}`} level={level} />
          ))}
        </div>

        {/* Spread row */}
        <div
          className={cn(
            "flex items-center justify-center h-8 border-y border-terminal-border-bright",
            "bg-terminal-panel font-mono text-xs",
          )}
        >
          <span className="text-terminal-muted mr-2">Spread</span>
          <span className="text-terminal-accent font-semibold">
            {formatPrice(spread)}
          </span>
          <span className="text-terminal-muted ml-3 mr-1">Mid</span>
          <span className="text-gray-200 font-semibold">
            {formatPrice(midPrice)}
          </span>
        </div>

        {/* Bids (best bid at top, lowest at bottom) */}
        <div className="flex flex-col min-h-0">
          {/* Column headers for bid side (reversed layout) */}
          <div className="flex items-center px-1 py-0.5 text-2xs text-terminal-muted font-mono border-b border-terminal-border/50">
            <span className="flex-1 pl-2">Price</span>
            <span className="w-[56px] text-left pl-2">Size</span>
            <span className="w-[72px] text-left pl-2">Total</span>
          </div>
          {bidLevels.map((level) => (
            <BidRow key={`b-${level.price}`} level={level} />
          ))}
        </div>
      </div>
    </div>
  );
}
