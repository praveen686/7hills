import { useCallback, useEffect, useRef, useState, useMemo } from "react";
import { useAtomValue } from "jotai";
import { selectedSymbolAtom } from "@/stores/market";
import { useTauriStream } from "@/hooks/useTauriStream";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TapeEntry {
  id: number;
  time: string;
  price: number;
  size: number;
  side: "BUY" | "SELL";
}

interface RawTrade {
  price: number;
  size: number;
  side: "BUY" | "SELL";
  timestamp: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_TRADES = 500;
const ROW_HEIGHT = 20;
const DEFAULT_MIN_SIZE = 0;

let _tradeId = 0;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Format ISO timestamp → HH:MM:SS.mmm */
function formatTradeTime(ts: string): string {
  const d = new Date(ts);
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  const ms = String(d.getMilliseconds()).padStart(3, "0");
  return `${hh}:${mm}:${ss}.${ms}`;
}

function formatPrice(p: number): string {
  return p.toLocaleString("en-IN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function formatSize(s: number): string {
  if (s >= 100_000) return `${(s / 1000).toFixed(0)}K`;
  return s.toLocaleString("en-IN");
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function TapePanel() {
  const symbol = useAtomValue(selectedSymbolAtom);
  const [trades, setTrades] = useState<TapeEntry[]>([]);
  const [minSize, setMinSize] = useState(DEFAULT_MIN_SIZE);
  const [autoScroll, setAutoScroll] = useState(true);

  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const lastPriceRef = useRef<number>(0);

  // ---- Reset on symbol change ----
  useEffect(() => {
    setTrades([]);
    lastPriceRef.current = 0;
  }, [symbol]);

  // ---- Subscribe to live trades ----
  const handleTrade = useCallback((raw: RawTrade) => {
    const entry: TapeEntry = {
      id: ++_tradeId,
      time: formatTradeTime(raw.timestamp),
      price: raw.price,
      size: raw.size,
      side: raw.side,
    };

    lastPriceRef.current = raw.price;

    setTrades((prev) => {
      const next = [...prev, entry];
      // Trim to MAX_TRADES
      if (next.length > MAX_TRADES) {
        return next.slice(next.length - MAX_TRADES);
      }
      return next;
    });
  }, []);

  useTauriStream<RawTrade>(`trade:${symbol}`, handleTrade);

  // ---- Auto-scroll to bottom ----
  useEffect(() => {
    if (autoScroll && scrollContainerRef.current) {
      scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
    }
  }, [trades, autoScroll]);

  // ---- Detect manual scroll ----
  const handleScroll = useCallback(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < ROW_HEIGHT * 2;
    setAutoScroll(atBottom);
  }, []);

  // ---- Filtered trades ----
  const filteredTrades = useMemo(() => {
    if (minSize <= 0) return trades;
    return trades.filter((t) => t.size >= minSize);
  }, [trades, minSize]);

  // ---- Determine large trade threshold (90th percentile) ----
  const largeThreshold = useMemo(() => {
    if (trades.length < 10) return Infinity;
    const sizes = trades.map((t) => t.size).sort((a, b) => a - b);
    return sizes[Math.floor(sizes.length * 0.9)] ?? Infinity;
  }, [trades]);

  // ---- Virtual scrolling ----
  const containerHeight = filteredTrades.length * ROW_HEIGHT;

  return (
    <div className="flex flex-col h-full w-full bg-terminal-surface">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-terminal-border">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono font-semibold text-terminal-accent">
            Time & Sales
          </span>
          <span className="text-2xs text-terminal-muted">{symbol}</span>
        </div>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1 text-2xs text-terminal-muted">
            Min:
            <input
              type="number"
              value={minSize || ""}
              onChange={(e) => setMinSize(Math.max(0, Number(e.target.value) || 0))}
              placeholder="0"
              className={cn(
                "w-14 px-1 py-0.5 rounded text-2xs font-mono",
                "bg-terminal-bg border border-terminal-border text-terminal-text-secondary",
                "focus:outline-none focus:border-terminal-accent",
              )}
            />
          </label>
          <span className="text-2xs text-terminal-muted">
            {filteredTrades.length}
          </span>
        </div>
      </div>

      {/* Column headers */}
      <div className="flex items-center px-2 py-1 border-b border-terminal-border text-2xs text-terminal-muted font-mono">
        <span className="w-[88px]">Time</span>
        <span className="flex-1 text-right">Price</span>
        <span className="w-[64px] text-right">Size</span>
        <span className="w-[10px]" />
      </div>

      {/* Trade list with virtual scrolling */}
      <div
        ref={scrollContainerRef}
        className="flex-1 min-h-0 overflow-y-auto"
        onScroll={handleScroll}
      >
        <div style={{ height: containerHeight, position: "relative" }}>
          {filteredTrades.map((trade, i) => {
            const isBuy = trade.side === "BUY";
            const isLarge = trade.size >= largeThreshold;

            return (
              <div
                key={trade.id}
                className={cn(
                  "absolute left-0 right-0 flex items-center px-2 font-mono text-2xs tabular-nums",
                  isLarge && "bg-terminal-warning/8 font-semibold",
                )}
                style={{
                  top: i * ROW_HEIGHT,
                  height: ROW_HEIGHT,
                }}
              >
                <span className="w-[88px] text-terminal-muted">{trade.time}</span>
                <span
                  className={cn(
                    "flex-1 text-right",
                    isBuy ? "text-terminal-profit" : "text-terminal-loss",
                  )}
                >
                  {formatPrice(trade.price)}
                </span>
                <span
                  className={cn(
                    "w-[64px] text-right",
                    isLarge ? "text-terminal-warning" : "text-terminal-text-secondary",
                  )}
                >
                  {formatSize(trade.size)}
                </span>
                <span className="w-[10px] flex justify-center">
                  <span
                    className={cn(
                      "inline-block w-1.5 h-1.5 rounded-full",
                      isBuy ? "bg-terminal-profit" : "bg-terminal-loss",
                    )}
                  />
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Footer */}
      {!autoScroll && (
        <button
          onClick={() => {
            setAutoScroll(true);
            if (scrollContainerRef.current) {
              scrollContainerRef.current.scrollTop =
                scrollContainerRef.current.scrollHeight;
            }
          }}
          className={cn(
            "absolute bottom-8 left-1/2 -translate-x-1/2 px-3 py-1 rounded",
            "bg-terminal-accent/20 text-terminal-accent text-2xs font-mono",
            "hover:bg-terminal-accent/30 transition-colors",
          )}
        >
          Scroll to latest
        </button>
      )}
    </div>
  );
}
