import { useEffect, useRef, useCallback } from "react";
import { useAtom, useAtomValue } from "jotai";

import {
  watchlistAtom,
  ticksAtom,
  selectedSymbolAtom,
  type TickData,
} from "@/stores/market";
import { useTauriStream } from "@/hooks/useTauriStream";

// ---------------------------------------------------------------------------
// Mini sparkline (pure SVG, no library)
// ---------------------------------------------------------------------------

function MiniSparkline({ data, color }: { data: number[]; color: string }) {
  if (data.length < 2) return null;

  const w = 48;
  const h = 16;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  const points = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * w;
      const y = h - ((v - min) / range) * h;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg width={w} height={h} className="flex-shrink-0">
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="1.2"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Single ticker chip
// ---------------------------------------------------------------------------

function TickerChip({
  symbol,
  tick,
  isActive,
  onSelect,
}: {
  symbol: string;
  tick: TickData | undefined;
  isActive: boolean;
  onSelect: () => void;
}) {
  // Keep a small price history for sparkline (last 20 ticks)
  const historyRef = useRef<number[]>([]);
  if (tick) {
    const hist = historyRef.current;
    hist.push(tick.ltp);
    if (hist.length > 20) hist.shift();
  }

  const ltp = tick?.ltp ?? 0;
  const changePct = tick?.changePct ?? 0;
  const isUp = changePct >= 0;
  const priceColor = isUp ? "text-terminal-profit" : "text-terminal-loss";
  const sparkColor = isUp ? "#00d4aa" : "#ff4d6a";

  return (
    <button
      onClick={onSelect}
      className={`
        flex items-center gap-2.5 px-3 py-1.5 rounded-md flex-shrink-0
        transition-colors cursor-pointer select-none
        ${isActive
          ? "bg-terminal-surface border border-terminal-accent/40"
          : "bg-transparent border border-transparent hover:bg-terminal-surface/60 hover:border-terminal-border"
        }
      `}
    >
      {/* Symbol name */}
      <span className="text-xs font-semibold text-gray-300 whitespace-nowrap">
        {symbol}
      </span>

      {/* LTP */}
      <span className={`text-xs font-mono tabular-nums ${priceColor}`}>
        {ltp > 0 ? ltp.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "--"}
      </span>

      {/* Change % */}
      <span className={`text-2xs font-mono tabular-nums ${priceColor}`}>
        {ltp > 0
          ? `${isUp ? "+" : ""}${changePct.toFixed(2)}%`
          : ""}
      </span>

      {/* Sparkline */}
      <MiniSparkline data={historyRef.current} color={sparkColor} />
    </button>
  );
}

// ---------------------------------------------------------------------------
// TickerBar
// ---------------------------------------------------------------------------

export function TickerBar() {
  const watchlist = useAtomValue(watchlistAtom);
  const [ticks, setTicks] = useAtom(ticksAtom);
  const [selectedSymbol, setSelectedSymbol] = useAtom(selectedSymbolAtom);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Listen for tick events via WebSocket
  const handleTick = useCallback(
    (data: TickData) => {
      setTicks((prev) => {
        const next = new Map(prev);
        next.set(data.symbol, data);
        return next;
      });
    },
    [setTicks],
  );

  useTauriStream<TickData>("tick", handleTick);

  const handleSelect = useCallback(
    (symbol: string) => setSelectedSymbol(symbol),
    [setSelectedSymbol],
  );

  // Horizontal scroll via mouse wheel
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const handleWheel = (e: WheelEvent) => {
      if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
        e.preventDefault();
        el.scrollLeft += e.deltaY;
      }
    };
    el.addEventListener("wheel", handleWheel, { passive: false });
    return () => el.removeEventListener("wheel", handleWheel);
  }, []);

  return (
    <div className="h-10 flex items-center bg-terminal-panel border-b border-terminal-border flex-shrink-0">
      <div
        ref={scrollRef}
        className="flex items-center gap-1 px-2 overflow-x-auto scrollbar-none"
        style={{ scrollbarWidth: "none" }}
      >
        {watchlist.map((symbol) => (
          <TickerChip
            key={symbol}
            symbol={symbol}
            tick={ticks.get(symbol)}
            isActive={symbol === selectedSymbol}
            onSelect={() => handleSelect(symbol)}
          />
        ))}
      </div>
    </div>
  );
}
