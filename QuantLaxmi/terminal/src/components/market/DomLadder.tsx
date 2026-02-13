import { useEffect, useRef, useCallback, useState } from "react";
import { useAtomValue } from "jotai";
import {
  selectedSymbolAtom,
  selectedOrderbookAtom,
  selectedTickAtom,
  type OrderbookData,
} from "@/stores/market";
import { useTauriCommand } from "@/hooks/useTauriCommand";
import { useTauriStream } from "@/hooks/useTauriStream";
import { useAtom } from "jotai";
import { orderbookAtom } from "@/stores/market";
import { themeAtom } from "@/stores/workspace";
import { getChartColors, withAlpha } from "@/lib/chartTheme";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ROW_HEIGHT = 22;
const PRICE_COL_WIDTH = 90;
const SIZE_COL_WIDTH = 70;
const TICK_SIZE = 0.05; // Minimum price increment for NSE

// ---------------------------------------------------------------------------
// Colors (theme-aware via CSS custom properties)
// ---------------------------------------------------------------------------

function getThemeAwareColors() {
  const c = getChartColors();
  const s = getComputedStyle(document.documentElement);
  const panelRaw = s.getPropertyValue("--terminal-panel").trim();
  const textRaw = s.getPropertyValue("--terminal-text").trim();
  return {
    bg: c.bg,
    surface: c.surface,
    panel: panelRaw ? `rgb(${panelRaw.split(/\s+/).join(", ")})` : c.surface,
    border: c.border,
    muted: c.text,
    accent: c.accent,
    profit: c.profit,
    profitDim: withAlpha(c.profit, "1f"),
    loss: c.loss,
    lossDim: withAlpha(c.loss, "1f"),
    currentRow: withAlpha(c.accent, "26"),
    text: textRaw ? `rgb(${textRaw.split(/\s+/).join(", ")})` : c.text,
    font: "11px 'JetBrains Mono', 'Fira Code', monospace",
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function roundToTick(price: number): number {
  return Math.round(price / TICK_SIZE) * TICK_SIZE;
}

function buildSizeMap(levels: [number, number][]): Map<number, number> {
  const map = new Map<number, number>();
  for (const [price, size] of levels) {
    map.set(roundToTick(price), size);
  }
  return map;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function DomLadder() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const rafRef = useRef<number>(0);

  const symbol = useAtomValue(selectedSymbolAtom);
  const book = useAtomValue(selectedOrderbookAtom);
  const tick = useAtomValue(selectedTickAtom);
  const [, setOrderbook] = useAtom(orderbookAtom);
  const theme = useAtomValue(themeAtom);

  const [scrollOffset, setScrollOffset] = useState(0);
  const { execute: placeOrder } = useTauriCommand<{ orderId: string }>("place_order");

  // Current price (center of ladder)
  const currentPrice = tick?.ltp ?? book?.midPrice ?? 0;

  // Subscribe to live book updates
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

  // ---- Canvas rendering ----
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let width = 0;
    let height = 0;

    const resize = () => {
      const rect = container.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      width = rect.width;
      height = rect.height;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(container);

    const render = () => {
      const C = getThemeAwareColors();

      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = C.bg;
      ctx.fillRect(0, 0, width, height);

      if (currentPrice <= 0) {
        ctx.fillStyle = C.muted;
        ctx.font = C.font;
        ctx.textAlign = "center";
        ctx.fillText("Waiting for data...", width / 2, height / 2);
        rafRef.current = requestAnimationFrame(render);
        return;
      }

      const bidMap = book ? buildSizeMap(book.bids) : new Map<number, number>();
      const askMap = book ? buildSizeMap(book.asks) : new Map<number, number>();

      // Find max size for bar scaling
      let maxSize = 1;
      for (const s of bidMap.values()) maxSize = Math.max(maxSize, s);
      for (const s of askMap.values()) maxSize = Math.max(maxSize, s);

      const visibleRows = Math.ceil(height / ROW_HEIGHT);
      const centerRow = Math.floor(visibleRows / 2);
      const centerPrice = roundToTick(currentPrice);

      ctx.font = C.font;

      for (let i = 0; i < visibleRows; i++) {
        const rowY = i * ROW_HEIGHT;
        const priceLevel = roundToTick(
          centerPrice + (centerRow - i + scrollOffset) * TICK_SIZE,
        );

        const bidSize = bidMap.get(priceLevel) ?? 0;
        const askSize = askMap.get(priceLevel) ?? 0;
        const isCurrentPrice =
          Math.abs(priceLevel - roundToTick(currentPrice)) < TICK_SIZE * 0.5;

        // Row background
        if (isCurrentPrice) {
          ctx.fillStyle = C.currentRow;
          ctx.fillRect(0, rowY, width, ROW_HEIGHT);
        } else if (i % 2 === 0) {
          ctx.fillStyle = C.surface;
          ctx.fillRect(0, rowY, width, ROW_HEIGHT);
        }

        // Bid size bar (left side)
        if (bidSize > 0) {
          const barWidth = (bidSize / maxSize) * SIZE_COL_WIDTH;
          ctx.fillStyle = C.profitDim;
          ctx.fillRect(SIZE_COL_WIDTH - barWidth, rowY + 1, barWidth, ROW_HEIGHT - 2);

          ctx.fillStyle = C.profit;
          ctx.textAlign = "right";
          ctx.fillText(
            bidSize.toLocaleString("en-IN"),
            SIZE_COL_WIDTH - 4,
            rowY + ROW_HEIGHT - 6,
          );
        }

        // Price column (center)
        const priceX = SIZE_COL_WIDTH + PRICE_COL_WIDTH / 2;
        ctx.textAlign = "center";
        if (isCurrentPrice) {
          ctx.fillStyle = C.accent;
        } else if (priceLevel > currentPrice) {
          ctx.fillStyle = C.loss;
        } else {
          ctx.fillStyle = C.profit;
        }
        ctx.fillText(priceLevel.toFixed(2), priceX, rowY + ROW_HEIGHT - 6);

        // Ask size bar (right side)
        if (askSize > 0) {
          const barStart = SIZE_COL_WIDTH + PRICE_COL_WIDTH;
          const barWidth = (askSize / maxSize) * SIZE_COL_WIDTH;
          ctx.fillStyle = C.lossDim;
          ctx.fillRect(barStart, rowY + 1, barWidth, ROW_HEIGHT - 2);

          ctx.fillStyle = C.loss;
          ctx.textAlign = "left";
          ctx.fillText(
            askSize.toLocaleString("en-IN"),
            barStart + 4,
            rowY + ROW_HEIGHT - 6,
          );
        }

        // Row border
        ctx.strokeStyle = C.border;
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(0, rowY + ROW_HEIGHT);
        ctx.lineTo(width, rowY + ROW_HEIGHT);
        ctx.stroke();
      }

      // Column dividers
      ctx.strokeStyle = C.border;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(SIZE_COL_WIDTH, 0);
      ctx.lineTo(SIZE_COL_WIDTH, height);
      ctx.moveTo(SIZE_COL_WIDTH + PRICE_COL_WIDTH, 0);
      ctx.lineTo(SIZE_COL_WIDTH + PRICE_COL_WIDTH, height);
      ctx.stroke();

      // Column headers
      ctx.fillStyle = C.panel;
      ctx.fillRect(0, 0, width, ROW_HEIGHT);
      ctx.strokeStyle = C.border;
      ctx.beginPath();
      ctx.moveTo(0, ROW_HEIGHT);
      ctx.lineTo(width, ROW_HEIGHT);
      ctx.stroke();

      ctx.fillStyle = C.muted;
      ctx.font = C.font;
      ctx.textAlign = "center";
      ctx.fillText("BID", SIZE_COL_WIDTH / 2, ROW_HEIGHT - 6);
      ctx.fillText("PRICE", SIZE_COL_WIDTH + PRICE_COL_WIDTH / 2, ROW_HEIGHT - 6);
      ctx.fillText(
        "ASK",
        SIZE_COL_WIDTH + PRICE_COL_WIDTH + SIZE_COL_WIDTH / 2,
        ROW_HEIGHT - 6,
      );

      rafRef.current = requestAnimationFrame(render);
    };

    rafRef.current = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(rafRef.current);
      ro.disconnect();
    };
  }, [book, currentPrice, scrollOffset, theme]);

  // ---- Click handler: detect bid/ask column ----
  const handleCanvasClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas || currentPrice <= 0) return;

      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      // Skip header row
      if (y < ROW_HEIGHT) return;

      const visibleRows = Math.ceil(rect.height / ROW_HEIGHT);
      const centerRow = Math.floor(visibleRows / 2);
      const row = Math.floor(y / ROW_HEIGHT);
      const centerPrice = roundToTick(currentPrice);
      const priceLevel = roundToTick(
        centerPrice + (centerRow - row + scrollOffset) * TICK_SIZE,
      );

      if (x < SIZE_COL_WIDTH) {
        // Clicked bid side -> SELL at this price
        placeOrder({
          symbol,
          side: "SELL",
          orderType: "LIMIT",
          price: priceLevel,
          quantity: 1,
        }).catch(() => {});
      } else if (x > SIZE_COL_WIDTH + PRICE_COL_WIDTH) {
        // Clicked ask side -> BUY at this price
        placeOrder({
          symbol,
          side: "BUY",
          orderType: "LIMIT",
          price: priceLevel,
          quantity: 1,
        }).catch(() => {});
      }
    },
    [currentPrice, scrollOffset, symbol, placeOrder],
  );

  // ---- Keyboard: Up/Down to scroll ----
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "ArrowUp") {
      e.preventDefault();
      setScrollOffset((prev) => prev + 1);
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      setScrollOffset((prev) => prev - 1);
    } else if (e.key === "Home") {
      e.preventDefault();
      setScrollOffset(0);
    }
  }, []);

  // ---- Wheel to scroll ----
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -1 : 1;
    setScrollOffset((prev) => prev + delta);
  }, []);

  return (
    <div className="flex flex-col h-full w-full bg-terminal-bg">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-terminal-border">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono font-semibold text-terminal-accent">
            DOM Ladder
          </span>
          <span className="text-2xs text-terminal-muted">{symbol}</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="text-2xs text-terminal-muted">LTP</span>
          <span className="text-xs font-mono font-semibold text-terminal-text">
            {currentPrice > 0 ? currentPrice.toFixed(2) : "--"}
          </span>
        </div>
      </div>

      {/* Canvas container */}
      <div
        ref={containerRef}
        className="flex-1 min-h-0 focus:outline-none cursor-crosshair"
        tabIndex={0}
        onKeyDown={handleKeyDown}
        onWheel={handleWheel}
      >
        <canvas
          ref={canvasRef}
          className="block w-full h-full"
          onClick={handleCanvasClick}
        />
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between px-3 py-1 border-t border-terminal-border text-2xs text-terminal-muted font-mono">
        <span>Click BID to Sell, ASK to Buy</span>
        <span>Arrows: Scroll | Home: Center</span>
      </div>
    </div>
  );
}
