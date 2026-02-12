import { useEffect, useRef, useCallback } from "react";
import { useAtomValue, useSetAtom } from "jotai";
import {
  createChart,
  type IChartApi,
  type ISeriesApi,
  type CandlestickData,
  type HistogramData,
  type Time,
  CrosshairMode,
  ColorType,
} from "lightweight-charts";
import { selectedSymbolAtom, selectedBarsAtom, barsAtom, type BarData } from "@/stores/market";
import { appModeAtom } from "@/stores/mode";
import { apiFetch } from "@/lib/api";
import { useTauriStream } from "@/hooks/useTauriStream";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Convert a BarData from backend to lightweight-charts CandlestickData. */
function toCandlestick(bar: BarData): CandlestickData<Time> {
  return {
    time: bar.time as Time,
    open: bar.open,
    high: bar.high,
    low: bar.low,
    close: bar.close,
  };
}

/** Convert a BarData to a volume histogram point. */
function toVolume(bar: BarData): HistogramData<Time> {
  return {
    time: bar.time as Time,
    value: bar.volume,
    color: bar.close >= bar.open ? "rgba(0,212,170,0.35)" : "rgba(255,77,106,0.35)",
  };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ChartPanel() {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null);

  const symbol = useAtomValue(selectedSymbolAtom);
  const bars = useAtomValue(selectedBarsAtom);
  const mode = useAtomValue(appModeAtom);
  const setBars = useSetAtom(barsAtom);

  // ---- Create chart on mount ----
  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#08080d" },
        textColor: "#6b6b8a",
        fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "#1e1e2e" },
        horzLines: { color: "#1e1e2e" },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: "rgba(79,142,255,0.4)",
          width: 1,
          style: 2,
          labelBackgroundColor: "#4f8eff",
        },
        horzLine: {
          color: "rgba(79,142,255,0.4)",
          width: 1,
          style: 2,
          labelBackgroundColor: "#4f8eff",
        },
      },
      timeScale: {
        borderColor: "#1e1e2e",
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 5,
        barSpacing: 8,
      },
      rightPriceScale: {
        borderColor: "#1e1e2e",
        scaleMargins: { top: 0.1, bottom: 0.25 },
      },
      handleScroll: { vertTouchDrag: false },
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: "#00d4aa",
      downColor: "#ff4d6a",
      borderUpColor: "#00d4aa",
      borderDownColor: "#ff4d6a",
      wickUpColor: "#00d4aa",
      wickDownColor: "#ff4d6a",
    });

    const volumeSeries = chart.addHistogramSeries({
      priceFormat: { type: "volume" },
      priceScaleId: "volume",
    });

    chart.priceScale("volume").applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
    });

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;
    volumeSeriesRef.current = volumeSeries;

    // ---- Responsive resize ----
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        chart.applyOptions({ width, height });
      }
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
      volumeSeriesRef.current = null;
    };
  }, []);

  // ---- Fetch historical bars from API when symbol changes ----
  useEffect(() => {
    let cancelled = false;

    async function fetchBars() {
      try {
        const data = await apiFetch<BarData[]>(
          `/api/market/bars/${symbol}?interval=daily`,
        );
        if (!cancelled && data.length > 0) {
          setBars((prev) => {
            const next = new Map(prev);
            next.set(symbol, data);
            return next;
          });
        }
      } catch {
        // API not available — chart stays empty until data arrives
      }
    }

    fetchBars();
    return () => {
      cancelled = true;
    };
  }, [symbol, setBars]);

  // ---- Load historical bars when bars atom changes ----
  useEffect(() => {
    if (!candleSeriesRef.current || !volumeSeriesRef.current) return;
    if (bars.length === 0) return;

    const candles = bars.map(toCandlestick);
    const volumes = bars.map(toVolume);

    candleSeriesRef.current.setData(candles);
    volumeSeriesRef.current.setData(volumes);

    chartRef.current?.timeScale().fitContent();
  }, [bars]);

  // ---- Subscribe to live bar updates ----
  const handleBar = useCallback(
    (bar: BarData) => {
      if (!candleSeriesRef.current || !volumeSeriesRef.current) return;
      candleSeriesRef.current.update(toCandlestick(bar));
      volumeSeriesRef.current.update(toVolume(bar));
    },
    [],
  );

  useTauriStream<BarData>(`bar:${symbol}`, handleBar);

  return (
    <div className="flex flex-col h-full w-full bg-terminal-bg">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-terminal-border">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono font-semibold text-terminal-accent">
            {symbol}
          </span>
          <span className="text-2xs text-terminal-muted">OHLCV</span>
          <span className="text-2xs text-terminal-muted">
            ({mode === "live" ? "LIVE" : "HIST"})
          </span>
        </div>
        <span className="text-2xs text-terminal-muted">IST</span>
      </div>

      {/* Chart container */}
      <div ref={containerRef} className="flex-1 min-h-0" />
    </div>
  );
}
