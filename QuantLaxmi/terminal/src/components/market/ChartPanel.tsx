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
import { getChartColors, withAlpha } from "@/lib/chartTheme";
import { themeAtom } from "@/stores/workspace";

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
function toVolume(bar: BarData, upColor: string, downColor: string): HistogramData<Time> {
  return {
    time: bar.time as Time,
    value: bar.volume,
    color: bar.close >= bar.open ? upColor : downColor,
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
  const volColorsRef = useRef({ up: "", down: "" });

  const symbol = useAtomValue(selectedSymbolAtom);
  const bars = useAtomValue(selectedBarsAtom);
  const mode = useAtomValue(appModeAtom);
  const setBars = useSetAtom(barsAtom);
  const theme = useAtomValue(themeAtom);

  // ---- Create chart on mount / theme change ----
  useEffect(() => {
    if (!containerRef.current) return;

    const c = getChartColors();
    volColorsRef.current = { up: withAlpha(c.profit, "59"), down: withAlpha(c.loss, "59") };

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: c.bg },
        textColor: c.text,
        fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: c.grid },
        horzLines: { color: c.grid },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: c.crosshair,
          width: 1,
          style: 2,
          labelBackgroundColor: c.accent,
        },
        horzLine: {
          color: c.crosshair,
          width: 1,
          style: 2,
          labelBackgroundColor: c.accent,
        },
      },
      timeScale: {
        borderColor: c.border,
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 5,
        barSpacing: 8,
      },
      rightPriceScale: {
        borderColor: c.border,
        scaleMargins: { top: 0.1, bottom: 0.25 },
      },
      handleScroll: { vertTouchDrag: false },
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: c.profit,
      downColor: c.loss,
      borderUpColor: c.profit,
      borderDownColor: c.loss,
      wickUpColor: c.profit,
      wickDownColor: c.loss,
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
  }, [theme]);

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
    const { up, down } = volColorsRef.current;
    const volumes = bars.map((b) => toVolume(b, up, down));

    candleSeriesRef.current.setData(candles);
    volumeSeriesRef.current.setData(volumes);

    chartRef.current?.timeScale().fitContent();
  }, [bars]);

  // ---- Subscribe to live bar updates ----
  const handleBar = useCallback(
    (bar: BarData) => {
      if (!candleSeriesRef.current || !volumeSeriesRef.current) return;
      candleSeriesRef.current.update(toCandlestick(bar));
      const { up, down } = volColorsRef.current;
      volumeSeriesRef.current.update(toVolume(bar, up, down));
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
