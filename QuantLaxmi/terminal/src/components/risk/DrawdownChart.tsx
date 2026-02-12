import { useEffect, useRef } from "react";
import {
  createChart,
  type IChartApi,
  type ISeriesApi,
  type AreaData,
  type Time,
  ColorType,
  LineStyle,
} from "lightweight-charts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface DrawdownPoint {
  time: string; // YYYY-MM-DD
  value: number; // negative pct, e.g. -2.3
}

interface DrawdownChartProps {
  /** Drawdown time-series (values should be <= 0) */
  data?: DrawdownPoint[];
  /** Max allowed drawdown as a negative pct, e.g. -5 */
  limitPct?: number;
  /** Height in pixels */
  height?: number;
}

// ---------------------------------------------------------------------------
// Default demo data — last 30 days
// ---------------------------------------------------------------------------

function generateDemoData(): DrawdownPoint[] {
  const points: DrawdownPoint[] = [];
  const now = Date.now();
  let dd = 0;
  for (let i = 29; i >= 0; i--) {
    const date = new Date(now - i * 86400000);
    const dateStr = date.toISOString().slice(0, 10);
    dd += (Math.random() - 0.45) * 0.4;
    dd = Math.min(dd, 0);
    dd = Math.max(dd, -4.5);
    points.push({ time: dateStr, value: dd });
  }
  return points;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function DrawdownChart({
  data,
  limitPct = -5,
  height = 200,
}: DrawdownChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Area"> | null>(null);

  const points = data ?? generateDemoData();

  // Create chart once
  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#6b6b8a",
        fontSize: 10,
        fontFamily: "JetBrains Mono, monospace",
      },
      grid: {
        vertLines: { color: "#1e1e2e" },
        horzLines: { color: "#1e1e2e" },
      },
      rightPriceScale: {
        borderColor: "#1e1e2e",
        scaleMargins: { top: 0.05, bottom: 0.05 },
        invertScale: true,
      },
      timeScale: {
        borderColor: "#1e1e2e",
        timeVisible: false,
      },
      crosshair: {
        horzLine: { color: "#4f8eff44", style: LineStyle.Dashed },
        vertLine: { color: "#4f8eff44", style: LineStyle.Dashed },
      },
    });

    const series = chart.addAreaSeries({
      lineColor: "#ff4d6a",
      lineWidth: 2,
      topColor: "rgba(255, 77, 106, 0.4)",
      bottomColor: "rgba(255, 77, 106, 0.02)",
      priceFormat: {
        type: "custom",
        formatter: (price: number) => `${price.toFixed(2)}%`,
      },
    });

    chartRef.current = chart;
    seriesRef.current = series;

    // Resize observer
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        chart.applyOptions({ width: entry.contentRect.width });
      }
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, [height]);

  // Update data
  useEffect(() => {
    if (!seriesRef.current || !chartRef.current) return;

    const mapped: AreaData<Time>[] = points.map((p) => ({
      time: p.time as Time,
      value: p.value,
    }));
    seriesRef.current.setData(mapped);

    // Max DD annotation via price line
    const maxDD = Math.min(...points.map((p) => p.value));
    seriesRef.current.createPriceLine({
      price: maxDD,
      color: "#ffb84d",
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      axisLabelVisible: true,
      title: `Max DD ${maxDD.toFixed(2)}%`,
    });

    // Limit line
    seriesRef.current.createPriceLine({
      price: limitPct,
      color: "#ff4d6a",
      lineWidth: 1,
      lineStyle: LineStyle.Dotted,
      axisLabelVisible: true,
      title: `Limit ${limitPct}%`,
    });

    chartRef.current.timeScale().fitContent();
  }, [points, limitPct]);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between px-1">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-terminal-muted">
          Drawdown Curve — 30d
        </h3>
        <span className="font-mono text-2xs text-terminal-loss">
          Current: {points[points.length - 1]?.value.toFixed(2) ?? "0.00"}%
        </span>
      </div>
      <div
        ref={containerRef}
        className="w-full rounded border border-terminal-border bg-terminal-surface"
        style={{ height }}
      />
    </div>
  );
}
