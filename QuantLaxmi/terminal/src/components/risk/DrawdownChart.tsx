import { useEffect, useRef, useState } from "react";
import {
  createChart,
  type IChartApi,
  type ISeriesApi,
  type AreaData,
  type Time,
  ColorType,
  LineStyle,
} from "lightweight-charts";
import { useAtomValue } from "jotai";
import { themeAtom } from "@/stores/workspace";
import { getChartColors, withAlpha } from "@/lib/chartTheme";
import { apiFetch } from "@/lib/api";

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
  const theme = useAtomValue(themeAtom);

  const [fetchedData, setFetchedData] = useState<DrawdownPoint[]>([]);

  useEffect(() => {
    if (data) return;
    let active = true;
    const fetchDD = () => {
      apiFetch<{
        drawdown_history: Array<{ date: string; drawdown_pct: number }>;
      }>("/api/portfolio").then((portfolio) => {
        if (!active) return;
        setFetchedData(
          (portfolio.drawdown_history ?? []).map((p) => ({
            time: p.date,
            value: -p.drawdown_pct,
          }))
        );
      }).catch(() => {});
    };
    fetchDD();
    const interval = setInterval(fetchDD, 10000);
    return () => { active = false; clearInterval(interval); };
  }, [data]);

  const points = data ?? fetchedData;

  // Create chart once (recreate on theme change)
  useEffect(() => {
    if (!containerRef.current) return;

    const c = getChartColors();
    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: c.text,
        fontSize: 10,
        fontFamily: "JetBrains Mono, monospace",
      },
      grid: {
        vertLines: { color: c.grid },
        horzLines: { color: c.grid },
      },
      rightPriceScale: {
        borderColor: c.border,
        scaleMargins: { top: 0.05, bottom: 0.05 },
        invertScale: true,
      },
      timeScale: {
        borderColor: c.border,
        timeVisible: false,
      },
      crosshair: {
        horzLine: { color: c.crosshair, style: LineStyle.Dashed },
        vertLine: { color: c.crosshair, style: LineStyle.Dashed },
      },
    });

    const series = chart.addAreaSeries({
      lineColor: c.loss,
      lineWidth: 2,
      topColor: withAlpha(c.loss, "66"),
      bottomColor: withAlpha(c.loss, "05"),
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
  }, [height, theme]);

  // Update data
  useEffect(() => {
    if (!seriesRef.current || !chartRef.current) return;

    const c = getChartColors();
    const mapped: AreaData<Time>[] = points.map((p) => ({
      time: p.time as Time,
      value: p.value,
    }));
    seriesRef.current.setData(mapped);

    if (points.length > 0) {
      // Max DD annotation via price line
      const maxDD = Math.min(...points.map((p) => p.value));
      seriesRef.current.createPriceLine({
        price: maxDD,
        color: c.warning,
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: `Max DD ${maxDD.toFixed(2)}%`,
      });

      // Limit line
      seriesRef.current.createPriceLine({
        price: limitPct,
        color: c.loss,
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
        axisLabelVisible: true,
        title: `Limit ${limitPct}%`,
      });

      chartRef.current.timeScale().fitContent();
    }
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
