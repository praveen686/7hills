import { useEffect, useRef, useState } from "react";
import {
  createChart,
  type IChartApi,
  type Time,
  ColorType,
  LineStyle,
} from "lightweight-charts";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface EquityPoint {
  time: string;
  value: number;
}

interface DrawdownPoint {
  time: string;
  value: number;
}

interface MonthlyReturn {
  year: number;
  month: number; // 1-12
  returnPct: number;
}

interface TradeRow {
  id: string;
  timestamp: string;
  symbol: string;
  side: "BUY" | "SELL";
  quantity: number;
  entryPrice: number;
  exitPrice: number;
  pnl: number;
  returnPct: number;
}

export interface BacktestResultData {
  totalReturn: number;
  sharpe: number;
  sortino: number;
  maxDD: number;
  winRate: number;
  profitFactor: number;
  totalTrades: number;
  equityCurve: EquityPoint[];
  drawdownCurve: DrawdownPoint[];
  monthlyReturns: MonthlyReturn[];
  trades: TradeRow[];
}

interface BacktestResultsProps {
  data?: BacktestResultData;
}

// ---------------------------------------------------------------------------
// Demo data
// ---------------------------------------------------------------------------

function generateDemo(): BacktestResultData {
  const equity: EquityPoint[] = [];
  const drawdown: DrawdownPoint[] = [];
  const now = Date.now();
  let eq = 10000000;
  let peak = eq;

  for (let i = 365; i >= 0; i--) {
    const d = new Date(now - i * 86400000);
    const dateStr = d.toISOString().slice(0, 10);
    eq += (Math.random() - 0.42) * 18000;
    eq = Math.max(eq, 9200000);
    peak = Math.max(peak, eq);
    const dd = ((eq - peak) / peak) * 100;
    equity.push({ time: dateStr, value: eq });
    drawdown.push({ time: dateStr, value: dd });
  }

  const monthly: MonthlyReturn[] = [];
  for (let y = 2025; y <= 2026; y++) {
    const maxM = y === 2026 ? 2 : 12;
    for (let m = 1; m <= maxM; m++) {
      monthly.push({ year: y, month: m, returnPct: (Math.random() - 0.35) * 3 });
    }
  }

  const trades: TradeRow[] = [];
  const symbols = ["NIFTY", "BANKNIFTY"];
  const sides: ("BUY" | "SELL")[] = ["BUY", "SELL"];
  for (let i = 0; i < 20; i++) {
    const sym = symbols[i % 2];
    const side = sides[Math.floor(Math.random() * 2)];
    const entry = sym === "NIFTY" ? 23000 + Math.random() * 500 : 49500 + Math.random() * 1000;
    const pnl = (Math.random() - 0.4) * 15000;
    trades.push({
      id: `bt-${i}`,
      timestamp: new Date(now - (i + 1) * 3 * 86400000).toISOString().slice(0, 16).replace("T", " "),
      symbol: sym,
      side,
      quantity: sym === "NIFTY" ? 50 : 25,
      entryPrice: Math.round(entry * 100) / 100,
      exitPrice: Math.round((entry + pnl / (sym === "NIFTY" ? 50 : 25)) * 100) / 100,
      pnl: Math.round(pnl),
      returnPct: Math.round((pnl / 10000000) * 10000) / 100,
    });
  }

  return {
    totalReturn: 6.09,
    sharpe: 1.87,
    sortino: 2.64,
    maxDD: 2.05,
    winRate: 0.64,
    profitFactor: 1.78,
    totalTrades: 93,
    equityCurve: equity,
    drawdownCurve: drawdown,
    monthlyReturns: monthly,
    trades,
  };
}

const DEMO = generateDemo();

// ---------------------------------------------------------------------------
// Metric pill
// ---------------------------------------------------------------------------

function MetricCard({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex flex-col items-center gap-0.5 rounded bg-terminal-surface px-3 py-2 border border-terminal-border min-w-[80px]">
      <span className="text-2xs text-terminal-muted uppercase tracking-wider">{label}</span>
      <span className={cn("font-mono text-sm font-bold", color ?? "text-gray-100")}>{value}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Charts
// ---------------------------------------------------------------------------

function EquityAndDrawdownChart({ equity, drawdown }: { equity: EquityPoint[]; drawdown: DrawdownPoint[] }) {
  const eqRef = useRef<HTMLDivElement>(null);
  const ddRef = useRef<HTMLDivElement>(null);
  const chartsRef = useRef<IChartApi[]>([]);

  useEffect(() => {
    if (!eqRef.current || !ddRef.current) return;

    const opts = {
      width: eqRef.current.clientWidth,
      layout: {
        background: { type: ColorType.Solid as const, color: "transparent" },
        textColor: "#6b6b8a",
        fontSize: 10,
        fontFamily: "JetBrains Mono, monospace",
      },
      grid: { vertLines: { color: "#1e1e2e" }, horzLines: { color: "#1e1e2e" } },
      rightPriceScale: { borderColor: "#1e1e2e" },
      timeScale: { borderColor: "#1e1e2e", timeVisible: false },
      crosshair: {
        horzLine: { color: "#4f8eff44", style: LineStyle.Dashed },
        vertLine: { color: "#4f8eff44", style: LineStyle.Dashed },
      },
    };

    const eqChart = createChart(eqRef.current, { ...opts, height: 200 });
    const eqSeries = eqChart.addAreaSeries({
      lineColor: "#4f8eff",
      lineWidth: 2,
      topColor: "rgba(79,142,255,0.25)",
      bottomColor: "rgba(79,142,255,0.01)",
      priceFormat: { type: "custom", formatter: (p: number) => `₹${(p / 100000).toFixed(1)}L` },
    });
    eqSeries.setData(equity.map((p) => ({ time: p.time as Time, value: p.value })));
    eqChart.timeScale().fitContent();

    const ddChart = createChart(ddRef.current, { ...opts, height: 100 });
    const ddSeries = ddChart.addAreaSeries({
      lineColor: "#ff4d6a",
      lineWidth: 2,
      topColor: "rgba(255,77,106,0.3)",
      bottomColor: "rgba(255,77,106,0.02)",
      invertFilledArea: true,
      priceFormat: { type: "custom", formatter: (p: number) => `${p.toFixed(2)}%` },
    });
    ddSeries.setData(drawdown.map((p) => ({ time: p.time as Time, value: p.value })));
    ddChart.timeScale().fitContent();

    // Sync time scales
    eqChart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
      if (range) ddChart.timeScale().setVisibleLogicalRange(range);
    });
    ddChart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
      if (range) eqChart.timeScale().setVisibleLogicalRange(range);
    });

    chartsRef.current = [eqChart, ddChart];

    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = entry.contentRect.width;
        eqChart.applyOptions({ width: w });
        ddChart.applyOptions({ width: w });
      }
    });
    ro.observe(eqRef.current);

    return () => {
      ro.disconnect();
      eqChart.remove();
      ddChart.remove();
    };
  }, [equity, drawdown]);

  return (
    <div className="flex flex-col gap-1">
      <span className="text-2xs text-terminal-muted uppercase tracking-wider px-1">Equity Curve</span>
      <div ref={eqRef} className="w-full rounded border border-terminal-border bg-terminal-surface" />
      <span className="text-2xs text-terminal-muted uppercase tracking-wider px-1 mt-1">Drawdown</span>
      <div ref={ddRef} className="w-full rounded border border-terminal-border bg-terminal-surface" />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Monthly returns heatmap
// ---------------------------------------------------------------------------

const MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function MonthlyHeatmap({ returns }: { returns: MonthlyReturn[] }) {
  const years = [...new Set(returns.map((r) => r.year))].sort();
  const maxAbs = Math.max(...returns.map((r) => Math.abs(r.returnPct)), 0.01);

  const getReturn = (year: number, month: number): MonthlyReturn | undefined =>
    returns.find((r) => r.year === year && r.month === month);

  const cellBg = (val: number | undefined): string => {
    if (val === undefined) return "transparent";
    const intensity = Math.min(Math.abs(val) / maxAbs, 1);
    const alpha = (0.1 + intensity * 0.6).toFixed(2);
    return val >= 0 ? `rgba(0,212,170,${alpha})` : `rgba(255,77,106,${alpha})`;
  };

  return (
    <section>
      <h3 className="text-xs font-semibold uppercase tracking-wider text-terminal-muted mb-2">
        Monthly Returns
      </h3>
      <div className="overflow-x-auto rounded border border-terminal-border">
        <table className="w-full text-2xs font-mono border-collapse">
          <thead>
            <tr className="bg-terminal-surface">
              <th className="px-2 py-1.5 text-left text-terminal-muted border-b border-terminal-border">Year</th>
              {MONTH_NAMES.map((m) => (
                <th key={m} className="px-2 py-1.5 text-center text-terminal-muted border-b border-terminal-border">{m}</th>
              ))}
              <th className="px-2 py-1.5 text-center text-terminal-accent font-semibold border-b border-l border-terminal-border">YTD</th>
            </tr>
          </thead>
          <tbody>
            {years.map((year) => {
              const yearReturns = returns.filter((r) => r.year === year);
              const ytd = yearReturns.reduce((s, r) => s + r.returnPct, 0);
              return (
                <tr key={year}>
                  <td className="px-2 py-1.5 text-gray-300 font-semibold border-b border-terminal-border">{year}</td>
                  {Array.from({ length: 12 }, (_, i) => {
                    const ret = getReturn(year, i + 1);
                    return (
                      <td
                        key={i}
                        className="px-2 py-1.5 text-center border-b border-terminal-border"
                        style={{ backgroundColor: cellBg(ret?.returnPct) }}
                      >
                        <span className={cn(
                          ret === undefined ? "text-terminal-muted/30" : ret.returnPct >= 0 ? "text-terminal-profit" : "text-terminal-loss",
                        )}>
                          {ret ? `${ret.returnPct >= 0 ? "+" : ""}${ret.returnPct.toFixed(1)}` : "--"}
                        </span>
                      </td>
                    );
                  })}
                  <td
                    className={cn(
                      "px-2 py-1.5 text-center font-semibold border-b border-l border-terminal-border",
                      ytd >= 0 ? "text-terminal-profit" : "text-terminal-loss",
                    )}
                  >
                    {ytd >= 0 ? "+" : ""}{ytd.toFixed(1)}%
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Trade list
// ---------------------------------------------------------------------------

type SortKey = "timestamp" | "pnl" | "returnPct" | "symbol";

function TradeList({ trades }: { trades: TradeRow[] }) {
  const [sortKey, setSortKey] = useState<SortKey>("timestamp");
  const [sortAsc, setSortAsc] = useState(false);

  const sorted = [...trades].sort((a, b) => {
    const mul = sortAsc ? 1 : -1;
    if (sortKey === "timestamp") return mul * a.timestamp.localeCompare(b.timestamp);
    if (sortKey === "symbol") return mul * a.symbol.localeCompare(b.symbol);
    return mul * ((a[sortKey as keyof TradeRow] as number) - (b[sortKey as keyof TradeRow] as number));
  });

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(false);
    }
  };

  const sortIcon = (key: SortKey) => {
    if (sortKey !== key) return "";
    return sortAsc ? " \u25B2" : " \u25BC";
  };

  return (
    <section>
      <h3 className="text-xs font-semibold uppercase tracking-wider text-terminal-muted mb-2">
        Trades ({trades.length})
      </h3>
      <div className="overflow-x-auto rounded border border-terminal-border max-h-[300px] overflow-y-auto">
        <table className="w-full text-xs font-mono border-collapse">
          <thead className="sticky top-0 z-10">
            <tr className="bg-terminal-surface text-terminal-muted">
              <th className="px-3 py-1.5 text-left cursor-pointer hover:text-gray-200" onClick={() => handleSort("timestamp")}>Time{sortIcon("timestamp")}</th>
              <th className="px-3 py-1.5 text-left cursor-pointer hover:text-gray-200" onClick={() => handleSort("symbol")}>Symbol{sortIcon("symbol")}</th>
              <th className="px-3 py-1.5 text-left">Side</th>
              <th className="px-3 py-1.5 text-right">Qty</th>
              <th className="px-3 py-1.5 text-right">Entry</th>
              <th className="px-3 py-1.5 text-right">Exit</th>
              <th className="px-3 py-1.5 text-right cursor-pointer hover:text-gray-200" onClick={() => handleSort("pnl")}>P&L{sortIcon("pnl")}</th>
              <th className="px-3 py-1.5 text-right cursor-pointer hover:text-gray-200" onClick={() => handleSort("returnPct")}>Ret%{sortIcon("returnPct")}</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((t) => (
              <tr key={t.id} className="border-t border-terminal-border hover:bg-terminal-surface/50">
                <td className="px-3 py-1 text-gray-400">{t.timestamp}</td>
                <td className="px-3 py-1 text-gray-200">{t.symbol}</td>
                <td className={cn("px-3 py-1", t.side === "BUY" ? "text-terminal-profit" : "text-terminal-loss")}>{t.side}</td>
                <td className="px-3 py-1 text-right text-gray-200">{t.quantity}</td>
                <td className="px-3 py-1 text-right text-gray-300">{t.entryPrice.toLocaleString("en-IN")}</td>
                <td className="px-3 py-1 text-right text-gray-300">{t.exitPrice.toLocaleString("en-IN")}</td>
                <td className={cn("px-3 py-1 text-right font-semibold", t.pnl >= 0 ? "text-terminal-profit" : "text-terminal-loss")}>
                  {t.pnl >= 0 ? "+" : ""}{t.pnl.toLocaleString("en-IN")}
                </td>
                <td className={cn("px-3 py-1 text-right", t.returnPct >= 0 ? "text-terminal-profit" : "text-terminal-loss")}>
                  {t.returnPct >= 0 ? "+" : ""}{t.returnPct.toFixed(2)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function BacktestResults({ data }: BacktestResultsProps) {
  const result = data ?? DEMO;

  return (
    <div className="flex flex-col gap-4 p-4 h-full overflow-y-auto">
      <h2 className="text-sm font-bold uppercase tracking-widest text-gray-100">
        Backtest Results
      </h2>

      {/* Summary metrics */}
      <div className="flex flex-wrap gap-2">
        <MetricCard label="Total Return" value={`${result.totalReturn >= 0 ? "+" : ""}${result.totalReturn.toFixed(2)}%`} color={result.totalReturn >= 0 ? "text-terminal-profit" : "text-terminal-loss"} />
        <MetricCard label="Sharpe" value={result.sharpe.toFixed(2)} />
        <MetricCard label="Sortino" value={result.sortino.toFixed(2)} />
        <MetricCard label="Max DD" value={`${result.maxDD.toFixed(2)}%`} color="text-terminal-loss" />
        <MetricCard label="Win Rate" value={`${(result.winRate * 100).toFixed(0)}%`} />
        <MetricCard label="PF" value={result.profitFactor.toFixed(2)} />
        <MetricCard label="Trades" value={String(result.totalTrades)} />
      </div>

      <EquityAndDrawdownChart equity={result.equityCurve} drawdown={result.drawdownCurve} />
      <MonthlyHeatmap returns={result.monthlyReturns} />
      <TradeList trades={result.trades} />
    </div>
  );
}
