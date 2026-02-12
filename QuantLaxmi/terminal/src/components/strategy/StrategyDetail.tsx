import { useEffect, useRef, useState } from "react";
import {
  createChart,
  type IChartApi,
  type Time,
  ColorType,
  LineStyle,
} from "lightweight-charts";
import { useTauriCommand } from "@/hooks/useTauriCommand";
import { cn } from "@/lib/utils";
import type { StrategyStatus } from "@/components/strategy/StrategyPanel";

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

interface Trade {
  id: string;
  timestamp: string;
  symbol: string;
  side: "BUY" | "SELL";
  quantity: number;
  entryPrice: number;
  exitPrice: number;
  pnl: number;
  holdingPeriod: string;
}

interface PositionRow {
  symbol: string;
  side: "LONG" | "SHORT";
  quantity: number;
  avgPrice: number;
  ltp: number;
  unrealizedPnl: number;
}

interface SignalHistoryItem {
  timestamp: string;
  direction: "BUY" | "SELL";
  conviction: number;
  acted: boolean;
}

interface StrategyDetailData {
  id: string;
  name: string;
  status: StrategyStatus;
  equity: number;
  returnPct: number;
  sharpe: number;
  sortino: number;
  maxDD: number;
  winRate: number;
  profitFactor: number;
  totalTrades: number;
  equityCurve: EquityPoint[];
  drawdownCurve: DrawdownPoint[];
  recentTrades: Trade[];
  positions: PositionRow[];
  signalHistory: SignalHistoryItem[];
  parameters: Record<string, string | number>;
}

interface StrategyDetailProps {
  strategyId: string;
  onBack?: () => void;
}

// ---------------------------------------------------------------------------
// Demo data
// ---------------------------------------------------------------------------

function generateEquityCurve(): EquityPoint[] {
  const pts: EquityPoint[] = [];
  const now = Date.now();
  let eq = 10000000;
  for (let i = 89; i >= 0; i--) {
    const d = new Date(now - i * 86400000);
    eq += (Math.random() - 0.42) * 25000;
    eq = Math.max(eq, 9500000);
    pts.push({ time: d.toISOString().slice(0, 10), value: eq });
  }
  return pts;
}

function generateDrawdownCurve(): DrawdownPoint[] {
  const pts: DrawdownPoint[] = [];
  const now = Date.now();
  let dd = 0;
  for (let i = 89; i >= 0; i--) {
    const d = new Date(now - i * 86400000);
    dd += (Math.random() - 0.45) * 0.3;
    dd = Math.min(dd, 0);
    dd = Math.max(dd, -4);
    pts.push({ time: d.toISOString().slice(0, 10), value: dd });
  }
  return pts;
}

const DEFAULT_DETAIL: StrategyDetailData = {
  id: "s25",
  name: "S25 Divergence Flow Field",
  status: "live",
  equity: 10609000,
  returnPct: 6.09,
  sharpe: 1.87,
  sortino: 2.64,
  maxDD: 2.05,
  winRate: 0.64,
  profitFactor: 1.78,
  totalTrades: 93,
  equityCurve: generateEquityCurve(),
  drawdownCurve: generateDrawdownCurve(),
  recentTrades: [
    { id: "t1", timestamp: "2026-02-12 14:32", symbol: "NIFTY", side: "SELL", quantity: 50, entryPrice: 23480, exitPrice: 23410, pnl: 3500, holdingPeriod: "1.2d" },
    { id: "t2", timestamp: "2026-02-11 10:15", symbol: "BANKNIFTY", side: "BUY", quantity: 25, entryPrice: 49820, exitPrice: 50190, pnl: 9250, holdingPeriod: "0.8d" },
    { id: "t3", timestamp: "2026-02-10 11:45", symbol: "NIFTY", side: "BUY", quantity: 50, entryPrice: 23320, exitPrice: 23395, pnl: 3750, holdingPeriod: "1.4d" },
    { id: "t4", timestamp: "2026-02-09 09:30", symbol: "BANKNIFTY", side: "SELL", quantity: 25, entryPrice: 50100, exitPrice: 50250, pnl: -3750, holdingPeriod: "0.5d" },
    { id: "t5", timestamp: "2026-02-07 13:20", symbol: "NIFTY", side: "BUY", quantity: 50, entryPrice: 23250, exitPrice: 23380, pnl: 6500, holdingPeriod: "1.1d" },
  ],
  positions: [
    { symbol: "NIFTY", side: "LONG", quantity: 50, avgPrice: 23420, ltp: 23485, unrealizedPnl: 3250 },
  ],
  signalHistory: [
    { timestamp: "2026-02-12 14:30", direction: "BUY", conviction: 0.82, acted: true },
    { timestamp: "2026-02-12 10:15", direction: "SELL", conviction: 0.45, acted: false },
    { timestamp: "2026-02-11 14:00", direction: "BUY", conviction: 0.71, acted: true },
    { timestamp: "2026-02-11 09:30", direction: "SELL", conviction: 0.88, acted: true },
    { timestamp: "2026-02-10 13:45", direction: "BUY", conviction: 0.65, acted: true },
  ],
  parameters: {
    threshold: 0.3,
    scale: 1.5,
    lookback: 20,
    cost_bps: 3,
    max_position: 1.0,
  },
};

// ---------------------------------------------------------------------------
// Chart component (equity + drawdown stacked)
// ---------------------------------------------------------------------------

function EquityChart({ equity, drawdown }: { equity: EquityPoint[]; drawdown: DrawdownPoint[] }) {
  const eqRef = useRef<HTMLDivElement>(null);
  const ddRef = useRef<HTMLDivElement>(null);
  const eqChartRef = useRef<IChartApi | null>(null);
  const ddChartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!eqRef.current || !ddRef.current) return;

    const chartOpts = {
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

    // Equity chart
    const eqChart = createChart(eqRef.current, { ...chartOpts, height: 180 });
    const eqSeries = eqChart.addAreaSeries({
      lineColor: "#4f8eff",
      lineWidth: 2,
      topColor: "rgba(79,142,255,0.3)",
      bottomColor: "rgba(79,142,255,0.02)",
      priceFormat: { type: "custom", formatter: (p: number) => `₹${(p / 100000).toFixed(1)}L` },
    });
    eqSeries.setData(equity.map((p) => ({ time: p.time as Time, value: p.value })));
    eqChart.timeScale().fitContent();
    eqChartRef.current = eqChart;

    // Drawdown chart
    const ddChart = createChart(ddRef.current, { ...chartOpts, height: 100 });
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
    ddChartRef.current = ddChart;

    // Sync time scales
    eqChart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
      if (range) ddChart.timeScale().setVisibleLogicalRange(range);
    });
    ddChart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
      if (range) eqChart.timeScale().setVisibleLogicalRange(range);
    });

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
// Metrics row
// ---------------------------------------------------------------------------

function MetricPill({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex flex-col items-center gap-0.5 rounded bg-terminal-surface px-3 py-1.5 border border-terminal-border">
      <span className="text-2xs text-terminal-muted uppercase">{label}</span>
      <span className={cn("font-mono text-sm font-semibold", color ?? "text-gray-100")}>{value}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function StrategyDetail({ strategyId, onBack }: StrategyDetailProps) {
  const { data, execute } = useTauriCommand<StrategyDetailData>("get_strategy_detail");
  const [detail, setDetail] = useState<StrategyDetailData>(DEFAULT_DETAIL);

  useEffect(() => {
    execute({ strategy_id: strategyId }).catch(() => {});
  }, [execute, strategyId]);

  useEffect(() => {
    if (data) setDetail(data);
  }, [data]);

  const statusColor = {
    live: "text-terminal-profit",
    paused: "text-terminal-warning",
    stopped: "text-terminal-loss",
  }[detail.status];

  return (
    <div className="flex flex-col gap-4 p-4 h-full overflow-y-auto">
      {/* Header */}
      <div className="flex items-center gap-3">
        {onBack && (
          <button
            onClick={onBack}
            className="rounded p-1.5 hover:bg-terminal-surface transition-colors text-terminal-muted hover:text-gray-100"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M10 12L6 8l4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        )}
        <div className="flex-1">
          <h2 className="text-sm font-bold text-gray-100">{detail.name}</h2>
          <span className={cn("text-2xs font-semibold uppercase", statusColor)}>{detail.status}</span>
        </div>
        <span className="font-mono text-lg font-bold text-gray-100">
          {`₹${(detail.equity / 100000).toFixed(1)}L`}
        </span>
      </div>

      {/* Metrics row */}
      <div className="flex flex-wrap gap-2">
        <MetricPill label="Return" value={`${detail.returnPct >= 0 ? "+" : ""}${detail.returnPct.toFixed(2)}%`} color={detail.returnPct >= 0 ? "text-terminal-profit" : "text-terminal-loss"} />
        <MetricPill label="Sharpe" value={detail.sharpe.toFixed(2)} />
        <MetricPill label="Sortino" value={detail.sortino.toFixed(2)} />
        <MetricPill label="Max DD" value={`${detail.maxDD.toFixed(2)}%`} color="text-terminal-loss" />
        <MetricPill label="Win Rate" value={`${(detail.winRate * 100).toFixed(0)}%`} />
        <MetricPill label="PF" value={detail.profitFactor.toFixed(2)} />
        <MetricPill label="Trades" value={String(detail.totalTrades)} />
      </div>

      {/* Charts */}
      <EquityChart equity={detail.equityCurve} drawdown={detail.drawdownCurve} />

      {/* Positions */}
      {detail.positions.length > 0 && (
        <section>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-terminal-muted mb-2">
            Open Positions
          </h3>
          <div className="overflow-x-auto rounded border border-terminal-border">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="bg-terminal-surface text-terminal-muted">
                  <th className="px-3 py-1.5 text-left">Symbol</th>
                  <th className="px-3 py-1.5 text-left">Side</th>
                  <th className="px-3 py-1.5 text-right">Qty</th>
                  <th className="px-3 py-1.5 text-right">Avg</th>
                  <th className="px-3 py-1.5 text-right">LTP</th>
                  <th className="px-3 py-1.5 text-right">P&L</th>
                </tr>
              </thead>
              <tbody>
                {detail.positions.map((pos) => (
                  <tr key={pos.symbol} className="border-t border-terminal-border hover:bg-terminal-surface/50">
                    <td className="px-3 py-1.5 text-gray-200">{pos.symbol}</td>
                    <td className={cn("px-3 py-1.5", pos.side === "LONG" ? "text-terminal-profit" : "text-terminal-loss")}>{pos.side}</td>
                    <td className="px-3 py-1.5 text-right text-gray-200">{pos.quantity}</td>
                    <td className="px-3 py-1.5 text-right text-gray-300">{pos.avgPrice.toLocaleString("en-IN")}</td>
                    <td className="px-3 py-1.5 text-right text-gray-100">{pos.ltp.toLocaleString("en-IN")}</td>
                    <td className={cn("px-3 py-1.5 text-right font-semibold", pos.unrealizedPnl >= 0 ? "text-terminal-profit" : "text-terminal-loss")}>
                      {pos.unrealizedPnl >= 0 ? "+" : ""}{pos.unrealizedPnl.toLocaleString("en-IN")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* Recent trades */}
      <section>
        <h3 className="text-xs font-semibold uppercase tracking-wider text-terminal-muted mb-2">
          Recent Trades
        </h3>
        <div className="overflow-x-auto rounded border border-terminal-border">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="bg-terminal-surface text-terminal-muted">
                <th className="px-3 py-1.5 text-left">Time</th>
                <th className="px-3 py-1.5 text-left">Symbol</th>
                <th className="px-3 py-1.5 text-left">Side</th>
                <th className="px-3 py-1.5 text-right">Qty</th>
                <th className="px-3 py-1.5 text-right">Entry</th>
                <th className="px-3 py-1.5 text-right">Exit</th>
                <th className="px-3 py-1.5 text-right">P&L</th>
                <th className="px-3 py-1.5 text-right">Hold</th>
              </tr>
            </thead>
            <tbody>
              {detail.recentTrades.map((t) => (
                <tr key={t.id} className="border-t border-terminal-border hover:bg-terminal-surface/50">
                  <td className="px-3 py-1.5 text-gray-400">{t.timestamp}</td>
                  <td className="px-3 py-1.5 text-gray-200">{t.symbol}</td>
                  <td className={cn("px-3 py-1.5", t.side === "BUY" ? "text-terminal-profit" : "text-terminal-loss")}>{t.side}</td>
                  <td className="px-3 py-1.5 text-right text-gray-200">{t.quantity}</td>
                  <td className="px-3 py-1.5 text-right text-gray-300">{t.entryPrice.toLocaleString("en-IN")}</td>
                  <td className="px-3 py-1.5 text-right text-gray-300">{t.exitPrice.toLocaleString("en-IN")}</td>
                  <td className={cn("px-3 py-1.5 text-right font-semibold", t.pnl >= 0 ? "text-terminal-profit" : "text-terminal-loss")}>
                    {t.pnl >= 0 ? "+" : ""}{t.pnl.toLocaleString("en-IN")}
                  </td>
                  <td className="px-3 py-1.5 text-right text-terminal-muted">{t.holdingPeriod}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Signal history */}
      <section>
        <h3 className="text-xs font-semibold uppercase tracking-wider text-terminal-muted mb-2">
          Signal History
        </h3>
        <div className="space-y-1">
          {detail.signalHistory.map((sig, i) => (
            <div
              key={i}
              className={cn(
                "flex items-center gap-3 rounded px-3 py-1.5 text-xs",
                sig.acted ? "bg-terminal-surface" : "bg-terminal-surface/40 opacity-60",
              )}
            >
              <span className="font-mono text-gray-400 min-w-[110px]">{sig.timestamp}</span>
              <span className={cn("font-semibold min-w-[32px]", sig.direction === "BUY" ? "text-terminal-profit" : "text-terminal-loss")}>
                {sig.direction}
              </span>
              <div className="flex-1 h-1 rounded-full bg-terminal-bg overflow-hidden max-w-[100px]">
                <div
                  className="h-full rounded-full bg-terminal-accent"
                  style={{ width: `${sig.conviction * 100}%` }}
                />
              </div>
              <span className="font-mono text-terminal-muted">{(sig.conviction * 100).toFixed(0)}%</span>
              <span className={cn("text-2xs", sig.acted ? "text-terminal-profit" : "text-terminal-muted")}>
                {sig.acted ? "ACTED" : "SKIPPED"}
              </span>
            </div>
          ))}
        </div>
      </section>

      {/* Parameters */}
      <section>
        <h3 className="text-xs font-semibold uppercase tracking-wider text-terminal-muted mb-2">
          Parameters
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2">
          {Object.entries(detail.parameters).map(([key, val]) => (
            <div key={key} className="rounded bg-terminal-surface border border-terminal-border px-3 py-1.5">
              <div className="text-2xs text-terminal-muted">{key}</div>
              <div className="font-mono text-xs text-gray-100">{String(val)}</div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
