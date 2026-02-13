import { useEffect, useState } from "react";
import { useAtomValue, useSetAtom } from "jotai";
import {
  Briefcase,
  TrendingUp,
  BarChart3,
  Zap,
  ArrowRight,
  Wifi,
  WifiOff,
  Loader2,
  ShieldCheck,
  LineChart,
  FlaskConical,
  Play,
  GitCompare,
} from "lucide-react";
import { connectionAtom, activeWorkspaceAtom } from "@/stores/workspace";
import type { WorkspaceId } from "@/stores/workspace";
import { userAtom, pageAtom } from "@/stores/auth";
import type { PageId } from "@/stores/auth";
import { apiFetch } from "@/lib/api";
import { cn, formatINR, formatPnl } from "@/lib/utils";

// ---------------------------------------------------------------------------
// API response types
// ---------------------------------------------------------------------------

interface PortfolioResponse {
  total_equity: number;
  daily_pnl: number;
  overall_return_pct: number;
}

interface StrategyItem {
  strategy_id: string;
  name: string;
  sharpe: number;
  return_pct: number;
  status: string;
  tier: string;
}

interface StrategiesResponse {
  count: number;
  strategies: StrategyItem[];
}

interface SignalItem {
  strategy_id: string;
  symbol: string;
  direction: "long" | "short";
  conviction: number;
  timestamp: number;
}

interface SignalsResponse {
  signals: SignalItem[];
}

// ---------------------------------------------------------------------------
// Connection Status Card
// ---------------------------------------------------------------------------

function ConnectionCard() {
  const conn = useAtomValue(connectionAtom);

  const feeds: { label: string; key: "zerodha" | "binance" | "fastapi"; icon: typeof Wifi }[] = [
    { label: "Zerodha", key: "zerodha", icon: Briefcase },
    { label: "Binance", key: "binance", icon: LineChart },
    { label: "API", key: "fastapi", icon: Zap },
  ];

  const statusClass = (s: string) =>
    s === "connected"
      ? "status-connected"
      : s === "connecting"
        ? "status-connecting"
        : "status-disconnected";

  const statusLabel = (s: string) =>
    s === "connected" ? "Online" : s === "connecting" ? "Connecting" : "Offline";

  return (
    <div className="panel h-full">
      <div className="panel-header">
        <span className="panel-title">Connections</span>
        {conn.latencyMs > 0 && (
          <span className="text-2xs font-mono text-terminal-muted">{conn.latencyMs}ms</span>
        )}
      </div>
      <div className="panel-body flex flex-col gap-3">
        {feeds.map((f) => {
          const st = conn[f.key];
          const Icon = st === "connected" ? Wifi : st === "connecting" ? Loader2 : WifiOff;
          return (
            <div key={f.key} className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Icon className="h-3.5 w-3.5 text-terminal-muted" />
                <span className="text-xs font-medium text-terminal-text">{f.label}</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className={cn("status-dot", statusClass(st))} />
                <span className="text-2xs text-terminal-muted">{statusLabel(st)}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Portfolio Summary Card
// ---------------------------------------------------------------------------

function PortfolioCard() {
  const [data, setData] = useState<PortfolioResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    apiFetch<PortfolioResponse>("/api/portfolio")
      .then((res) => {
        if (!cancelled) setData(res);
      })
      .catch(() => {
        if (!cancelled) setData(null);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const equity = data?.total_equity ?? 0;
  const pnl = data?.daily_pnl ?? 0;
  const retPct = data?.overall_return_pct ?? 0;

  return (
    <div className="panel h-full">
      <div className="panel-header">
        <span className="panel-title">Portfolio Summary</span>
        <Briefcase className="h-3.5 w-3.5 text-terminal-muted" />
      </div>
      <div className="panel-body">
        {loading ? (
          <div className="flex items-center justify-center h-20 text-terminal-muted text-xs">
            <Loader2 className="h-4 w-4 animate-spin mr-2" />
            Loading portfolio...
          </div>
        ) : (
          <div className="grid grid-cols-3 gap-4">
            <div className="metric-card">
              <div className="metric-label">Total Equity</div>
              <div className="metric-value">{data ? formatINR(equity) : "--"}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Daily P&L</div>
              <div
                className={cn(
                  "metric-value",
                  pnl >= 0 ? "text-terminal-profit" : "text-terminal-loss",
                )}
              >
                {data ? formatPnl(pnl) : "--"}
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Overall Return</div>
              <div
                className={cn(
                  "metric-value",
                  retPct >= 0 ? "text-terminal-profit" : "text-terminal-loss",
                )}
              >
                {data ? `${formatPnl(retPct)}%` : "--"}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Strategy Overview Card
// ---------------------------------------------------------------------------

function StrategyOverviewCard() {
  const [strategies, setStrategies] = useState<StrategyItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    apiFetch<StrategiesResponse>("/api/strategies")
      .then((res) => {
        if (!cancelled) {
          const sorted = [...res.strategies].sort((a, b) => b.sharpe - a.sharpe).slice(0, 5);
          setStrategies(sorted);
        }
      })
      .catch(() => {
        if (!cancelled) setStrategies([]);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="panel h-full">
      <div className="panel-header">
        <span className="panel-title">Top Strategies by Sharpe</span>
        <TrendingUp className="h-3.5 w-3.5 text-terminal-muted" />
      </div>
      <div className="panel-body">
        {loading ? (
          <div className="flex items-center justify-center h-20 text-terminal-muted text-xs">
            <Loader2 className="h-4 w-4 animate-spin mr-2" />
            Loading strategies...
          </div>
        ) : strategies.length === 0 ? (
          <div className="flex items-center justify-center h-20 text-terminal-muted text-xs font-mono">
            No strategies available
          </div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Strategy</th>
                <th>Status</th>
                <th className="text-right">Sharpe</th>
                <th className="text-right">Return</th>
                <th>Tier</th>
              </tr>
            </thead>
            <tbody>
              {strategies.map((s) => (
                <tr key={s.strategy_id}>
                  <td className="font-semibold text-terminal-text">{s.name || s.strategy_id}</td>
                  <td>
                    <span
                      className={cn(
                        "inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-2xs font-semibold",
                        s.status === "live" || s.status === "active"
                          ? "bg-terminal-profit/20 text-terminal-profit"
                          : s.status === "paused"
                            ? "bg-terminal-warning/20 text-terminal-warning"
                            : "bg-terminal-muted/20 text-terminal-muted",
                      )}
                    >
                      {s.status.toUpperCase()}
                    </span>
                  </td>
                  <td className="text-right">{s.sharpe.toFixed(2)}</td>
                  <td
                    className={cn(
                      "text-right",
                      s.return_pct >= 0 ? "text-terminal-profit" : "text-terminal-loss",
                    )}
                  >
                    {formatPnl(s.return_pct)}%
                  </td>
                  <td className="text-terminal-muted">{s.tier}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Recent Signals Card
// ---------------------------------------------------------------------------

function RecentSignalsCard() {
  const [signals, setSignals] = useState<SignalItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    apiFetch<SignalsResponse>("/api/signals/today")
      .then((res) => {
        if (!cancelled) setSignals(res.signals.slice(0, 10));
      })
      .catch(() => {
        if (!cancelled) setSignals([]);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const formatTime = (ts: number) => {
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString("en-IN", {
      timeZone: "Asia/Kolkata",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
  };

  return (
    <div className="panel h-full">
      <div className="panel-header">
        <span className="panel-title">Recent Signals</span>
        <Zap className="h-3.5 w-3.5 text-terminal-muted" />
      </div>
      <div className="panel-body">
        {loading ? (
          <div className="flex items-center justify-center h-20 text-terminal-muted text-xs">
            <Loader2 className="h-4 w-4 animate-spin mr-2" />
            Loading signals...
          </div>
        ) : signals.length === 0 ? (
          <div className="flex items-center justify-center h-20 text-terminal-muted text-xs font-mono">
            No signals today
          </div>
        ) : (
          <div className="flex flex-col gap-1.5 max-h-64 overflow-y-auto">
            {signals.map((sig, i) => (
              <div
                key={`${sig.strategy_id}-${sig.symbol}-${i}`}
                className="flex items-center justify-between rounded border border-terminal-border/50 bg-terminal-panel px-2 py-1.5"
              >
                <div className="flex items-center gap-2">
                  <span
                    className={cn(
                      "text-2xs font-bold uppercase px-1 py-0.5 rounded",
                      sig.direction === "long"
                        ? "bg-terminal-profit/20 text-terminal-profit"
                        : "bg-terminal-loss/20 text-terminal-loss",
                    )}
                  >
                    {sig.direction}
                  </span>
                  <span className="text-xs font-mono font-semibold text-terminal-text">
                    {sig.symbol}
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-2xs text-terminal-muted">{sig.strategy_id}</span>
                  <span className="text-2xs font-mono text-terminal-text-secondary">
                    {(sig.conviction * 100).toFixed(0)}%
                  </span>
                  <span className="text-2xs font-mono text-terminal-muted">
                    {formatTime(sig.timestamp)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Recent Backtests Card
// ---------------------------------------------------------------------------

interface BacktestJob {
  backtest_id: string;
  status: string;
  strategy_id: string;
  start_date: string;
  end_date: string;
  created_at: string;
  result: {
    sharpe_ratio: number;
    total_return: number;
    max_drawdown: number;
    n_trades: number;
  } | null;
}

function RecentBacktestsCard() {
  const [jobs, setJobs] = useState<BacktestJob[]>([]);
  const [loading, setLoading] = useState(true);
  const setPage = useSetAtom(pageAtom);
  const setWorkspace = useSetAtom(activeWorkspaceAtom);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    apiFetch<BacktestJob[]>("/api/backtest/history")
      .then((res) => {
        if (!cancelled) {
          setJobs(res.filter((j) => j.status === "completed" && j.result).slice(0, 5));
        }
      })
      .catch(() => {
        if (!cancelled) setJobs([]);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, []);

  return (
    <div className="panel h-full">
      <div className="panel-header">
        <span className="panel-title">Recent Backtests</span>
        <FlaskConical className="h-3.5 w-3.5 text-terminal-muted" />
      </div>
      <div className="panel-body">
        {loading ? (
          <div className="flex items-center justify-center h-20 text-terminal-muted text-xs">
            <Loader2 className="h-4 w-4 animate-spin mr-2" />
            Loading...
          </div>
        ) : jobs.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-20 gap-2">
            <span className="text-xs text-terminal-muted">No backtests run yet</span>
            <button
              onClick={() => { setWorkspace("backtest"); setPage("terminal"); }}
              className="btn-neutral inline-flex items-center gap-1.5 text-2xs"
            >
              <FlaskConical className="h-3 w-3" />
              Run Your First Backtest
            </button>
          </div>
        ) : (
          <div className="flex flex-col gap-1.5">
            {jobs.map((j) => {
              const r = j.result!;
              return (
                <div
                  key={j.backtest_id}
                  className="flex items-center justify-between rounded border border-terminal-border/50 bg-terminal-panel px-2.5 py-1.5"
                >
                  <div className="flex flex-col">
                    <span className="text-xs font-medium text-terminal-text">{j.strategy_id}</span>
                    <span className="text-2xs text-terminal-muted font-mono">
                      {j.start_date} to {j.end_date}
                    </span>
                  </div>
                  <div className="flex items-center gap-3 text-2xs font-mono">
                    <span className={cn(
                      "font-semibold",
                      r.sharpe_ratio >= 1.5 ? "text-terminal-profit" : r.sharpe_ratio >= 0 ? "text-terminal-warning" : "text-terminal-loss",
                    )}>
                      SR {r.sharpe_ratio.toFixed(2)}
                    </span>
                    <span className={cn(
                      r.total_return >= 0 ? "text-terminal-profit" : "text-terminal-loss",
                    )}>
                      {(r.total_return * 100).toFixed(1)}%
                    </span>
                    <span className="text-terminal-muted">{r.n_trades} trades</span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Quick Actions
// ---------------------------------------------------------------------------

function QuickActions() {
  const setPage = useSetAtom(pageAtom);
  const setWorkspace = useSetAtom(activeWorkspaceAtom);

  const navigate = (page: PageId, workspace?: WorkspaceId) => {
    if (workspace) setWorkspace(workspace);
    setPage(page);
  };

  const actions: {
    label: string;
    icon: typeof Play;
    page: PageId;
    workspace?: WorkspaceId;
  }[] = [
    { label: "Go to Trading", icon: Play, page: "terminal", workspace: "trading" },
    { label: "Run Backtest", icon: FlaskConical, page: "terminal", workspace: "backtest" },
    { label: "Compare Strategies", icon: GitCompare, page: "strategies" },
    { label: "View Risk", icon: ShieldCheck, page: "terminal", workspace: "monitor" },
    { label: "All Strategies", icon: BarChart3, page: "strategies" },
  ];

  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">Quick Actions</span>
        <ArrowRight className="h-3.5 w-3.5 text-terminal-muted" />
      </div>
      <div className="panel-body">
        <div className="flex flex-wrap gap-2">
          {actions.map((a) => (
            <button
              key={a.label}
              onClick={() => navigate(a.page, a.workspace)}
              className="btn-neutral inline-flex items-center gap-1.5"
            >
              <a.icon className="h-3.5 w-3.5" />
              {a.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Dashboard Page (default export for lazy loading)
// ---------------------------------------------------------------------------

export default function DashboardPage() {
  const user = useAtomValue(userAtom);

  return (
    <div className="flex flex-col gap-4 p-4 h-full overflow-y-auto">
      {/* Greeting */}
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-bold text-terminal-text">
          {user ? `Welcome back, ${user.name.split(" ")[0]}` : "Dashboard"}
        </h1>
        <span className="text-2xs font-mono text-terminal-muted">
          {new Date().toLocaleDateString("en-IN", {
            weekday: "long",
            year: "numeric",
            month: "short",
            day: "numeric",
          })}
        </span>
      </div>

      {/* Top row: Portfolio (2 cols) + Connection (1 col) */}
      <div className="grid grid-cols-3 gap-4">
        <div className="col-span-2">
          <PortfolioCard />
        </div>
        <div className="col-span-1">
          <ConnectionCard />
        </div>
      </div>

      {/* Middle row: Strategies (2 cols) + Signals (1 col) */}
      <div className="grid grid-cols-3 gap-4">
        <div className="col-span-2">
          <StrategyOverviewCard />
        </div>
        <div className="col-span-1">
          <RecentSignalsCard />
        </div>
      </div>

      {/* Third row: Recent Backtests (2 cols) + Quick Actions (1 col) */}
      <div className="grid grid-cols-3 gap-4">
        <div className="col-span-2">
          <RecentBacktestsCard />
        </div>
        <div className="col-span-1">
          <QuickActions />
        </div>
      </div>
    </div>
  );
}
