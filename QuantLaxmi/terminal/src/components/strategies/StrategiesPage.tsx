// ============================================================
// StrategiesPage — two-column strategy browser + config editor
// ============================================================

import { useEffect, useState, useCallback } from "react";
import { useSetAtom } from "jotai";
import { apiFetch } from "@/lib/api";
import { cn } from "@/lib/utils";
import * as Switch from "@radix-ui/react-switch";
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Beaker,
  Pause,
  Settings2,
  BarChart3,
  Target,
  Calendar,
  Hash,
  Layers,
  GitCompare,
  FlaskConical,
} from "lucide-react";
import { ParamEditor, type ParamMeta } from "./ParamEditor";
import { BacktestCompare } from "@/components/backtest/BacktestCompare";
import { pageAtom } from "@/stores/auth";
import { activeWorkspaceAtom } from "@/stores/workspace";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface StrategySummary {
  strategy_id: string;
  name: string;
  status: string;
  sharpe: number;
  return_pct: number;
  max_dd: number;
  win_rate: number;
  n_closed: number;
  tier: string;
  best_config: string;
  equity: number;
  n_open: number;
}

interface StrategyDetail {
  strategy_id: string;
  name: string;
  status: string;
  sharpe: number;
  return_pct: number;
  max_dd: number;
  win_rate: number;
  n_closed: number;
  n_open: number;
  tier: string;
  best_config: string;
  date_range: string;
  equity: number;
  metadata: Record<string, unknown>;
}

interface StrategiesListResponse {
  count: number;
  strategies: StrategySummary[];
}

interface ParamResponse {
  params: Record<string, ParamMeta>;
}

interface ToggleResponse {
  strategy_id: string;
  enabled: boolean;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const TIER_NAMES: Record<string, string> = {
  tier1: "Alpha",
  tier2: "Marginal",
  tier3: "Experimental",
  tier4: "Inactive",
  signal_only: "Research",
};

const TIER_ORDER: Record<string, number> = {
  tier1: 0,
  tier2: 1,
  tier3: 2,
  signal_only: 3,
  tier4: 4,
};

// ---------------------------------------------------------------------------
// Sharpe badge
// ---------------------------------------------------------------------------

function SharpeBadge({ sharpe }: { sharpe: number }) {
  const color =
    sharpe >= 1.5
      ? "bg-terminal-profit/20 text-terminal-profit"
      : sharpe >= 0
        ? "bg-terminal-warning/20 text-terminal-warning"
        : "bg-terminal-loss/20 text-terminal-loss";

  return (
    <span
      className={cn(
        "inline-flex items-center rounded px-1.5 py-0.5 text-2xs font-mono font-semibold tabular-nums",
        color,
      )}
    >
      {sharpe.toFixed(2)}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Status badge
// ---------------------------------------------------------------------------

function StatusBadge({ status }: { status: string }) {
  const configs: Record<string, { label: string; cls: string }> = {
    active: { label: "ACTIVE", cls: "bg-terminal-profit/20 text-terminal-profit" },
    marginal: { label: "MARGINAL", cls: "bg-terminal-warning/20 text-terminal-warning" },
    research: { label: "RESEARCH", cls: "bg-terminal-info/20 text-terminal-info" },
    negative: { label: "NEGATIVE", cls: "bg-terminal-loss/20 text-terminal-loss" },
    inactive: { label: "INACTIVE", cls: "bg-terminal-muted/20 text-terminal-muted" },
    unknown: { label: "UNKNOWN", cls: "bg-terminal-muted/20 text-terminal-muted" },
  };
  const c = configs[status] ?? configs.unknown;
  return (
    <span className={cn("inline-flex items-center rounded px-1.5 py-0.5 text-2xs font-semibold", c.cls)}>
      {c.label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Tier icon
// ---------------------------------------------------------------------------

function TierIcon({ tier }: { tier: string }) {
  const cls = "h-3.5 w-3.5 text-terminal-muted";
  switch (tier) {
    case "tier1":
      return <TrendingUp className={cn(cls, "text-terminal-profit")} />;
    case "tier2":
      return <Activity className={cn(cls, "text-terminal-warning")} />;
    case "tier3":
      return <Beaker className={cn(cls, "text-terminal-info")} />;
    case "tier4":
      return <Pause className={cn(cls, "text-terminal-muted")} />;
    case "signal_only":
      return <TrendingDown className={cn(cls, "text-terminal-muted")} />;
    default:
      return null;
  }
}

// ---------------------------------------------------------------------------
// Strategy row in left sidebar
// ---------------------------------------------------------------------------

function StrategyRow({
  strategy,
  selected,
  enabled,
  onSelect,
  onToggle,
}: {
  strategy: StrategySummary;
  selected: boolean;
  enabled: boolean;
  onSelect: () => void;
  onToggle: () => void;
}) {
  return (
    <button
      onClick={onSelect}
      className={cn(
        "flex items-center gap-2.5 w-full rounded-md px-2.5 py-2 text-left transition-colors group",
        selected
          ? "bg-terminal-accent/10 border border-terminal-accent/30"
          : "border border-transparent hover:bg-terminal-panel",
      )}
    >
      <TierIcon tier={strategy.tier} />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1.5">
          <span className="text-xs font-medium text-terminal-text truncate">{strategy.name}</span>
        </div>
        <span className="text-2xs text-terminal-muted font-mono">{strategy.strategy_id}</span>
      </div>
      <div className="flex items-center gap-2 shrink-0">
        <SharpeBadge sharpe={strategy.sharpe} />
        <StatusBadge status={strategy.status} />
        <div
          onClick={(e) => {
            e.stopPropagation();
            onToggle();
          }}
        >
          <Switch.Root
            checked={enabled}
            className={cn(
              "relative inline-flex h-4 w-7 shrink-0 cursor-pointer rounded-full border border-terminal-border transition-colors",
              enabled ? "bg-terminal-accent" : "bg-terminal-panel",
            )}
          >
            <Switch.Thumb
              className={cn(
                "pointer-events-none block h-3 w-3 rounded-full bg-white shadow-sm transition-transform",
                enabled ? "translate-x-3" : "translate-x-0",
              )}
            />
          </Switch.Root>
        </div>
      </div>
    </button>
  );
}

// ---------------------------------------------------------------------------
// Metric card
// ---------------------------------------------------------------------------

function MetricCard({
  label,
  value,
  icon: Icon,
  color,
}: {
  label: string;
  value: string;
  icon: React.ElementType;
  color?: string;
}) {
  return (
    <div className="panel p-3">
      <div className="flex items-center gap-1.5 mb-1">
        <Icon className="h-3.5 w-3.5 text-terminal-muted" />
        <span className="metric-label">{label}</span>
      </div>
      <span className={cn("text-lg font-semibold font-mono tabular-nums", color)}>{value}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Detail panel (right side)
// ---------------------------------------------------------------------------

function StrategyDetailPanel({
  detail,
  localParams,
  onParamChange,
  onSave,
  onReset,
  onRunBacktest,
}: {
  detail: StrategyDetail;
  localParams: Record<string, ParamMeta> | null;
  onParamChange: (key: string, value: number | boolean | string) => void;
  onSave: () => void;
  onReset: () => void;
  onRunBacktest: (strategyId: string) => void;
}) {
  const sharpeColor =
    detail.sharpe >= 1.5
      ? "text-terminal-profit"
      : detail.sharpe >= 0
        ? "text-terminal-warning"
        : "text-terminal-loss";

  const retColor = detail.return_pct >= 0 ? "text-terminal-profit" : "text-terminal-loss";

  return (
    <div className="flex flex-col gap-4 p-4 h-full overflow-y-auto">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-lg font-bold text-terminal-text">{detail.name}</h2>
          <span className="text-xs font-mono text-terminal-muted">{detail.strategy_id}</span>
        </div>
        <div className="flex items-center gap-2">
          <StatusBadge status={detail.status} />
          <span className="text-2xs text-terminal-muted px-1.5 py-0.5 rounded bg-terminal-panel border border-terminal-border">
            {TIER_NAMES[detail.tier] ?? detail.tier}
          </span>
        </div>
      </div>

      {/* Description */}
      {typeof detail.metadata?.description === "string" && (
        <p className="text-xs text-terminal-text-secondary leading-relaxed">
          {detail.metadata.description}
        </p>
      )}

      {/* Metrics grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
        <MetricCard
          label="Sharpe"
          value={detail.sharpe.toFixed(2)}
          icon={BarChart3}
          color={sharpeColor}
        />
        <MetricCard
          label="Return"
          value={`${detail.return_pct >= 0 ? "+" : ""}${detail.return_pct.toFixed(2)}%`}
          icon={TrendingUp}
          color={retColor}
        />
        <MetricCard
          label="Max Drawdown"
          value={`${detail.max_dd.toFixed(2)}%`}
          icon={TrendingDown}
          color="text-terminal-loss"
        />
        <MetricCard
          label="Win Rate"
          value={`${detail.win_rate.toFixed(1)}%`}
          icon={Target}
        />
        <MetricCard
          label="Trades"
          value={String(detail.n_closed)}
          icon={Hash}
        />
        <MetricCard
          label="Best Config"
          value={detail.best_config || "--"}
          icon={Settings2}
        />
      </div>

      {/* Date range + Run Backtest */}
      <div className="flex items-center justify-between">
        {detail.date_range && (
          <div className="flex items-center gap-2 text-xs text-terminal-muted">
            <Calendar className="h-3.5 w-3.5" />
            <span>Backtest period: {detail.date_range}</span>
          </div>
        )}
        <button
          onClick={() => onRunBacktest(detail.strategy_id)}
          className="inline-flex items-center gap-1.5 rounded-md bg-terminal-accent px-3 py-1.5 text-xs font-medium text-white hover:bg-terminal-accent-dim transition-colors"
        >
          <FlaskConical className="h-3.5 w-3.5" />
          Run Backtest
        </button>
      </div>

      {/* Parameter editor */}
      {localParams && Object.keys(localParams).length > 0 && (
        <div className="panel">
          <div className="panel-header">
            <span className="panel-title flex items-center gap-1.5">
              <Layers className="h-3.5 w-3.5" />
              Parameters
            </span>
          </div>
          <div className="panel-body">
            <ParamEditor
              params={localParams}
              onChange={onParamChange}
              onSave={onSave}
              onReset={onReset}
            />
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function StrategiesPage() {
  const [strategies, setStrategies] = useState<StrategySummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<StrategyDetail | null>(null);
  const [params, setParams] = useState<Record<string, ParamMeta> | null>(null);
  const [localParams, setLocalParams] = useState<Record<string, ParamMeta> | null>(null);
  const [enabledMap, setEnabledMap] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState(true);
  const [view, setView] = useState<"strategies" | "compare">("strategies");
  const setPage = useSetAtom(pageAtom);
  const setWorkspace = useSetAtom(activeWorkspaceAtom);

  // Navigate to backtest workspace for a specific strategy
  const handleRunBacktest = useCallback(
    (_strategyId: string) => {
      setWorkspace("backtest");
      setPage("terminal");
    },
    [setWorkspace, setPage],
  );

  // Fetch strategy list
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await apiFetch<StrategiesListResponse>("/api/strategies");
        if (!cancelled) {
          setStrategies(res.strategies);
          setLoading(false);
          // Default all to enabled
          const map: Record<string, boolean> = {};
          for (const s of res.strategies) {
            map[s.strategy_id] = true;
          }
          setEnabledMap(map);
        }
      } catch {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Fetch detail + params when selection changes
  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      setParams(null);
      setLocalParams(null);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const [d, p] = await Promise.all([
          apiFetch<StrategyDetail>(`/api/strategies/${selectedId}`),
          apiFetch<ParamResponse>(`/api/strategies/${selectedId}/params`),
        ]);
        if (!cancelled) {
          setDetail(d);
          setParams(p.params);
          setLocalParams(structuredClone(p.params));
        }
      } catch {
        // strategy may not be in registry — show detail from list
        if (!cancelled) {
          const summary = strategies.find((s) => s.strategy_id === selectedId);
          if (summary) {
            setDetail({
              ...summary,
              date_range: "",
              metadata: {},
            });
          }
          setParams(null);
          setLocalParams(null);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedId, strategies]);

  // Toggle handler
  const handleToggle = useCallback(
    async (strategyId: string) => {
      // Optimistic toggle
      setEnabledMap((prev) => ({ ...prev, [strategyId]: !prev[strategyId] }));
      try {
        const res = await apiFetch<ToggleResponse>(`/api/strategies/${strategyId}/toggle`, {
          method: "POST",
        });
        setEnabledMap((prev) => ({ ...prev, [strategyId]: res.enabled }));
      } catch {
        // Revert on failure
        setEnabledMap((prev) => ({ ...prev, [strategyId]: !prev[strategyId] }));
      }
    },
    [],
  );

  // Param change (local state only until save)
  const handleParamChange = useCallback(
    (key: string, value: number | boolean | string) => {
      setLocalParams((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          [key]: { ...prev[key], value },
        };
      });
    },
    [],
  );

  // Save params
  const handleSave = useCallback(async () => {
    if (!selectedId || !localParams) return;
    const body: Record<string, number | boolean | string> = {};
    for (const [key, meta] of Object.entries(localParams)) {
      body[key] = meta.value;
    }
    try {
      const res = await apiFetch<ParamResponse>(`/api/strategies/${selectedId}/params`, {
        method: "PUT",
        body: JSON.stringify(body),
      });
      setParams(res.params);
      setLocalParams(structuredClone(res.params));
    } catch {
      // keep local state on failure
    }
  }, [selectedId, localParams]);

  // Reset params to server state
  const handleReset = useCallback(() => {
    if (params) {
      setLocalParams(structuredClone(params));
    }
  }, [params]);

  // Group strategies by tier
  const grouped = strategies.reduce<Record<string, StrategySummary[]>>((acc, s) => {
    const tier = s.tier || "tier4";
    if (!acc[tier]) acc[tier] = [];
    acc[tier].push(s);
    return acc;
  }, {});

  const sortedTiers = Object.keys(grouped).sort(
    (a, b) => (TIER_ORDER[a] ?? 99) - (TIER_ORDER[b] ?? 99),
  );

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* View tabs: Strategies | Compare */}
      <div className="flex items-center gap-1 px-4 pt-3 pb-2 border-b border-terminal-border bg-terminal-surface shrink-0">
        <button
          onClick={() => setView("strategies")}
          className={cn(
            "inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors",
            view === "strategies"
              ? "bg-terminal-accent/15 text-terminal-accent"
              : "text-terminal-muted hover:text-terminal-text hover:bg-terminal-panel/60",
          )}
        >
          <Layers className="h-3.5 w-3.5" />
          Strategies
        </button>
        <button
          onClick={() => setView("compare")}
          className={cn(
            "inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors",
            view === "compare"
              ? "bg-terminal-accent/15 text-terminal-accent"
              : "text-terminal-muted hover:text-terminal-text hover:bg-terminal-panel/60",
          )}
        >
          <GitCompare className="h-3.5 w-3.5" />
          Compare Strategies
        </button>
      </div>

      {/* Compare view */}
      {view === "compare" && (
        <div className="flex-1 overflow-hidden">
          <BacktestCompare />
        </div>
      )}

      {/* Strategies view */}
      {view === "strategies" && (
      <div className="flex flex-1 overflow-hidden">
      {/* Left sidebar — strategy list */}
      <div className="w-80 shrink-0 border-r border-terminal-border flex flex-col overflow-hidden bg-terminal-surface">
        <div className="px-3 py-3 border-b border-terminal-border">
          <h1 className="text-sm font-bold uppercase tracking-widest text-terminal-text">
            Strategies
          </h1>
          <p className="text-2xs text-terminal-muted mt-0.5">
            {strategies.length} registered
          </p>
        </div>
        <div className="flex-1 overflow-y-auto p-2 flex flex-col gap-3">
          {loading ? (
            <div className="flex items-center justify-center h-32 text-terminal-muted text-xs font-mono">
              Loading strategies...
            </div>
          ) : strategies.length === 0 ? (
            <div className="flex items-center justify-center h-32 text-terminal-muted text-xs font-mono">
              No strategies found
            </div>
          ) : (
            sortedTiers.map((tier) => (
              <div key={tier}>
                <div className="flex items-center gap-1.5 px-2 mb-1.5">
                  <TierIcon tier={tier} />
                  <span className="text-2xs font-semibold uppercase tracking-wider text-terminal-muted">
                    {TIER_NAMES[tier] ?? tier}
                  </span>
                  <span className="text-2xs text-terminal-muted font-mono">
                    ({grouped[tier].length})
                  </span>
                </div>
                <div className="flex flex-col gap-0.5">
                  {grouped[tier].map((s) => (
                    <StrategyRow
                      key={s.strategy_id}
                      strategy={s}
                      selected={selectedId === s.strategy_id}
                      enabled={enabledMap[s.strategy_id] ?? true}
                      onSelect={() => setSelectedId(s.strategy_id)}
                      onToggle={() => handleToggle(s.strategy_id)}
                    />
                  ))}
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Right panel — detail view */}
      <div className="flex-1 overflow-hidden bg-terminal-bg">
        {detail ? (
          <StrategyDetailPanel
            detail={detail}
            localParams={localParams}
            onParamChange={handleParamChange}
            onSave={handleSave}
            onReset={handleReset}
            onRunBacktest={handleRunBacktest}
          />
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-terminal-muted">
            <Layers className="h-10 w-10 mb-3 opacity-30" />
            <p className="text-sm">Select a strategy to view details</p>
            <p className="text-xs mt-1">Click any strategy in the sidebar</p>
          </div>
        )}
      </div>
      </div>
      )}
    </div>
  );
}
