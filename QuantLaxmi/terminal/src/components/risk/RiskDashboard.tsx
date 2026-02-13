import { useCallback, useEffect, useState } from "react";
import { useTauriCommand } from "@/hooks/useTauriCommand";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Greeks {
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
}

interface VaR {
  var95: number;
  var99: number;
}

interface Drawdown {
  currentPct: number;
  maxPct: number;
  limitPct: number;
}

interface Exposure {
  gross: number;
  net: number;
  long: number;
  short: number;
}

interface CircuitBreaker {
  name: string;
  status: "ok" | "tripped";
  reason?: string;
}

interface RiskState {
  greeks: Greeks;
  var: VaR;
  drawdown: Drawdown;
  exposure: Exposure;
  circuitBreakers: CircuitBreaker[];
  killSwitchActive: boolean;
}

const EMPTY_RISK: RiskState = {
  greeks: { delta: 0, gamma: 0, theta: 0, vega: 0 },
  var: { var95: 0, var99: 0 },
  drawdown: { currentPct: 0, maxPct: 0, limitPct: 5.0 },
  exposure: { gross: 0, net: 0, long: 0, short: 0 },
  circuitBreakers: [],
  killSwitchActive: false,
};

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function MiniCard({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="flex flex-col gap-0.5 rounded-md bg-terminal-surface px-3 py-2 border border-terminal-border">
      <span className="text-2xs font-medium uppercase tracking-wider text-terminal-muted">
        {label}
      </span>
      <span className={cn("font-mono text-sm font-semibold", color ?? "text-terminal-text")}>
        {value}
      </span>
    </div>
  );
}

function GreeksRow({ greeks }: { greeks: Greeks }) {
  return (
    <section>
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-terminal-muted">
        Portfolio Greeks
      </h3>
      <div className="grid grid-cols-4 gap-2">
        <MiniCard label="Delta" value={(greeks?.delta ?? 0).toFixed(3)} />
        <MiniCard label="Gamma" value={(greeks?.gamma ?? 0).toFixed(4)} />
        <MiniCard
          label="Theta"
          value={(greeks?.theta ?? 0).toLocaleString("en-IN")}
          color="text-terminal-loss"
        />
        <MiniCard label="Vega" value={(greeks?.vega ?? 0).toLocaleString("en-IN")} />
      </div>
    </section>
  );
}

function VaRSection({ varData }: { varData: VaR }) {
  return (
    <section>
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-terminal-muted">
        Value at Risk
      </h3>
      <div className="grid grid-cols-2 gap-2">
        <MiniCard
          label="VaR 95%"
          value={`₹${((varData?.var95 ?? 0) / 1000).toFixed(0)}K`}
          color="text-terminal-warning"
        />
        <MiniCard
          label="VaR 99%"
          value={`₹${((varData?.var99 ?? 0) / 1000).toFixed(0)}K`}
          color="text-terminal-loss"
        />
      </div>
    </section>
  );
}

function DrawdownSection({ dd }: { dd: Drawdown }) {
  const cur = dd?.currentPct ?? 0;
  const mx = dd?.maxPct ?? 0;
  const lim = dd?.limitPct ?? 5;
  const barWidth = Math.min((cur / lim) * 100, 100);
  const maxBarWidth = Math.min((mx / lim) * 100, 100);

  return (
    <section>
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-terminal-muted">
        Drawdown
      </h3>
      <div className="grid grid-cols-2 gap-2 mb-3">
        <MiniCard
          label="Current DD"
          value={`${cur.toFixed(2)}%`}
          color="text-terminal-loss"
        />
        <MiniCard
          label="Max DD"
          value={`${mx.toFixed(2)}%`}
          color="text-terminal-loss"
        />
      </div>
      <div className="relative h-4 rounded-full bg-terminal-surface border border-terminal-border overflow-hidden">
        {/* Max DD marker */}
        <div
          className="absolute top-0 h-full border-r-2 border-terminal-warning opacity-60"
          style={{ left: `${maxBarWidth}%` }}
        />
        {/* Current DD bar */}
        <div
          className="h-full rounded-full bg-gradient-to-r from-terminal-loss/40 to-terminal-loss transition-all duration-500"
          style={{ width: `${barWidth}%` }}
        />
        {/* Limit line */}
        <div
          className="absolute top-0 h-full border-r-2 border-dashed border-red-500"
          style={{ left: "100%" }}
        />
      </div>
      <div className="flex justify-between mt-1 text-2xs text-terminal-muted font-mono">
        <span>0%</span>
        <span>Limit: {lim}%</span>
      </div>
    </section>
  );
}

function ExposureSection({ exposure }: { exposure: Exposure }) {
  const fmt = (v: number | undefined) => `₹${((v ?? 0) / 100000).toFixed(1)}L`;
  const net = exposure?.net ?? 0;
  return (
    <section>
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-terminal-muted">
        Exposure
      </h3>
      <div className="grid grid-cols-4 gap-2">
        <MiniCard label="Gross" value={fmt(exposure?.gross)} />
        <MiniCard
          label="Net"
          value={fmt(net)}
          color={net >= 0 ? "text-terminal-profit" : "text-terminal-loss"}
        />
        <MiniCard label="Long" value={fmt(exposure?.long)} color="text-terminal-profit" />
        <MiniCard label="Short" value={fmt(exposure?.short)} color="text-terminal-loss" />
      </div>
    </section>
  );
}

function CircuitBreakerList({ breakers }: { breakers: CircuitBreaker[] }) {
  return (
    <section>
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-terminal-muted">
        Circuit Breakers
      </h3>
      <div className="space-y-1">
        {breakers.map((cb) => (
          <div
            key={cb.name}
            className={cn(
              "flex items-center justify-between rounded px-3 py-1.5 text-xs border",
              cb.status === "ok"
                ? "border-terminal-border bg-terminal-surface"
                : "border-terminal-loss/40 bg-terminal-loss/10",
            )}
          >
            <div className="flex items-center gap-2">
              <span
                className={cn(
                  "h-2 w-2 rounded-full",
                  cb.status === "ok" ? "bg-terminal-profit" : "bg-terminal-loss animate-pulse",
                )}
              />
              <span className="text-terminal-text-secondary">{cb.name}</span>
            </div>
            {cb.status === "tripped" && (
              <span className="text-terminal-loss text-2xs">{cb.reason}</span>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function RiskDashboard() {
  const { data, execute } = useTauriCommand<RiskState>("get_risk_state");
  const [risk, setRisk] = useState<RiskState>(EMPTY_RISK);
  const [killConfirm, setKillConfirm] = useState(false);

  useEffect(() => {
    execute().catch(() => {
      /* use defaults on error */
    });
    const interval = setInterval(() => {
      execute().catch(() => {});
    }, 2000);
    return () => clearInterval(interval);
  }, [execute]);

  useEffect(() => {
    if (!data) return;
    // Map snake_case API response to camelCase frontend types
    const raw = data as any;
    try {
      const mapped: RiskState = {
        greeks: {
          delta: raw.greeks?.net_delta ?? raw.greeks?.delta ?? 0,
          gamma: raw.greeks?.net_gamma ?? raw.greeks?.gamma ?? 0,
          theta: raw.greeks?.net_theta ?? raw.greeks?.theta ?? 0,
          vega: raw.greeks?.net_vega ?? raw.greeks?.vega ?? 0,
        },
        var: {
          var95: raw.var?.var_95 ?? raw.var?.var95 ?? 0,
          var99: raw.var?.var_99 ?? raw.var?.var99 ?? 0,
        },
        drawdown: {
          currentPct: raw.portfolio_drawdown_pct ?? raw.drawdown?.currentPct ?? 0,
          maxPct: raw.drawdown?.maxPct ?? raw.drawdown?.max_pct ?? 0,
          limitPct: raw.drawdown?.limitPct ?? raw.drawdown?.limit_pct ?? 5.0,
        },
        exposure: {
          gross: raw.concentration?.gross_exposure ?? raw.exposure?.gross ?? 0,
          net: raw.concentration?.net_exposure ?? raw.exposure?.net ?? 0,
          long: raw.concentration?.long_exposure ?? raw.exposure?.long ?? 0,
          short: raw.concentration?.short_exposure ?? raw.exposure?.short ?? 0,
        },
        circuitBreakers: Array.isArray(raw.circuitBreakers ?? raw.circuit_breakers)
          ? (raw.circuitBreakers ?? raw.circuit_breakers)
          : [],
        killSwitchActive: raw.circuit_breaker_active ?? raw.killSwitchActive ?? false,
      };
      setRisk(mapped);
    } catch {
      // Keep defaults on mapping failure
    }
  }, [data]);

  const handleKillSwitch = useCallback(() => {
    if (!killConfirm) {
      setKillConfirm(true);
      return;
    }
    setRisk((prev) => ({ ...prev, killSwitchActive: !prev.killSwitchActive }));
    setKillConfirm(false);
  }, [killConfirm]);

  const handleKillCancel = useCallback(() => {
    setKillConfirm(false);
  }, []);

  return (
    <div className="flex flex-col gap-4 p-4 h-full overflow-y-auto scrollbar-thin">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-bold uppercase tracking-widest text-terminal-text">
          Risk Dashboard
        </h2>
        <div className="flex items-center gap-2">
          {killConfirm && (
            <button
              onClick={handleKillCancel}
              className="rounded px-3 py-1.5 text-xs font-medium bg-terminal-surface border border-terminal-border text-terminal-muted hover:text-terminal-text transition-colors"
            >
              Cancel
            </button>
          )}
          <button
            onClick={handleKillSwitch}
            className={cn(
              "rounded-md px-4 py-2 text-xs font-bold uppercase tracking-wider transition-all",
              risk.killSwitchActive
                ? "bg-terminal-loss text-white shadow-lg shadow-terminal-loss/30"
                : killConfirm
                  ? "bg-terminal-loss/80 text-white animate-pulse"
                  : "bg-terminal-loss/20 text-terminal-loss border border-terminal-loss/40 hover:bg-terminal-loss/30",
            )}
          >
            {risk.killSwitchActive
              ? "KILL ACTIVE — Click to Resume"
              : killConfirm
                ? "CONFIRM KILL ALL?"
                : "KILL SWITCH"}
          </button>
        </div>
      </div>

      <GreeksRow greeks={risk.greeks} />
      <VaRSection varData={risk.var} />
      <DrawdownSection dd={risk.drawdown} />
      <ExposureSection exposure={risk.exposure} />
      <CircuitBreakerList breakers={risk.circuitBreakers} />
    </div>
  );
}
