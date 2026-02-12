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

const DEFAULT_RISK: RiskState = {
  greeks: { delta: 0.42, gamma: 0.018, theta: -1240, vega: 3850 },
  var: { var95: 185000, var99: 312000 },
  drawdown: { currentPct: 1.82, maxPct: 3.47, limitPct: 5.0 },
  exposure: { gross: 4250000, net: 720000, long: 2485000, short: 1765000 },
  circuitBreakers: [
    { name: "Max Drawdown", status: "ok" },
    { name: "Daily Loss Limit", status: "ok" },
    { name: "Position Concentration", status: "ok" },
    { name: "Gross Exposure", status: "ok" },
    { name: "VaR Breach", status: "ok" },
    { name: "Correlation Spike", status: "tripped", reason: "Intra-day corr > 0.85" },
  ],
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
      <span className={cn("font-mono text-sm font-semibold", color ?? "text-gray-100")}>
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
        <MiniCard label="Delta" value={greeks.delta.toFixed(3)} />
        <MiniCard label="Gamma" value={greeks.gamma.toFixed(4)} />
        <MiniCard
          label="Theta"
          value={greeks.theta.toLocaleString("en-IN")}
          color="text-terminal-loss"
        />
        <MiniCard label="Vega" value={greeks.vega.toLocaleString("en-IN")} />
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
          value={`₹${(varData.var95 / 1000).toFixed(0)}K`}
          color="text-terminal-warning"
        />
        <MiniCard
          label="VaR 99%"
          value={`₹${(varData.var99 / 1000).toFixed(0)}K`}
          color="text-terminal-loss"
        />
      </div>
    </section>
  );
}

function DrawdownSection({ dd }: { dd: Drawdown }) {
  const barWidth = Math.min((dd.currentPct / dd.limitPct) * 100, 100);
  const maxBarWidth = Math.min((dd.maxPct / dd.limitPct) * 100, 100);

  return (
    <section>
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-terminal-muted">
        Drawdown
      </h3>
      <div className="grid grid-cols-2 gap-2 mb-3">
        <MiniCard
          label="Current DD"
          value={`${dd.currentPct.toFixed(2)}%`}
          color="text-terminal-loss"
        />
        <MiniCard
          label="Max DD"
          value={`${dd.maxPct.toFixed(2)}%`}
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
        <span>Limit: {dd.limitPct}%</span>
      </div>
    </section>
  );
}

function ExposureSection({ exposure }: { exposure: Exposure }) {
  const fmt = (v: number) => `₹${(v / 100000).toFixed(1)}L`;
  return (
    <section>
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-terminal-muted">
        Exposure
      </h3>
      <div className="grid grid-cols-4 gap-2">
        <MiniCard label="Gross" value={fmt(exposure.gross)} />
        <MiniCard
          label="Net"
          value={fmt(exposure.net)}
          color={exposure.net >= 0 ? "text-terminal-profit" : "text-terminal-loss"}
        />
        <MiniCard label="Long" value={fmt(exposure.long)} color="text-terminal-profit" />
        <MiniCard label="Short" value={fmt(exposure.short)} color="text-terminal-loss" />
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
              <span className="text-gray-200">{cb.name}</span>
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
  const [risk, setRisk] = useState<RiskState>(DEFAULT_RISK);
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
    if (data) setRisk(data);
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
        <h2 className="text-sm font-bold uppercase tracking-widest text-gray-100">
          Risk Dashboard
        </h2>
        <div className="flex items-center gap-2">
          {killConfirm && (
            <button
              onClick={handleKillCancel}
              className="rounded px-3 py-1.5 text-xs font-medium bg-terminal-surface border border-terminal-border text-terminal-muted hover:text-gray-100 transition-colors"
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
