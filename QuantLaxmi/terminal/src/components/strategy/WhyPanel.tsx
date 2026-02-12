import { useCallback, useEffect } from "react";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface GateDecision {
  gate: string;
  passed: boolean;
  reason: string;
}

interface RiskSnapshot {
  drawdownPct: number;
  grossExposure: number;
  var95: number;
  positionCount: number;
}

interface TradeChainEvent {
  type: "signal" | "gate" | "order" | "fill";
  timestamp: string;
  label: string;
  detail: string;
}

export interface WhyPanelData {
  signalId: string;
  strategy: string;
  symbol: string;
  direction: "BUY" | "SELL";
  conviction: number;
  regime: string;
  timestamp: string;
  gates: GateDecision[];
  riskSnapshot: RiskSnapshot;
  tradeChain: TradeChainEvent[];
}

interface WhyPanelProps {
  data: WhyPanelData | null;
  open: boolean;
  onClose: () => void;
}

// ---------------------------------------------------------------------------
// Demo data
// ---------------------------------------------------------------------------

export const DEMO_WHY: WhyPanelData = {
  signalId: "sig-abc-123",
  strategy: "S25 DFF",
  symbol: "NIFTY",
  direction: "BUY",
  conviction: 0.82,
  regime: "trending",
  timestamp: "2026-02-12 14:30:15",
  gates: [
    { gate: "DrawdownGate", passed: true, reason: "DD 1.82% < limit 5%" },
    { gate: "ExposureGate", passed: true, reason: "Gross 42.5L < limit 100L" },
    { gate: "VaRGate", passed: true, reason: "VaR95 1.85L < limit 5L" },
    { gate: "ConvictionGate", passed: true, reason: "0.82 > threshold 0.50" },
    { gate: "CorrelationGate", passed: false, reason: "Intra-day corr 0.87 > limit 0.85" },
    { gate: "CircuitBreakerGate", passed: true, reason: "No breakers tripped" },
  ],
  riskSnapshot: {
    drawdownPct: 1.82,
    grossExposure: 4250000,
    var95: 185000,
    positionCount: 3,
  },
  tradeChain: [
    { type: "signal", timestamp: "14:30:15.042", label: "Signal Generated", detail: "DFF composite Z=1.34, direction=BUY, conviction=0.82" },
    { type: "gate", timestamp: "14:30:15.044", label: "Gate Evaluation", detail: "5/6 gates passed, CorrelationGate FAILED" },
    { type: "order", timestamp: "14:30:15.051", label: "Order Submitted", detail: "MARKET BUY NIFTY 50 lots (gate override: manual)" },
    { type: "fill", timestamp: "14:30:15.187", label: "Fill Received", detail: "Filled 50 @ 23,485.20, slippage 0.8 pts" },
  ],
};

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function SignalHeader({ data }: { data: WhyPanelData }) {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <span
          className={cn(
            "rounded px-2 py-0.5 text-xs font-bold",
            data.direction === "BUY"
              ? "bg-terminal-profit/20 text-terminal-profit"
              : "bg-terminal-loss/20 text-terminal-loss",
          )}
        >
          {data.direction}
        </span>
        <span className="font-mono text-sm font-semibold text-gray-100">{data.symbol}</span>
        <span className="text-xs text-terminal-accent">{data.strategy}</span>
      </div>
      <div className="grid grid-cols-3 gap-2">
        <div className="rounded bg-terminal-surface border border-terminal-border px-2 py-1.5">
          <div className="text-2xs text-terminal-muted">Conviction</div>
          <div className="font-mono text-sm text-gray-100">{(data.conviction * 100).toFixed(0)}%</div>
        </div>
        <div className="rounded bg-terminal-surface border border-terminal-border px-2 py-1.5">
          <div className="text-2xs text-terminal-muted">Regime</div>
          <div className="font-mono text-sm text-gray-100">{data.regime}</div>
        </div>
        <div className="rounded bg-terminal-surface border border-terminal-border px-2 py-1.5">
          <div className="text-2xs text-terminal-muted">Time</div>
          <div className="font-mono text-sm text-gray-100">{data.timestamp.split(" ")[1]}</div>
        </div>
      </div>
    </div>
  );
}

function GateDecisions({ gates }: { gates: GateDecision[] }) {
  return (
    <section>
      <h4 className="text-xs font-semibold uppercase tracking-wider text-terminal-muted mb-2">
        Gate Decisions
      </h4>
      <div className="space-y-1">
        {gates.map((g) => (
          <div
            key={g.gate}
            className={cn(
              "flex items-center gap-2 rounded px-3 py-1.5 text-xs border",
              g.passed
                ? "border-terminal-border bg-terminal-surface"
                : "border-terminal-loss/30 bg-terminal-loss/5",
            )}
          >
            <span
              className={cn(
                "flex-shrink-0 h-4 w-4 rounded-full flex items-center justify-center text-2xs font-bold",
                g.passed
                  ? "bg-terminal-profit/20 text-terminal-profit"
                  : "bg-terminal-loss/20 text-terminal-loss",
              )}
            >
              {g.passed ? "\u2713" : "\u2717"}
            </span>
            <span className="font-medium text-gray-200 min-w-[120px]">{g.gate}</span>
            <span className="text-terminal-muted flex-1 truncate">{g.reason}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function RiskAtTime({ snapshot }: { snapshot: RiskSnapshot }) {
  return (
    <section>
      <h4 className="text-xs font-semibold uppercase tracking-wider text-terminal-muted mb-2">
        Risk State at Signal Time
      </h4>
      <div className="grid grid-cols-2 gap-2">
        <div className="rounded bg-terminal-surface border border-terminal-border px-2 py-1.5">
          <div className="text-2xs text-terminal-muted">Drawdown</div>
          <div className="font-mono text-xs text-terminal-loss">{snapshot.drawdownPct.toFixed(2)}%</div>
        </div>
        <div className="rounded bg-terminal-surface border border-terminal-border px-2 py-1.5">
          <div className="text-2xs text-terminal-muted">Gross Exposure</div>
          <div className="font-mono text-xs text-gray-100">{`₹${(snapshot.grossExposure / 100000).toFixed(1)}L`}</div>
        </div>
        <div className="rounded bg-terminal-surface border border-terminal-border px-2 py-1.5">
          <div className="text-2xs text-terminal-muted">VaR 95%</div>
          <div className="font-mono text-xs text-terminal-warning">{`₹${(snapshot.var95 / 1000).toFixed(0)}K`}</div>
        </div>
        <div className="rounded bg-terminal-surface border border-terminal-border px-2 py-1.5">
          <div className="text-2xs text-terminal-muted">Positions</div>
          <div className="font-mono text-xs text-gray-100">{snapshot.positionCount}</div>
        </div>
      </div>
    </section>
  );
}

function TradeChainTimeline({ events }: { events: TradeChainEvent[] }) {
  const typeConfig: Record<string, { color: string; bg: string }> = {
    signal: { color: "text-terminal-accent", bg: "bg-terminal-accent" },
    gate: { color: "text-terminal-warning", bg: "bg-terminal-warning" },
    order: { color: "text-terminal-info", bg: "bg-terminal-info" },
    fill: { color: "text-terminal-profit", bg: "bg-terminal-profit" },
  };

  return (
    <section>
      <h4 className="text-xs font-semibold uppercase tracking-wider text-terminal-muted mb-2">
        Trade Chain
      </h4>
      <div className="relative space-y-0 pl-4">
        {/* Vertical line */}
        <div className="absolute left-[7px] top-2 bottom-2 w-px bg-terminal-border" />

        {events.map((ev, i) => {
          const cfg = typeConfig[ev.type] ?? typeConfig.signal;
          return (
            <div key={i} className="relative flex gap-3 py-2">
              {/* Dot */}
              <div className={cn("absolute left-[-13px] top-3 h-2.5 w-2.5 rounded-full border-2 border-terminal-bg", cfg.bg)} />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-2xs text-terminal-muted">{ev.timestamp}</span>
                  <span className={cn("text-xs font-semibold", cfg.color)}>{ev.label}</span>
                </div>
                <p className="text-xs text-gray-300 mt-0.5 leading-relaxed">{ev.detail}</p>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function WhyPanel({ data, open, onClose }: WhyPanelProps) {
  // Close on Escape
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    },
    [onClose],
  );

  useEffect(() => {
    if (open) {
      window.addEventListener("keydown", handleKeyDown);
      return () => window.removeEventListener("keydown", handleKeyDown);
    }
  }, [open, handleKeyDown]);

  const panelData = data ?? DEMO_WHY;

  return (
    <>
      {/* Backdrop */}
      {open && (
        <div
          className="fixed inset-0 z-40 bg-black/40 transition-opacity"
          onClick={onClose}
        />
      )}

      {/* Sliding panel */}
      <div
        className={cn(
          "fixed top-0 right-0 z-50 h-full w-[420px] max-w-[90vw] bg-terminal-panel border-l border-terminal-border shadow-2xl transition-transform duration-300",
          open ? "translate-x-0" : "translate-x-full",
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-terminal-border">
          <h3 className="text-xs font-bold uppercase tracking-widest text-gray-100">
            Signal Explanation
          </h3>
          <button
            onClick={onClose}
            className="rounded p-1 hover:bg-terminal-surface text-terminal-muted hover:text-gray-100 transition-colors"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M12 4L4 12M4 4l8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex flex-col gap-5 p-4 h-[calc(100%-48px)] overflow-y-auto">
          <SignalHeader data={panelData} />
          <GateDecisions gates={panelData.gates} />
          <RiskAtTime snapshot={panelData.riskSnapshot} />
          <TradeChainTimeline events={panelData.tradeChain} />

          {/* Signal ID footer */}
          <div className="mt-auto pt-4 border-t border-terminal-border">
            <span className="text-2xs text-terminal-muted font-mono">
              ID: {panelData.signalId}
            </span>
          </div>
        </div>
      </div>
    </>
  );
}
