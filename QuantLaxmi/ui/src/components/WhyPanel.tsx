"use client";

import { clsx } from "clsx";
import type {
  TradeDecisionChain,
  WhyEvent,
} from "@/lib/types";
import { formatDateTime, formatPct } from "@/lib/formatters";

// ============================================================
// WHY PANEL — Operator Explainability Side Panel
//
// Shows the full decision chain for any signal/trade:
//   Signal (strategy-specific fields) -> Gate Decision -> Order -> Fill
//
// Data reads directly from WAL (no inference, no recomputation).
// Every field maps 1:1 to Strategy Contract V1 and journal records.
// ============================================================

interface WhyPanelProps {
  isOpen: boolean;
  chain: TradeDecisionChain | undefined;
  isLoading: boolean;
  error: Error | null;
  onClose: () => void;
}

// ---------- Strategy Why Field Labels ----------
const STRATEGY_FIELD_LABELS: Record<string, Record<string, string>> = {
  s1_vrp: {
    composite: "VRP Composite",
    sig_pctile: "Signal Percentile",
    skew_premium: "Skew Premium",
    left_tail: "Left Tail Density",
    atm_iv: "ATM IV",
  },
  s4_iv_mr: {
    atm_iv: "ATM IV",
    iv_pctile: "IV Percentile",
    spot: "Spot Price",
    regime: "Regime",
  },
  s5_hawkes: {
    gex_regime: "GEX Regime",
    components: "Signal Components",
    reasoning: "Reasoning",
    raw_score: "Raw Score",
    smoothed_score: "Smoothed Score",
  },
  s7_regime: {
    sub_strategy: "Sub-Strategy",
    regime: "Regime",
    entropy: "Entropy",
    mi: "Mutual Information",
    z_score: "Z-Score",
    pct_b: "% Bollinger",
    confidence: "Confidence",
    vpin: "VPIN",
  },
  s8_expiry_theta: {
    structure: "Structure",
    short_call: "Short Call",
    long_call: "Long Call",
    short_put: "Short Put",
    long_put: "Long Put",
    spot: "Spot",
    max_profit: "Max Profit",
    dte: "DTE",
    vix: "VIX",
    premium_pct: "Premium %",
  },
  s9_momentum: {
    delivery_z: "Delivery Z",
    oi_price: "OI-Price Conc.",
    rel_strength: "Relative Strength",
    composite_score: "Composite Score",
  },
  s10_gamma_scalp: {
    structure: "Structure",
    iv_pctile: "IV Percentile",
    atm_iv: "ATM IV",
    spot: "Spot",
    dte: "DTE",
    gamma_per_lot: "Gamma/Lot",
    vanna_exposure: "Vanna Exp.",
  },
  s11_pairs: {
    pair: "Pair Symbol",
    hedge_ratio: "Hedge Ratio",
    z_score: "Z-Score",
    half_life: "Half-Life",
    coint_pvalue: "Coint. p-value",
  },
};

// ---------- Main Component ----------

export function WhyPanel({
  isOpen,
  chain,
  isLoading,
  error,
  onClose,
}: WhyPanelProps) {
  return (
    <>
      {/* Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 transition-opacity"
          onClick={onClose}
        />
      )}

      {/* Panel */}
      <div
        className={clsx(
          "fixed top-0 right-0 h-full w-[480px] bg-gray-950 border-l border-gray-800",
          "z-50 transform transition-transform duration-200 ease-out",
          "overflow-y-auto",
          isOpen ? "translate-x-0" : "translate-x-full"
        )}
      >
        {/* Header */}
        <div className="sticky top-0 bg-gray-950/95 backdrop-blur-sm border-b border-gray-800 px-5 py-4 z-10">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-sm font-semibold text-white tracking-wide uppercase">
                Why Panel
              </h2>
              {chain && (
                <p className="text-xs text-gray-500 mt-0.5 font-mono">
                  {chain.strategy_id} / {chain.symbol} / {chain.date}
                </p>
              )}
            </div>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-gray-800 text-gray-500 hover:text-gray-300 transition-colors"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-5 space-y-4">
          {isLoading && <LoadingState />}
          {error && <ErrorState message={error.message} />}
          {!isLoading && !error && !chain && <EmptyState />}
          {chain && <DecisionChain chain={chain} />}
        </div>
      </div>
    </>
  );
}

// ---------- Decision Chain Renderer ----------

function DecisionChain({ chain }: { chain: TradeDecisionChain }) {
  const strategyId = chain.strategy_id;

  return (
    <div className="space-y-4">
      {/* Step 1: Signal(s) */}
      {chain.signals.map((signal, i) => (
        <ChainStep
          key={`sig-${signal.seq}`}
          step={i === 0 ? 1 : undefined}
          label="Signal Generated"
          icon="signal"
          color="accent"
        >
          <SignalSection event={signal} strategyId={strategyId} />
        </ChainStep>
      ))}

      {/* Step 2: Gate Decision(s) */}
      {chain.gates.map((gate, i) => (
        <ChainStep
          key={`gate-${gate.seq}`}
          step={chain.signals.length > 0 ? 2 : undefined}
          label="Risk Gate"
          icon={gate.payload.approved ? "pass" : "block"}
          color={gate.payload.approved ? "profit" : "loss"}
        >
          <GateSection event={gate} />
        </ChainStep>
      ))}

      {/* No gate decisions found */}
      {chain.gates.length === 0 && chain.signals.length > 0 && (
        <ChainStep label="Risk Gate" icon="block" color="gray">
          <p className="text-xs text-gray-600">No gate decision recorded</p>
        </ChainStep>
      )}

      {/* Step 3: Order(s) */}
      {chain.orders.map((order) => (
        <ChainStep
          key={`ord-${order.seq}`}
          step={3}
          label="Order"
          icon="order"
          color="accent"
        >
          <OrderSection event={order} />
        </ChainStep>
      ))}

      {/* Step 4: Fill(s) */}
      {chain.fills.map((fill) => (
        <ChainStep
          key={`fill-${fill.seq}`}
          step={4}
          label="Execution Fill"
          icon="fill"
          color="profit"
        >
          <FillSection event={fill} />
        </ChainStep>
      ))}

      {/* Risk Alerts */}
      {chain.risk_alerts.map((alert) => (
        <ChainStep
          key={`alert-${alert.seq}`}
          label="Risk Alert"
          icon="block"
          color="loss"
        >
          <RiskAlertSection event={alert} />
        </ChainStep>
      ))}

      {/* Portfolio Snapshot */}
      {chain.snapshot && (
        <ChainStep label="Portfolio State" icon="snapshot" color="gray">
          <SnapshotSection event={chain.snapshot} />
        </ChainStep>
      )}
    </div>
  );
}

// ---------- Chain Step Wrapper ----------

function ChainStep({
  step,
  label,
  icon,
  color,
  children,
}: {
  step?: number;
  label: string;
  icon: string;
  color: string;
  children: React.ReactNode;
}) {
  const iconMap: Record<string, React.ReactNode> = {
    signal: (
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9.348 14.651a3.75 3.75 0 010-5.303m5.304 0a3.75 3.75 0 010 5.303m-7.425 2.122a6.75 6.75 0 010-9.546m9.546 0a6.75 6.75 0 010 9.546M5.106 18.894c-3.808-3.808-3.808-9.98 0-13.789m13.788 0c3.808 3.808 3.808 9.981 0 13.79" />
      </svg>
    ),
    pass: (
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
    block: (
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
      </svg>
    ),
    order: (
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
      </svg>
    ),
    fill: (
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
      </svg>
    ),
    snapshot: (
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M6.827 6.175A2.31 2.31 0 015.186 7.23c-.38.054-.757.112-1.134.175C2.999 7.58 2.25 8.507 2.25 9.574V18a2.25 2.25 0 002.25 2.25h15A2.25 2.25 0 0021.75 18V9.574c0-1.067-.75-1.994-1.802-2.169a47.865 47.865 0 00-1.134-.175 2.31 2.31 0 01-1.64-1.055l-.822-1.316a2.192 2.192 0 00-1.736-1.039 48.774 48.774 0 00-5.232 0 2.192 2.192 0 00-1.736 1.039l-.821 1.316z" />
      </svg>
    ),
  };

  const colorMap: Record<string, string> = {
    accent: "border-accent/30 text-accent",
    profit: "border-profit/30 text-profit",
    loss: "border-loss/30 text-loss",
    gray: "border-gray-700 text-gray-500",
  };

  return (
    <div className={clsx("border rounded-lg bg-gray-900/50", colorMap[color] || colorMap.gray)}>
      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-gray-800/50">
        <span className={colorMap[color]}>{iconMap[icon] || iconMap.signal}</span>
        <span className="text-xs font-medium uppercase tracking-wider text-gray-400">
          {step && <span className="text-gray-600 mr-1">#{step}</span>}
          {label}
        </span>
      </div>
      <div className="px-4 py-3">{children}</div>
    </div>
  );
}

// ---------- Signal Section ----------

function SignalSection({
  event,
  strategyId,
}: {
  event: WhyEvent;
  strategyId: string;
}) {
  const p = event.payload;
  const direction = String(p.direction || "");
  const conviction = Number(p.conviction || 0);
  const regime = String(p.regime || "");
  const components = (p.components || {}) as Record<string, unknown>;
  const reasoning = String(p.reasoning || "");

  const dirColor =
    direction === "long"
      ? "text-profit"
      : direction === "short"
        ? "text-loss"
        : "text-gray-400";

  return (
    <div className="space-y-3">
      {/* Core signal fields */}
      <div className="grid grid-cols-3 gap-2">
        <FieldItem label="Direction" value={direction.toUpperCase()} className={dirColor} />
        <FieldItem label="Conviction" value={conviction.toFixed(3)} mono />
        <FieldItem label="Regime" value={regime || "--"} />
      </div>
      <div className="grid grid-cols-3 gap-2">
        <FieldItem label="Instrument" value={String(p.instrument_type || "FUT")} />
        <FieldItem label="Strike" value={Number(p.strike || 0) > 0 ? String(p.strike) : "--"} mono />
        <FieldItem label="TTL Bars" value={String(p.ttl_bars || "--")} mono />
      </div>

      {/* Strategy-specific components (Why Fields) */}
      {Object.keys(components).length > 0 && (
        <div className="pt-2 border-t border-gray-800/50">
          <p className="text-[10px] text-gray-600 uppercase tracking-wider mb-2">
            Strategy Why Fields
          </p>
          <StrategyComponents
            strategyId={strategyId}
            components={components}
          />
        </div>
      )}

      {/* Reasoning (if present) */}
      {reasoning && (
        <div className="pt-2 border-t border-gray-800/50">
          <p className="text-[10px] text-gray-600 uppercase tracking-wider mb-1">
            Reasoning
          </p>
          <p className="text-xs text-gray-300 leading-relaxed">{reasoning}</p>
        </div>
      )}

      <p className="text-[10px] text-gray-700 font-mono">
        seq={event.seq} ts={formatDateTime(event.ts)}
      </p>
    </div>
  );
}

// ---------- Strategy-specific Components ----------

function StrategyComponents({
  strategyId,
  components,
}: {
  strategyId: string;
  components: Record<string, unknown>;
}) {
  const labels = STRATEGY_FIELD_LABELS[strategyId] || {};

  // Flatten nested "components" if S5 Hawkes returns {components: {...}}
  const flatComponents: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(components)) {
    if (key === "components" && typeof value === "object" && value !== null) {
      // Flatten nested components (S5 Hawkes)
      for (const [k2, v2] of Object.entries(value as Record<string, unknown>)) {
        flatComponents[k2] = v2;
      }
    } else {
      flatComponents[key] = value;
    }
  }

  const entries = Object.entries(flatComponents);
  if (entries.length === 0) return null;

  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
      {entries.map(([key, value]) => {
        const label = labels[key] || key.replace(/_/g, " ");
        const display = formatComponentValue(value);
        return (
          <div key={key} className="flex items-baseline justify-between">
            <span className="text-[10px] text-gray-500 capitalize truncate mr-2">
              {label}
            </span>
            <span className="text-xs font-mono text-gray-300 shrink-0">
              {display}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function formatComponentValue(value: unknown): string {
  if (value === null || value === undefined) return "--";
  if (typeof value === "number") {
    if (Number.isInteger(value)) return String(value);
    return value.toFixed(4);
  }
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "string") return value || "--";
  if (Array.isArray(value)) return `[${value.length}]`;
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

// ---------- Gate Decision Section ----------

function GateSection({ event }: { event: WhyEvent }) {
  const p = event.payload;
  const approved = Boolean(p.approved);
  const gate = String(p.gate || "");
  const reason = String(p.reason || "");

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <span
          className={clsx(
            "inline-flex items-center px-2 py-0.5 rounded text-xs font-medium border",
            approved
              ? "bg-profit/10 text-profit border-profit/20"
              : "bg-loss/10 text-loss border-loss/20"
          )}
        >
          {approved ? "APPROVED" : "BLOCKED"}
        </span>
        <span className="text-xs text-gray-500 font-mono">{gate}</span>
      </div>

      {reason && (
        <p className="text-xs text-gray-400">{reason}</p>
      )}

      <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 pt-1">
        <FieldItem label="Weight" value={Number(p.adjusted_weight || 0).toFixed(4)} mono />
        <FieldItem label="VPIN" value={Number(p.vpin || 0).toFixed(4)} mono />
        <FieldItem label="Portfolio DD" value={formatPct(Number(p.portfolio_dd || 0) * 100)} mono />
        <FieldItem label="Strategy DD" value={formatPct(Number(p.strategy_dd || 0) * 100)} mono />
        <FieldItem label="Total Exposure" value={Number(p.total_exposure || 0).toFixed(4)} mono />
      </div>

      <p className="text-[10px] text-gray-700 font-mono">
        seq={event.seq} ts={formatDateTime(event.ts)}
      </p>
    </div>
  );
}

// ---------- Order Section ----------

function OrderSection({ event }: { event: WhyEvent }) {
  const p = event.payload;
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-3 gap-2">
        <FieldItem label="Action" value={String(p.action || "")} />
        <FieldItem label="Side" value={String(p.side || "").toUpperCase()} />
        <FieldItem label="Type" value={String(p.order_type || "")} />
      </div>
      {Number(p.quantity || 0) > 0 && (
        <div className="grid grid-cols-2 gap-2">
          <FieldItem label="Qty" value={String(p.quantity)} mono />
          <FieldItem label="Price" value={Number(p.price || 0).toFixed(2)} mono />
        </div>
      )}
      <p className="text-[10px] text-gray-700 font-mono">
        order_id={String(p.order_id || "--")} seq={event.seq}
      </p>
    </div>
  );
}

// ---------- Fill Section ----------

function FillSection({ event }: { event: WhyEvent }) {
  const p = event.payload;
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-3 gap-2">
        <FieldItem label="Side" value={String(p.side || "").toUpperCase()} />
        <FieldItem label="Qty" value={String(p.quantity || 0)} mono />
        <FieldItem label="Price" value={Number(p.price || 0).toFixed(2)} mono />
      </div>
      <div className="grid grid-cols-2 gap-2">
        <FieldItem label="Fees" value={Number(p.fees || 0).toFixed(2)} mono />
        <FieldItem label="Partial" value={p.is_partial ? "Yes" : "No"} />
      </div>
      <p className="text-[10px] text-gray-700 font-mono">
        fill_id={String(p.fill_id || "--")} seq={event.seq}
      </p>
    </div>
  );
}

// ---------- Risk Alert Section ----------

function RiskAlertSection({ event }: { event: WhyEvent }) {
  const p = event.payload;
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-2 gap-2">
        <FieldItem label="Alert Type" value={String(p.alert_type || "")} />
        <FieldItem label="State" value={String(p.new_state || "")} />
      </div>
      <div className="grid grid-cols-2 gap-2">
        <FieldItem label="Threshold" value={Number(p.threshold || 0).toFixed(4)} mono />
        <FieldItem label="Current" value={Number(p.current_value || 0).toFixed(4)} mono />
      </div>
      {String(p.detail || "") && (
        <p className="text-xs text-gray-400">{String(p.detail)}</p>
      )}
    </div>
  );
}

// ---------- Snapshot Section ----------

function SnapshotSection({ event }: { event: WhyEvent }) {
  const p = event.payload;
  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
      <FieldItem label="Equity" value={Number(p.equity || 0).toFixed(4)} mono />
      <FieldItem label="Peak" value={Number(p.peak_equity || 0).toFixed(4)} mono />
      <FieldItem label="Portfolio DD" value={formatPct(Number(p.portfolio_dd || 0) * 100)} mono />
      <FieldItem label="Exposure" value={Number(p.total_exposure || 0).toFixed(4)} mono />
      <FieldItem label="VPIN" value={Number(p.vpin || 0).toFixed(4)} mono />
      <FieldItem label="Positions" value={String(p.position_count || 0)} mono />
      <FieldItem label="Regime" value={String(p.regime || "--")} />
    </div>
  );
}

// ---------- Shared Field Item ----------

function FieldItem({
  label,
  value,
  mono = false,
  className = "",
}: {
  label: string;
  value: string;
  mono?: boolean;
  className?: string;
}) {
  return (
    <div>
      <p className="text-[10px] text-gray-600 uppercase tracking-wider">{label}</p>
      <p className={clsx("text-xs", mono ? "font-mono" : "", className || "text-gray-300")}>
        {value}
      </p>
    </div>
  );
}

// ---------- States ----------

function LoadingState() {
  return (
    <div className="flex flex-col items-center justify-center py-16 space-y-3">
      <div className="w-6 h-6 border-2 border-accent border-t-transparent rounded-full animate-spin" />
      <p className="text-xs text-gray-500">Loading decision chain from WAL...</p>
    </div>
  );
}

function ErrorState({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 space-y-2">
      <svg className="w-8 h-8 text-loss" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
      </svg>
      <p className="text-xs text-gray-500 text-center max-w-xs">{message}</p>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-16 space-y-2">
      <svg className="w-8 h-8 text-gray-700" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9 5.25h.008v.008H12v-.008z" />
      </svg>
      <p className="text-xs text-gray-600 text-center">
        Click a signal or trade to see its decision chain
      </p>
    </div>
  );
}
