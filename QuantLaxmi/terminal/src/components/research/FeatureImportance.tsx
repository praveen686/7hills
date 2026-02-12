import { useMemo, useState } from "react";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Feature {
  name: string;
  importance: number; // 0-1 normalized
  ic: number; // information coefficient
  group: string;
}

interface FeatureImportanceProps {
  features?: Feature[];
  topN?: number;
}

// ---------------------------------------------------------------------------
// Group colors
// ---------------------------------------------------------------------------

const GROUP_COLORS: Record<string, string> = {
  technical: "#4f8eff",
  volatility: "#ff4d6a",
  volume: "#00d4aa",
  momentum: "#ffb84d",
  divergence: "#a78bfa",
  crypto: "#f97316",
  news: "#06b6d4",
  macro: "#ec4899",
  options: "#8b5cf6",
  structure: "#14b8a6",
};

function groupColor(group: string): string {
  return GROUP_COLORS[group.toLowerCase()] ?? "#6b6b8a";
}

// ---------------------------------------------------------------------------
// Demo data
// ---------------------------------------------------------------------------

const DEMO_FEATURES: Feature[] = [
  { name: "dff_composite", importance: 1.0, ic: 0.082, group: "divergence" },
  { name: "dff_energy_z", importance: 0.92, ic: 0.075, group: "divergence" },
  { name: "iv_zscore_20d", importance: 0.88, ic: 0.071, group: "volatility" },
  { name: "ret_1d", importance: 0.85, ic: 0.068, group: "technical" },
  { name: "ns_sent_mean", importance: 0.82, ic: 0.065, group: "news" },
  { name: "vpin", importance: 0.79, ic: 0.062, group: "volume" },
  { name: "dff_r_hat", importance: 0.77, ic: 0.060, group: "divergence" },
  { name: "rsi_14", importance: 0.74, ic: 0.058, group: "momentum" },
  { name: "iv_skew", importance: 0.72, ic: 0.055, group: "options" },
  { name: "atr_pct", importance: 0.70, ic: 0.053, group: "volatility" },
  { name: "macd_hist", importance: 0.68, ic: 0.051, group: "momentum" },
  { name: "fii_net_oi", importance: 0.65, ic: 0.048, group: "structure" },
  { name: "btc_ret_1h", importance: 0.63, ic: 0.046, group: "crypto" },
  { name: "vol_regime", importance: 0.60, ic: 0.044, group: "volatility" },
  { name: "dff_d_hat", importance: 0.58, ic: 0.042, group: "divergence" },
  { name: "obv_slope", importance: 0.55, ic: 0.040, group: "volume" },
  { name: "ns_pos_ratio", importance: 0.53, ic: 0.038, group: "news" },
  { name: "stochastic_k", importance: 0.50, ic: 0.036, group: "momentum" },
  { name: "pcr_oi", importance: 0.48, ic: 0.034, group: "options" },
  { name: "dii_net_value", importance: 0.46, ic: 0.032, group: "structure" },
  { name: "bbwidth", importance: 0.44, ic: 0.030, group: "volatility" },
  { name: "eth_funding", importance: 0.42, ic: 0.028, group: "crypto" },
  { name: "adx_14", importance: 0.40, ic: 0.026, group: "momentum" },
  { name: "ret_5d", importance: 0.38, ic: 0.024, group: "technical" },
  { name: "dff_momentum", importance: 0.36, ic: 0.022, group: "divergence" },
  { name: "vwap_dev", importance: 0.34, ic: 0.020, group: "volume" },
  { name: "term_structure", importance: 0.32, ic: 0.018, group: "options" },
  { name: "gdp_surprise", importance: 0.30, ic: 0.016, group: "macro" },
  { name: "inr_usd_ret", importance: 0.28, ic: 0.014, group: "macro" },
  { name: "hawkes_intensity", importance: 0.26, ic: 0.012, group: "structure" },
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function FeatureImportance({ features, topN = 30 }: FeatureImportanceProps) {
  const data = features ?? DEMO_FEATURES;
  const [hoveredGroup, setHoveredGroup] = useState<string | null>(null);

  const sorted = useMemo(
    () => [...data].sort((a, b) => b.importance - a.importance).slice(0, topN),
    [data, topN],
  );

  const maxImportance = sorted[0]?.importance ?? 1;

  // Unique groups for legend
  const groups = useMemo(() => {
    const seen = new Set<string>();
    const result: string[] = [];
    for (const f of sorted) {
      if (!seen.has(f.group)) {
        seen.add(f.group);
        result.push(f.group);
      }
    }
    return result;
  }, [sorted]);

  return (
    <div className="flex flex-col gap-3 p-4 h-full overflow-y-auto">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-bold uppercase tracking-widest text-gray-100">
          Feature Importance
        </h2>
        <span className="text-2xs text-terminal-muted font-mono">
          Top {sorted.length} / {data.length}
        </span>
      </div>

      {/* Group legend */}
      <div className="flex flex-wrap gap-2">
        {groups.map((g) => (
          <button
            key={g}
            onMouseEnter={() => setHoveredGroup(g)}
            onMouseLeave={() => setHoveredGroup(null)}
            className={cn(
              "flex items-center gap-1 px-1.5 py-0.5 rounded text-2xs font-medium transition-opacity",
              hoveredGroup && hoveredGroup !== g ? "opacity-30" : "opacity-100",
            )}
          >
            <span
              className="h-2 w-2 rounded-full flex-shrink-0"
              style={{ backgroundColor: groupColor(g) }}
            />
            <span className="text-terminal-muted capitalize">{g}</span>
          </button>
        ))}
      </div>

      {/* Bar chart */}
      <div className="flex flex-col gap-0.5">
        {sorted.map((feature) => {
          const barPct = (feature.importance / maxImportance) * 100;
          const dimmed = hoveredGroup !== null && feature.group !== hoveredGroup;
          return (
            <div
              key={feature.name}
              className={cn(
                "flex items-center gap-2 py-1 transition-opacity",
                dimmed ? "opacity-20" : "opacity-100",
              )}
            >
              {/* Feature name */}
              <span className="text-2xs font-mono text-gray-300 w-[140px] truncate text-right flex-shrink-0">
                {feature.name}
              </span>

              {/* Bar */}
              <div className="flex-1 h-4 bg-terminal-surface rounded overflow-hidden relative">
                <div
                  className="h-full rounded transition-all duration-300"
                  style={{
                    width: `${barPct}%`,
                    backgroundColor: groupColor(feature.group),
                    opacity: 0.75,
                  }}
                />
              </div>

              {/* IC value */}
              <span className="text-2xs font-mono text-terminal-muted w-[48px] text-right flex-shrink-0 tabular-nums">
                IC {feature.ic.toFixed(3)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
