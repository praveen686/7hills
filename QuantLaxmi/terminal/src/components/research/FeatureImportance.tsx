import { useMemo, useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import { apiFetch } from "@/lib/api";

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
  technical: "rgb(var(--terminal-accent))",
  volatility: "rgb(var(--terminal-loss))",
  volume: "rgb(var(--terminal-profit))",
  momentum: "rgb(var(--terminal-warning))",
  divergence: "#a78bfa",
  crypto: "#f97316",
  news: "#06b6d4",
  macro: "#ec4899",
  options: "#8b5cf6",
  structure: "#14b8a6",
};

function groupColor(group: string): string {
  return GROUP_COLORS[group.toLowerCase()] ?? "rgb(var(--terminal-muted))";
}

// ---------------------------------------------------------------------------
// Demo data
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function FeatureImportance({ features, topN = 30 }: FeatureImportanceProps) {
  const [fetchedFeatures, setFetchedFeatures] = useState<Feature[]>([]);

  useEffect(() => {
    if (features) return;
    apiFetch<Array<{
      feature: string;
      ic_mean: number;
      icir: number | null;
      source: string;
    }>>("/api/research/feature-ic").then((list) => {
      const maxAbsIcir = Math.max(...list.map((f) => Math.abs(f.icir ?? f.ic_mean)), 0.001);
      setFetchedFeatures(list.map((f) => ({
        name: f.feature,
        importance: Math.abs(f.icir ?? f.ic_mean) / maxAbsIcir,
        ic: f.ic_mean,
        group: f.source.replace(/^s\d+_/, "").replace(/_/g, " ").split(" ")[0] || "other",
      })));
    }).catch(() => {});
  }, [features]);

  const data = features ?? fetchedFeatures;
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
        <h2 className="text-sm font-bold uppercase tracking-widest text-terminal-text">
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
      {sorted.length === 0 ? (
        <div className="flex items-center justify-center h-32 text-terminal-muted text-xs font-mono">
          No feature importance data
        </div>
      ) : (
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
              <span className="text-2xs font-mono text-terminal-text-secondary w-[140px] truncate text-right flex-shrink-0">
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
      )}
    </div>
  );
}
