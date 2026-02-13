import { useMemo, useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import { apiFetch } from "@/lib/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface AlphaPoint {
  day: number; // lag in days
  ic: number; // information coefficient
}

interface AlphaDecayProps {
  data?: AlphaPoint[];
  threshold?: number;
  signalName?: string;
}

// ---------------------------------------------------------------------------
// Demo data — declining IC over 30-day horizon
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function AlphaDecay({
  data,
  threshold = 0.02,
  signalName = "DFF Composite",
}: AlphaDecayProps) {
  const [fetchedPoints, setFetchedPoints] = useState<AlphaPoint[]>([]);

  useEffect(() => {
    if (data) return;
    apiFetch<Array<{
      feature: string;
      ic_mean: number;
      horizon: string;
    }>>("/api/research/feature-ic").then((list) => {
      // Group by horizon and average IC
      const byHorizon = new Map<string, number[]>();
      for (const f of list) {
        if (!f.horizon) continue;
        const arr = byHorizon.get(f.horizon) ?? [];
        arr.push(Math.abs(f.ic_mean));
        byHorizon.set(f.horizon, arr);
      }
      const points: AlphaPoint[] = [];
      for (const [h, ics] of byHorizon) {
        const day = parseInt(h);
        if (isNaN(day)) continue;
        points.push({ day, ic: ics.reduce((a, b) => a + b, 0) / ics.length });
      }
      points.sort((a, b) => a.day - b.day);
      if (points.length > 0) setFetchedPoints(points);
    }).catch(() => {});
  }, [data]);

  const points = data ?? fetchedPoints;

  // SVG dimensions
  const svgW = 500;
  const svgH = 180;
  const padL = 45;
  const padR = 15;
  const padT = 15;
  const padB = 30;
  const plotW = svgW - padL - padR;
  const plotH = svgH - padT - padB;

  const { polyline, thresholdY, currentIC, belowThreshold } = useMemo(() => {
    if (points.length === 0) {
      return { polyline: "", thresholdY: 0, currentIC: 0, belowThreshold: false };
    }

    const maxIC = Math.max(...points.map((p) => p.ic), threshold * 1.5);
    const minIC = Math.min(...points.map((p) => p.ic), 0);
    const range = maxIC - minIC || 1;
    const maxDay = points[points.length - 1].day;

    const pts = points.map((p) => {
      const x = padL + (p.day / maxDay) * plotW;
      const y = padT + (1 - (p.ic - minIC) / range) * plotH;
      return `${x},${y}`;
    });

    const thY = padT + (1 - (threshold - minIC) / range) * plotH;
    const lastIC = points[points.length - 1].ic;

    return {
      polyline: pts.join(" "),
      thresholdY: thY,
      currentIC: lastIC,
      belowThreshold: lastIC < threshold,
    };
  }, [points, threshold, plotW, plotH]);

  // Y-axis labels
  const maxIC = Math.max(...points.map((p) => p.ic), threshold * 1.5);
  const minIC = Math.min(...points.map((p) => p.ic), 0);
  const yTicks = [maxIC, (maxIC + minIC) / 2, minIC];

  return (
    <div className="flex flex-col gap-3 p-4 h-full">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-bold uppercase tracking-widest text-terminal-text">
          Alpha Decay
        </h2>
        <span className="text-2xs text-terminal-muted font-mono">{signalName}</span>
      </div>

      {/* Alert if below threshold */}
      {belowThreshold && (
        <div className="flex items-center gap-2 rounded border border-terminal-warning/40 bg-terminal-warning/10 px-3 py-1.5">
          <span className="h-2 w-2 rounded-full bg-terminal-warning animate-pulse" />
          <span className="text-xs text-terminal-warning">
            IC dropped below threshold ({threshold.toFixed(3)}) — signal may be decaying
          </span>
        </div>
      )}

      {/* SVG Chart */}
      <div className="flex-1 min-h-[180px] rounded border border-terminal-border bg-terminal-surface p-2">
        <svg
          viewBox={`0 0 ${svgW} ${svgH}`}
          className="w-full h-full"
          preserveAspectRatio="xMidYMid meet"
        >
          {/* Grid lines */}
          {yTicks.map((tick, i) => {
            const y = padT + (1 - (tick - minIC) / (maxIC - minIC || 1)) * plotH;
            return (
              <g key={i}>
                <line x1={padL} x2={svgW - padR} y1={y} y2={y} stroke="rgb(var(--terminal-border))" strokeWidth={1} />
                <text x={padL - 5} y={y + 3} textAnchor="end" fill="rgb(var(--terminal-muted))" fontSize={9} fontFamily="JetBrains Mono">
                  {tick.toFixed(3)}
                </text>
              </g>
            );
          })}

          {/* X-axis labels */}
          {[0, 10, 20, 30].filter((d) => d <= (points[points.length - 1]?.day ?? 30)).map((d) => {
            const x = padL + (d / (points[points.length - 1]?.day ?? 30)) * plotW;
            return (
              <text key={d} x={x} y={svgH - 5} textAnchor="middle" fill="rgb(var(--terminal-muted))" fontSize={9} fontFamily="JetBrains Mono">
                {d}d
              </text>
            );
          })}

          {/* Threshold line */}
          <line
            x1={padL}
            x2={svgW - padR}
            y1={thresholdY}
            y2={thresholdY}
            stroke="rgb(var(--terminal-warning))"
            strokeWidth={1}
            strokeDasharray="4,3"
          />
          <text x={svgW - padR + 2} y={thresholdY + 3} fill="rgb(var(--terminal-warning))" fontSize={8} fontFamily="JetBrains Mono">
            thr
          </text>

          {/* IC line */}
          <polyline
            points={polyline}
            fill="none"
            stroke="rgb(var(--terminal-accent))"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {/* Fill area under the curve */}
          {points.length > 0 && (
            <polygon
              points={`${padL},${padT + plotH} ${polyline} ${padL + (points[points.length - 1].day / (points[points.length - 1]?.day ?? 30)) * plotW},${padT + plotH}`}
              fill="rgb(var(--terminal-accent) / 0.1)"
            />
          )}

          {/* End point dot */}
          {points.length > 0 && (() => {
            const last = points[points.length - 1];
            const x = padL + (last.day / (points[points.length - 1]?.day ?? 30)) * plotW;
            const y = padT + (1 - (last.ic - minIC) / (maxIC - minIC || 1)) * plotH;
            return (
              <circle
                cx={x}
                cy={y}
                r={4}
                fill={belowThreshold ? "rgb(var(--terminal-loss))" : "rgb(var(--terminal-accent))"}
                stroke="rgb(var(--terminal-surface))"
                strokeWidth={2}
              />
            );
          })()}
        </svg>
      </div>

      {/* Current IC */}
      <div className="flex items-center justify-between text-xs font-mono">
        <span className="text-terminal-muted">
          Current IC at {points[points.length - 1]?.day ?? 0}d lag:
        </span>
        <span className={cn(
          "font-semibold",
          belowThreshold ? "text-terminal-loss" : "text-terminal-profit",
        )}>
          {currentIC.toFixed(4)}
        </span>
      </div>
    </div>
  );
}
