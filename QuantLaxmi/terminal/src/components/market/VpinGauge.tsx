import { useMemo, useCallback, useState, useEffect } from "react";
import { useAtomValue } from "jotai";
import { selectedSymbolAtom, selectedVpinAtom, vpinAtom } from "@/stores/market";
import { useTauriStream } from "@/hooks/useTauriStream";
import { useAtom } from "jotai";
import { cn, clamp } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GAUGE_SIZE = 160;
const STROKE_WIDTH = 14;
const RADIUS = (GAUGE_SIZE - STROKE_WIDTH) / 2;
const CENTER = GAUGE_SIZE / 2;

// Arc spans 180 degrees (semicircle), from 180deg (left) to 0deg (right)
const START_ANGLE = Math.PI; // 180 degrees
const END_ANGLE = 0; // 0 degrees (2*PI equivalent for sweep)
const ARC_LENGTH = Math.PI; // 180 degrees total

const HISTORY_LENGTH = 60;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Get color for a VPIN value (green -> yellow -> red). */
function vpinColor(value: number): string {
  const v = clamp(value, 0, 1);
  if (v <= 0.3) {
    // Green to yellow-green
    const t = v / 0.3;
    const r = Math.round(t * 255);
    const g = Math.round(212 - t * 42);
    return `rgb(${r},${g},0)`;
  } else if (v <= 0.6) {
    // Yellow-green to orange
    const t = (v - 0.3) / 0.3;
    const r = 255;
    const g = Math.round(170 - t * 86);
    return `rgb(${r},${g},0)`;
  } else {
    // Orange to red
    const t = (v - 0.6) / 0.4;
    const r = 255;
    const g = Math.round(84 - t * 84);
    return `rgb(${r},${g},${Math.round(t * 40)})`;
  }
}

/** Get SVG arc path for a semicircle segment. */
function describeArc(
  cx: number,
  cy: number,
  r: number,
  startAngle: number,
  endAngle: number,
): string {
  // SVG arcs go clockwise; our angles go counter-clockwise (math convention)
  // We draw from startAngle to endAngle sweeping clockwise in screen coords
  const x1 = cx + r * Math.cos(startAngle);
  const y1 = cy - r * Math.sin(startAngle);
  const x2 = cx + r * Math.cos(endAngle);
  const y2 = cy - r * Math.sin(endAngle);

  const sweep = startAngle > endAngle ? 0 : 1;
  const largeArc = Math.abs(startAngle - endAngle) > Math.PI ? 1 : 0;

  return `M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} ${sweep} ${x2} ${y2}`;
}

/** Get label text for VPIN level. */
function vpinLabel(value: number): string {
  if (value <= 0.3) return "LOW";
  if (value <= 0.6) return "MODERATE";
  return "HIGH";
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function VpinGauge() {
  const symbol = useAtomValue(selectedSymbolAtom);
  const vpin = useAtomValue(selectedVpinAtom);
  const [, setVpinMap] = useAtom(vpinAtom);
  const [history, setHistory] = useState<number[]>([]);

  const value = vpin ?? 0;

  // ---- Subscribe to live VPIN updates ----
  const handleVpin = useCallback(
    (data: { symbol: string; vpin: number }) => {
      setVpinMap((prev) => {
        const next = new Map(prev);
        next.set(data.symbol, data.vpin);
        return next;
      });

      if (data.symbol === symbol) {
        setHistory((prev) => {
          const next = [...prev, data.vpin];
          if (next.length > HISTORY_LENGTH) {
            return next.slice(next.length - HISTORY_LENGTH);
          }
          return next;
        });
      }
    },
    [symbol, setVpinMap],
  );

  useTauriStream<{ symbol: string; vpin: number }>("vpin_update", handleVpin);

  // Reset history on symbol change
  useEffect(() => {
    setHistory([]);
  }, [symbol]);

  // ---- Gauge arc paths ----
  const { backgroundArc, valueArc, needleAngle, color } = useMemo(() => {
    const v = clamp(value, 0, 1);
    const fillAngle = START_ANGLE - v * ARC_LENGTH;

    return {
      backgroundArc: describeArc(CENTER, CENTER, RADIUS, START_ANGLE, END_ANGLE),
      valueArc: describeArc(CENTER, CENTER, RADIUS, START_ANGLE, fillAngle),
      needleAngle: START_ANGLE - v * ARC_LENGTH,
      color: vpinColor(v),
    };
  }, [value]);

  // ---- Mini sparkline for history ----
  const sparklinePath = useMemo(() => {
    if (history.length < 2) return "";

    const w = GAUGE_SIZE - 20;
    const h = 24;
    const stepX = w / (HISTORY_LENGTH - 1);

    const points = history.map((v, i) => {
      const x = 10 + i * stepX;
      const y = h - clamp(v, 0, 1) * h;
      return `${x},${y}`;
    });

    return `M ${points.join(" L ")}`;
  }, [history]);

  return (
    <div className="flex flex-col items-center justify-center h-full w-full bg-terminal-surface p-3">
      {/* SVG Gauge */}
      <svg
        width={GAUGE_SIZE}
        height={GAUGE_SIZE * 0.6 + 10}
        viewBox={`0 ${CENTER - RADIUS - STROKE_WIDTH} ${GAUGE_SIZE} ${RADIUS + STROKE_WIDTH * 2 + 10}`}
        className="overflow-visible"
      >
        {/* Gradient definition */}
        <defs>
          <linearGradient id="vpin-grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#00d4aa" />
            <stop offset="33%" stopColor="#ffb84d" />
            <stop offset="66%" stopColor="#ff8c00" />
            <stop offset="100%" stopColor="#ff4d6a" />
          </linearGradient>
        </defs>

        {/* Background arc (full semicircle, dim) */}
        <path
          d={backgroundArc}
          fill="none"
          stroke="#1e1e2e"
          strokeWidth={STROKE_WIDTH}
          strokeLinecap="round"
        />

        {/* Value arc (colored fill) */}
        {value > 0.005 && (
          <path
            d={valueArc}
            fill="none"
            stroke={color}
            strokeWidth={STROKE_WIDTH}
            strokeLinecap="round"
            className="transition-all duration-300"
          />
        )}

        {/* Tick marks */}
        {[0, 0.3, 0.6, 1.0].map((tick) => {
          const angle = START_ANGLE - tick * ARC_LENGTH;
          const outerR = RADIUS + STROKE_WIDTH / 2 + 3;
          const innerR = RADIUS + STROKE_WIDTH / 2 + 8;
          const x1 = CENTER + outerR * Math.cos(angle);
          const y1 = CENTER - outerR * Math.sin(angle);
          const x2 = CENTER + innerR * Math.cos(angle);
          const y2 = CENTER - innerR * Math.sin(angle);

          return (
            <g key={tick}>
              <line
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke="#6b6b8a"
                strokeWidth={1}
              />
              <text
                x={x2 + (Math.cos(angle) > 0.1 ? 4 : Math.cos(angle) < -0.1 ? -4 : 0)}
                y={y2 + 4}
                fill="#6b6b8a"
                fontSize="8"
                fontFamily="'JetBrains Mono', monospace"
                textAnchor={Math.cos(angle) > 0.1 ? "start" : Math.cos(angle) < -0.1 ? "end" : "middle"}
              >
                {tick.toFixed(1)}
              </text>
            </g>
          );
        })}

        {/* Needle */}
        <line
          x1={CENTER}
          y1={CENTER}
          x2={CENTER + (RADIUS - 10) * Math.cos(needleAngle)}
          y2={CENTER - (RADIUS - 10) * Math.sin(needleAngle)}
          stroke={color}
          strokeWidth={2}
          strokeLinecap="round"
          className="transition-all duration-300"
        />
        <circle cx={CENTER} cy={CENTER} r={4} fill={color} />
        <circle cx={CENTER} cy={CENTER} r={2} fill="#08080d" />

        {/* Value text */}
        <text
          x={CENTER}
          y={CENTER + 18}
          textAnchor="middle"
          fill={color}
          fontSize="22"
          fontWeight="bold"
          fontFamily="'JetBrains Mono', monospace"
          className="transition-all duration-300"
        >
          {value.toFixed(3)}
        </text>
      </svg>

      {/* Label */}
      <div className="flex flex-col items-center mt-1 gap-0.5">
        <span className="text-xs font-mono font-bold text-terminal-muted tracking-widest">
          VPIN
        </span>
        <span
          className={cn(
            "text-2xs font-mono font-semibold px-2 py-0.5 rounded",
            value <= 0.3 && "text-terminal-profit bg-terminal-profit/10",
            value > 0.3 && value <= 0.6 && "text-terminal-warning bg-terminal-warning/10",
            value > 0.6 && "text-terminal-loss bg-terminal-loss/10",
          )}
        >
          {vpinLabel(value)}
        </span>
      </div>

      {/* Mini sparkline */}
      {history.length >= 2 && (
        <div className="mt-2 w-full">
          <svg
            width="100%"
            height={28}
            viewBox={`0 -2 ${GAUGE_SIZE} 28`}
            preserveAspectRatio="none"
          >
            <path
              d={sparklinePath}
              fill="none"
              stroke={color}
              strokeWidth={1.5}
              opacity={0.7}
            />
          </svg>
        </div>
      )}
    </div>
  );
}
