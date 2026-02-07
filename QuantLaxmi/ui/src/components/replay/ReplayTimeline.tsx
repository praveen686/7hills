"use client";

import { useMemo, useRef } from "react";
import { clsx } from "clsx";
import type { TimelineMarker } from "@/lib/types";
import { formatTime } from "@/lib/formatters";

// ============================================================
// Replay Timeline — Horizontal bar with event markers
//
// Trading session: 09:15 IST — 15:30 IST
// Colored dots per event type, vertical cursor at current time
// ============================================================

interface ReplayTimelineProps {
  markers: TimelineMarker[] | undefined;
  currentTs: string;
  selectedDate: string | null;
  onMarkerClick: (marker: TimelineMarker) => void;
}

// Trading session bounds (minutes from midnight)
const SESSION_START = 9 * 60 + 15; // 09:15
const SESSION_END = 15 * 60 + 30; // 15:30
const SESSION_DURATION = SESSION_END - SESSION_START; // 375 minutes

const TYPE_COLORS: Record<string, string> = {
  signal: "#3b82f6",       // blue
  gate_decision: "#f59e0b", // amber
  order: "#8b5cf6",        // violet
  fill: "#22c55e",         // green
  risk_alert: "#ef4444",   // red
  snapshot: "#6b7280",     // gray
};

const TYPE_LABELS: Record<string, string> = {
  signal: "Signal",
  gate_decision: "Gate",
  order: "Order",
  fill: "Fill",
  risk_alert: "Alert",
  snapshot: "Snapshot",
};

function tsToMinutes(ts: string): number {
  try {
    const d = new Date(ts);
    return d.getUTCHours() * 60 + d.getUTCMinutes() + d.getUTCSeconds() / 60;
  } catch {
    return SESSION_START;
  }
}

function minutesToPct(minutes: number): number {
  const clamped = Math.max(SESSION_START, Math.min(SESSION_END, minutes));
  return ((clamped - SESSION_START) / SESSION_DURATION) * 100;
}

export function ReplayTimeline({
  markers,
  currentTs,
  selectedDate,
  onMarkerClick,
}: ReplayTimelineProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  const cursorPct = useMemo(() => {
    if (!currentTs) return 0;
    return minutesToPct(tsToMinutes(currentTs));
  }, [currentTs]);

  // Group markers at same position to avoid overlap
  const positionedMarkers = useMemo(() => {
    if (!markers) return [];
    return markers.map((m) => ({
      ...m,
      pct: minutesToPct(tsToMinutes(m.ts)),
    }));
  }, [markers]);

  if (!selectedDate) {
    return (
      <div className="card p-6 flex items-center justify-center text-gray-600 text-sm">
        Select a date to view the timeline
      </div>
    );
  }

  return (
    <div className="card p-4">
      {/* Legend */}
      <div className="flex items-center gap-4 mb-3">
        {Object.entries(TYPE_LABELS).map(([type, label]) => (
          <div key={type} className="flex items-center gap-1.5">
            <span
              className="w-2.5 h-2.5 rounded-full"
              style={{ backgroundColor: TYPE_COLORS[type] }}
            />
            <span className="text-[10px] text-gray-500 uppercase tracking-wider">
              {label}
            </span>
          </div>
        ))}
      </div>

      {/* Timeline Bar */}
      <div ref={containerRef} className="relative h-14 bg-gray-900 rounded-lg overflow-hidden">
        {/* Time labels */}
        <div className="absolute top-0 left-0 right-0 flex justify-between px-2 pt-1">
          <span className="text-[10px] text-gray-600 font-mono">09:15</span>
          <span className="text-[10px] text-gray-600 font-mono">10:30</span>
          <span className="text-[10px] text-gray-600 font-mono">12:00</span>
          <span className="text-[10px] text-gray-600 font-mono">13:30</span>
          <span className="text-[10px] text-gray-600 font-mono">15:30</span>
        </div>

        {/* Hour gridlines */}
        {[10, 11, 12, 13, 14, 15].map((hour) => (
          <div
            key={hour}
            className="absolute top-0 bottom-0 w-px bg-gray-800"
            style={{ left: `${minutesToPct(hour * 60)}%` }}
          />
        ))}

        {/* Event markers */}
        {positionedMarkers.map((m) => (
          <button
            key={`${m.seq}-${m.event_type}`}
            className="absolute transform -translate-x-1/2 bottom-2 group cursor-pointer"
            style={{ left: `${m.pct}%` }}
            onClick={() => onMarkerClick(m)}
          >
            <span
              className={clsx(
                "block w-3 h-3 rounded-full border-2 border-gray-900",
                "transition-transform hover:scale-150"
              )}
              style={{ backgroundColor: TYPE_COLORS[m.event_type] || "#6b7280" }}
            />
            {/* Tooltip on hover */}
            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block z-20">
              <div className="bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs whitespace-nowrap shadow-lg">
                <p className="text-gray-300 font-medium">
                  {TYPE_LABELS[m.event_type] || m.event_type}
                </p>
                <p className="text-gray-500 font-mono text-[10px]">
                  {formatTime(m.ts)}
                </p>
                <p className="text-gray-400 text-[10px] mt-0.5 max-w-[200px] truncate">
                  {m.summary}
                </p>
                {m.strategy_id && (
                  <p className="text-gray-600 text-[10px]">
                    {m.strategy_id} / {m.symbol}
                  </p>
                )}
              </div>
            </div>
          </button>
        ))}

        {/* Cursor line */}
        {currentTs && (
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-accent z-10 transition-[left] duration-200"
            style={{ left: `${cursorPct}%` }}
          >
            <div className="absolute -top-0.5 left-1/2 -translate-x-1/2 w-2 h-2 rounded-full bg-accent" />
          </div>
        )}
      </div>
    </div>
  );
}
