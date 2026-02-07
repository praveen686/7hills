"use client";

import { useRef, useEffect } from "react";
import { clsx } from "clsx";
import type { StepEvent } from "@/lib/types";
import { formatTime } from "@/lib/formatters";

// ============================================================
// Replay Event Log — Scrollable list of recent events
// ============================================================

interface ReplayEventLogProps {
  events: StepEvent[];
  onEventClick: (event: StepEvent) => void;
}

const TYPE_BADGES: Record<string, { label: string; className: string }> = {
  signal: { label: "SIG", className: "bg-blue-500/20 text-blue-400 border-blue-500/30" },
  gate_decision: { label: "GATE", className: "bg-amber-500/20 text-amber-400 border-amber-500/30" },
  order: { label: "ORD", className: "bg-violet-500/20 text-violet-400 border-violet-500/30" },
  fill: { label: "FILL", className: "bg-green-500/20 text-green-400 border-green-500/30" },
  risk_alert: { label: "RISK", className: "bg-red-500/20 text-red-400 border-red-500/30" },
  snapshot: { label: "SNAP", className: "bg-gray-500/20 text-gray-400 border-gray-500/30" },
};

function summarizePayload(event: StepEvent): string {
  const p = event.payload;
  const t = event.event_type;

  if (t === "signal") {
    return `${p.direction || "?"} conviction=${Number(p.conviction || 0).toFixed(2)}`;
  }
  if (t === "gate_decision") {
    return `${p.gate || "?"}: ${p.approved ? "PASS" : "BLOCK"}`;
  }
  if (t === "order") {
    return `${p.action || "?"} ${p.side || "?"} ${p.order_type || ""}`;
  }
  if (t === "fill") {
    return `${p.side || "?"} @${p.price || 0} qty=${p.quantity || 0}`;
  }
  if (t === "risk_alert") {
    return `${p.alert_type || "?"}: ${p.new_state || ""}`;
  }
  if (t === "snapshot") {
    return `equity=${Number(p.equity || 0).toFixed(4)} dd=${Number(p.portfolio_dd || 0).toFixed(4)}`;
  }
  return JSON.stringify(p).slice(0, 80);
}

export function ReplayEventLog({ events, onEventClick }: ReplayEventLogProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new events arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [events.length]);

  if (events.length === 0) {
    return (
      <div className="card p-6 flex items-center justify-center text-gray-600 text-sm h-full">
        Events will appear here during playback
      </div>
    );
  }

  return (
    <div className="card flex flex-col h-full">
      <div className="px-4 py-3 border-b border-gray-800">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          Event Log
          <span className="text-gray-600 font-normal ml-2">{events.length} events</span>
        </h3>
      </div>

      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto divide-y divide-gray-800/50"
        style={{ maxHeight: "400px" }}
      >
        {events.map((event, idx) => {
          const badge = TYPE_BADGES[event.event_type] || TYPE_BADGES.snapshot;
          const hasStrategy = !!event.strategy_id;

          return (
            <button
              key={`${event.seq}-${idx}`}
              className="w-full text-left px-4 py-2.5 hover:bg-gray-900/50 transition-colors flex items-center gap-3"
              onClick={() => onEventClick(event)}
            >
              {/* Time */}
              <span className="text-[10px] font-mono text-gray-600 shrink-0 w-16">
                {formatTime(event.ts)}
              </span>

              {/* Type badge */}
              <span
                className={clsx(
                  "inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-bold border shrink-0",
                  badge.className
                )}
              >
                {badge.label}
              </span>

              {/* Strategy / Symbol */}
              {hasStrategy && (
                <span className="text-xs text-gray-500 shrink-0">
                  {event.strategy_id}/{event.symbol}
                </span>
              )}

              {/* Summary */}
              <span className="text-xs text-gray-400 truncate">
                {summarizePayload(event)}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
