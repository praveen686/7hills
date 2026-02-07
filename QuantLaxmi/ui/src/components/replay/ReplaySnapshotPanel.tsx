"use client";

import { clsx } from "clsx";
import type { ReplaySnapshot } from "@/lib/types";
import { formatPct } from "@/lib/formatters";

// ============================================================
// Replay Snapshot Panel — Current portfolio state card
// ============================================================

interface ReplaySnapshotPanelProps {
  snapshot: ReplaySnapshot | null;
}

export function ReplaySnapshotPanel({ snapshot }: ReplaySnapshotPanelProps) {
  if (!snapshot) {
    return (
      <div className="card p-6 flex items-center justify-center text-gray-600 text-sm">
        No snapshot yet — press Play to start
      </div>
    );
  }

  const p = snapshot.payload;
  const ddPct = (p.portfolio_dd ?? 0) * 100;

  return (
    <div className="card p-4">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
        Portfolio Snapshot
      </h3>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <StatCard
          label="Equity"
          value={Number(p.equity ?? 0).toFixed(4)}
          mono
        />
        <StatCard
          label="Peak"
          value={Number(p.peak_equity ?? 0).toFixed(4)}
          mono
        />
        <StatCard
          label="Drawdown"
          value={formatPct(ddPct)}
          className={ddPct > 2 ? "text-loss" : "text-gray-300"}
          mono
        />
        <StatCard
          label="Exposure"
          value={Number(p.total_exposure ?? 0).toFixed(4)}
          mono
        />
        <StatCard
          label="VPIN"
          value={Number(p.vpin ?? 0).toFixed(4)}
          className={Number(p.vpin ?? 0) > 0.5 ? "text-loss" : "text-gray-300"}
          mono
        />
        <StatCard
          label="Positions"
          value={String(p.position_count ?? 0)}
          mono
        />
        <StatCard
          label="Regime"
          value={String(p.regime ?? "--")}
        />
        <StatCard
          label="Snapshot Time"
          value={snapshot.ts ? new Date(snapshot.ts).toLocaleTimeString("en-IN", {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
            hour12: false,
          }) : "--"}
          mono
        />
      </div>
    </div>
  );
}

function StatCard({
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
      <p className={clsx("text-sm", mono ? "font-mono" : "", className || "text-gray-300")}>
        {value}
      </p>
    </div>
  );
}
