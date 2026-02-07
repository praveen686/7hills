"use client";

import type { Signal } from "@/lib/types";
import type { WhyPanelTarget } from "@/hooks/useWhyPanel";
import { formatTime, formatCurrencyPrecise, pnlColor } from "@/lib/formatters";
import { clsx } from "clsx";

// ============================================================
// Today's Signals Table
// ============================================================

interface SignalTableProps {
  signals: Signal[];
  maxRows?: number;
  onSignalClick?: (target: WhyPanelTarget) => void;
}

function DirectionBadge({ direction }: { direction: Signal["direction"] }) {
  const styles = {
    BUY: "bg-profit/10 text-profit border-profit/20",
    SELL: "bg-loss/10 text-loss border-loss/20",
    HOLD: "bg-gray-800 text-gray-400 border-gray-700",
  };

  return (
    <span
      className={clsx(
        "inline-flex items-center px-2 py-0.5 rounded text-xs font-medium border",
        styles[direction]
      )}
    >
      {direction}
    </span>
  );
}

function StrengthBar({ strength }: { strength: number }) {
  const pct = Math.min(Math.max(strength * 100, 0), 100);
  const color =
    pct >= 70
      ? "bg-profit"
      : pct >= 40
        ? "bg-yellow-500"
        : "bg-gray-600";

  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs font-mono text-gray-400">
        {strength.toFixed(2)}
      </span>
    </div>
  );
}

export function SignalTable({ signals, maxRows = 20, onSignalClick }: SignalTableProps) {
  const displayed = signals.slice(0, maxRows);

  if (displayed.length === 0) {
    return (
      <div className="flex items-center justify-center h-32 text-sm text-gray-600">
        No signals generated today
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="border-b border-gray-800">
            <th className="table-header">Time</th>
            <th className="table-header">Symbol</th>
            <th className="table-header">Direction</th>
            <th className="table-header">Price</th>
            <th className="table-header">Target</th>
            <th className="table-header">SL</th>
            <th className="table-header">Strength</th>
            <th className="table-header">Strategy</th>
          </tr>
        </thead>
        <tbody>
          {displayed.map((signal) => (
            <tr
              key={signal.id}
              className={clsx(
                "border-b border-gray-800/50 hover:bg-gray-900/50 transition-colors",
                onSignalClick && "cursor-pointer hover:bg-accent/5"
              )}
              onClick={() => {
                if (onSignalClick) {
                  // Extract date from signal timestamp (YYYY-MM-DD)
                  const date = signal.timestamp.slice(0, 10);
                  onSignalClick({
                    strategy_id: signal.strategy_id,
                    symbol: signal.symbol,
                    date,
                  });
                }
              }}
            >
              <td className="table-cell font-mono text-gray-400 text-xs">
                {formatTime(signal.timestamp)}
              </td>
              <td className="table-cell font-medium text-white">
                {signal.symbol}
              </td>
              <td className="table-cell">
                <DirectionBadge direction={signal.direction} />
              </td>
              <td className="table-cell font-mono">
                {formatCurrencyPrecise(signal.price)}
              </td>
              <td className="table-cell font-mono">
                {signal.target ? (
                  <span className={pnlColor(signal.target - signal.price)}>
                    {formatCurrencyPrecise(signal.target)}
                  </span>
                ) : (
                  <span className="text-gray-600">--</span>
                )}
              </td>
              <td className="table-cell font-mono">
                {signal.stop_loss ? (
                  <span className="text-loss">
                    {formatCurrencyPrecise(signal.stop_loss)}
                  </span>
                ) : (
                  <span className="text-gray-600">--</span>
                )}
              </td>
              <td className="table-cell">
                <StrengthBar strength={signal.strength} />
              </td>
              <td className="table-cell text-xs text-gray-500">
                {signal.strategy_name}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
