"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchOptionChain, fetchVIX } from "@/lib/api";
import { formatCurrencyPrecise, formatNumber, pnlColor } from "@/lib/formatters";
import { clsx } from "clsx";
import type { OptionChainEntry } from "@/lib/types";

// ============================================================
// Market Page — Option Chain, IV Surface, Tick Chart
// ============================================================

const INDICES = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"];

export default function MarketPage() {
  const [selectedIndex, setSelectedIndex] = useState("NIFTY");
  const [selectedExpiry, setSelectedExpiry] = useState("");

  const { data: vix } = useQuery({
    queryKey: ["vix"],
    queryFn: fetchVIX,
    refetchInterval: 10000,
  });

  const { data: optionChain, isLoading: chainLoading } = useQuery<OptionChainEntry[]>({
    queryKey: ["optionChain", selectedIndex, selectedExpiry],
    queryFn: () => fetchOptionChain(selectedIndex, selectedExpiry || undefined),
    refetchInterval: 5000,
  });

  return (
    <div className="space-y-6 animate-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-white">Market</h1>
          <p className="text-sm text-gray-500 mt-1">
            Option chains, implied volatility, and tick data
          </p>
        </div>
        {vix && (
          <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gray-900 border border-gray-800">
            <span className="text-sm text-gray-400">VIX</span>
            <span className="text-lg font-mono font-semibold text-white">
              {vix.value.toFixed(2)}
            </span>
          </div>
        )}
      </div>

      {/* Index Selector */}
      <div className="flex items-center gap-4">
        <div className="flex gap-1 bg-gray-900 rounded-lg p-1">
          {INDICES.map((idx) => (
            <button
              key={idx}
              onClick={() => setSelectedIndex(idx)}
              className={clsx(
                "px-4 py-1.5 text-xs font-medium rounded-md transition-colors",
                selectedIndex === idx
                  ? "bg-accent text-white"
                  : "text-gray-500 hover:text-gray-300"
              )}
            >
              {idx}
            </button>
          ))}
        </div>

        <input
          type="date"
          value={selectedExpiry}
          onChange={(e) => setSelectedExpiry(e.target.value)}
          className="input-field max-w-[180px]"
        />
      </div>

      {/* Option Chain Heatmap */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <p className="card-header mb-0">
            Option Chain — {selectedIndex}
          </p>
          <span className="text-xs text-gray-500">Expiry: {selectedExpiry || "Latest available"}</span>
        </div>

        {chainLoading ? (
          <div className="animate-pulse space-y-2">
            {Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className="h-8 bg-gray-800 rounded" />
            ))}
          </div>
        ) : optionChain && optionChain.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800">
                  <th className="table-header text-profit" colSpan={4}>
                    CALLS
                  </th>
                  <th className="table-header text-center text-accent">
                    Strike
                  </th>
                  <th className="table-header text-loss" colSpan={4}>
                    PUTS
                  </th>
                </tr>
                <tr className="border-b border-gray-800">
                  <th className="table-header">OI</th>
                  <th className="table-header">IV</th>
                  <th className="table-header">Delta</th>
                  <th className="table-header">LTP</th>
                  <th className="table-header text-center" />
                  <th className="table-header">LTP</th>
                  <th className="table-header">Delta</th>
                  <th className="table-header">IV</th>
                  <th className="table-header">OI</th>
                </tr>
              </thead>
              <tbody>
                {optionChain.map((row, i) => (
                  <tr key={i} className="border-b border-gray-800/50 hover:bg-gray-900/50">
                    <td className="table-cell font-mono text-xs">
                      {formatNumber(row.ce_oi)}
                    </td>
                    <td className="table-cell font-mono text-xs">
                      <IVCell iv={row.ce_iv} />
                    </td>
                    <td className="table-cell font-mono text-xs text-gray-400">
                      {row.ce_delta.toFixed(3)}
                    </td>
                    <td className="table-cell font-mono text-xs font-medium text-profit">
                      {formatCurrencyPrecise(row.ce_ltp)}
                    </td>
                    <td className="table-cell font-mono text-xs font-bold text-accent text-center bg-gray-900">
                      {formatNumber(row.strike)}
                    </td>
                    <td className="table-cell font-mono text-xs font-medium text-loss">
                      {formatCurrencyPrecise(row.pe_ltp)}
                    </td>
                    <td className="table-cell font-mono text-xs text-gray-400">
                      {row.pe_delta.toFixed(3)}
                    </td>
                    <td className="table-cell font-mono text-xs">
                      <IVCell iv={row.pe_iv} />
                    </td>
                    <td className="table-cell font-mono text-xs">
                      {formatNumber(row.pe_oi)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="flex items-center justify-center h-48 text-sm text-gray-600">
            <div className="text-center">
              <p>No option chain data available</p>
              <p className="text-xs text-gray-700 mt-1">
                Connect to the backend to see live data
              </p>
            </div>
          </div>
        )}
      </div>

      {/* IV Surface + Tick Chart placeholders */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* IV Surface */}
        <div className="card">
          <p className="card-header mb-4">IV Surface</p>
          <div className="flex items-center justify-center h-64 border border-dashed border-gray-800 rounded-lg">
            <div className="text-center">
              <svg
                className="w-10 h-10 text-gray-700 mx-auto mb-2"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1}
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
              </svg>
              <p className="text-xs text-gray-600">3D IV Surface</p>
              <p className="text-[10px] text-gray-700 mt-1">
                Requires WebGL rendering — coming soon
              </p>
            </div>
          </div>
        </div>

        {/* Tick Chart */}
        <div className="card">
          <p className="card-header mb-4">Tick Chart — {selectedIndex}</p>
          <div className="flex items-center justify-center h-64 border border-dashed border-gray-800 rounded-lg">
            <div className="text-center">
              <svg
                className="w-10 h-10 text-gray-700 mx-auto mb-2"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1}
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
              <p className="text-xs text-gray-600">
                TradingView Lightweight Charts
              </p>
              <p className="text-[10px] text-gray-700 mt-1">
                Real-time tick data via WebSocket
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------- IV Heatmap Cell ----------

function IVCell({ iv }: { iv: number }) {
  // Color intensity based on IV level
  const intensity = Math.min(iv / 50, 1);
  const bg =
    iv > 30
      ? `rgba(239, 68, 68, ${intensity * 0.3})`
      : iv > 20
        ? `rgba(234, 179, 8, ${intensity * 0.3})`
        : `rgba(34, 197, 94, ${intensity * 0.2})`;

  return (
    <span
      className="inline-block px-1.5 py-0.5 rounded"
      style={{ backgroundColor: bg }}
    >
      {iv.toFixed(1)}%
    </span>
  );
}
