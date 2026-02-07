"use client";

import { useState } from "react";
import type { MissedOpportunity } from "@/lib/types";

interface Props {
  opportunities: MissedOpportunity[] | undefined;
  isLoading: boolean;
}

type SortKey = "hypothetical_pnl_pct" | "conviction" | "ts";

export function MissedOpportunityTable({ opportunities, isLoading }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>("hypothetical_pnl_pct");
  const [sortDesc, setSortDesc] = useState(true);

  if (isLoading) {
    return <div className="bg-gray-900 rounded-xl p-6 animate-pulse h-64" />;
  }

  if (!opportunities || opportunities.length === 0) {
    return (
      <div className="bg-gray-900 rounded-xl p-6 text-center text-gray-500">
        No missed opportunities found for this period
      </div>
    );
  }

  const sorted = [...opportunities].sort((a, b) => {
    if (sortKey === "ts") {
      return sortDesc ? b.ts.localeCompare(a.ts) : a.ts.localeCompare(b.ts);
    }
    const aVal = a[sortKey] as number;
    const bVal = b[sortKey] as number;
    return sortDesc ? bVal - aVal : aVal - bVal;
  });

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDesc(!sortDesc);
    } else {
      setSortKey(key);
      setSortDesc(true);
    }
  };

  return (
    <div className="bg-gray-900 rounded-xl overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-800">
              <th className="px-4 py-3 text-left text-xs text-gray-500 uppercase">Date</th>
              <th className="px-4 py-3 text-left text-xs text-gray-500 uppercase">Strategy</th>
              <th className="px-4 py-3 text-left text-xs text-gray-500 uppercase">Symbol</th>
              <th className="px-4 py-3 text-left text-xs text-gray-500 uppercase">Dir</th>
              <th
                className="px-4 py-3 text-right text-xs text-gray-500 uppercase cursor-pointer hover:text-gray-300"
                onClick={() => handleSort("conviction")}
              >
                Conviction {sortKey === "conviction" ? (sortDesc ? "↓" : "↑") : ""}
              </th>
              <th className="px-4 py-3 text-left text-xs text-gray-500 uppercase">Block Reason</th>
              <th
                className="px-4 py-3 text-right text-xs text-gray-500 uppercase cursor-pointer hover:text-gray-300"
                onClick={() => handleSort("hypothetical_pnl_pct")}
              >
                Hyp. P&L {sortKey === "hypothetical_pnl_pct" ? (sortDesc ? "↓" : "↑") : ""}
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((opp, i) => {
              const pnlColor = opp.hypothetical_pnl_pct >= 0 ? "text-profit" : "text-loss";
              const dirBadge = opp.direction === "long"
                ? "text-profit"
                : "text-loss";
              return (
                <tr key={i} className="border-b border-gray-800/50 hover:bg-gray-800/50">
                  <td className="px-4 py-2.5 text-gray-400 font-mono text-xs">
                    {opp.ts.slice(0, 10)}
                  </td>
                  <td className="px-4 py-2.5 text-gray-300">{opp.strategy_id}</td>
                  <td className="px-4 py-2.5 text-white font-medium">{opp.symbol}</td>
                  <td className={`px-4 py-2.5 font-medium ${dirBadge}`}>
                    {opp.direction.toUpperCase()}
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-accent">
                    {opp.conviction.toFixed(2)}
                  </td>
                  <td className="px-4 py-2.5 text-gray-400 text-xs max-w-[200px] truncate">
                    {opp.block_reason}
                  </td>
                  <td className={`px-4 py-2.5 text-right font-mono ${pnlColor}`}>
                    {opp.price_data_available
                      ? `${(opp.hypothetical_pnl_pct * 100).toFixed(2)}%`
                      : "\u2014"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
