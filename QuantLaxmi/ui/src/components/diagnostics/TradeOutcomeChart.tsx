"use client";

import type { TradeAnalytics } from "@/lib/types";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
} from "recharts";

interface Props {
  trades: TradeAnalytics[];
}

export function TradeOutcomeChart({ trades }: Props) {
  if (!trades || trades.length === 0) {
    return (
      <div className="bg-gray-900 rounded-xl p-6 text-center text-gray-500 h-64 flex items-center justify-center">
        No trade data for chart
      </div>
    );
  }

  const data = trades.slice(0, 30).map((t, i) => ({
    name: `${t.symbol.slice(0, 6)}`,
    pnl: +(t.pnl_pct * 100).toFixed(2),
    mfm: +(t.mfm * 100).toFixed(2),
    efficiency: +(t.efficiency * 100).toFixed(1),
    idx: i,
  }));

  return (
    <div className="bg-gray-900 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-400 mb-3">Trade P&L vs MFM (top 30)</h3>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
          <XAxis
            dataKey="name"
            tick={{ fill: "#6b7280", fontSize: 10 }}
            axisLine={{ stroke: "#374151" }}
          />
          <YAxis
            tick={{ fill: "#6b7280", fontSize: 10 }}
            axisLine={{ stroke: "#374151" }}
            tickFormatter={(v: number) => `${v}%`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#111827",
              border: "1px solid #374151",
              borderRadius: "8px",
              fontSize: "12px",
            }}
            formatter={(value: number, name: string) => [`${value}%`, name]}
          />
          <ReferenceLine y={0} stroke="#4b5563" />
          <Bar dataKey="mfm" fill="#3b82f620" name="MFM" />
          <Bar dataKey="pnl" name="P&L">
            {data.map((entry, index) => (
              <Cell
                key={index}
                fill={entry.pnl >= 0 ? "#10b981" : "#ef4444"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
