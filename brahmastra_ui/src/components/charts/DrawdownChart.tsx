"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { DrawdownPoint } from "@/lib/types";
import { formatPctUnsigned, formatDate } from "@/lib/formatters";

// ============================================================
// Drawdown Area Chart (inverted, red-themed)
// ============================================================

interface DrawdownChartProps {
  data: DrawdownPoint[];
  height?: number;
}

interface DDTooltipPayloadEntry {
  value: number;
}

function DDTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: DDTooltipPayloadEntry[];
  label?: string;
}) {
  if (!active || !payload?.length || !label) return null;

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 shadow-xl">
      <p className="text-xs text-gray-400 mb-1">{formatDate(label)}</p>
      <p className="text-sm font-mono text-loss">
        DD: -{formatPctUnsigned(Math.abs(payload[0].value))}
      </p>
    </div>
  );
}

export function DrawdownChart({ data, height = 200 }: DrawdownChartProps) {
  if (!data || data.length === 0) {
    return (
      <div
        className="flex items-center justify-center text-gray-600 text-sm"
        style={{ height }}
      >
        No drawdown data available
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
        <defs>
          <linearGradient id="ddGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#ef4444" stopOpacity={0.4} />
            <stop offset="100%" stopColor="#ef4444" stopOpacity={0.05} />
          </linearGradient>
        </defs>

        <CartesianGrid
          strokeDasharray="3 3"
          stroke="#1e293b"
          vertical={false}
        />

        <XAxis
          dataKey="date"
          tick={{ fill: "#6b7280", fontSize: 11 }}
          tickLine={false}
          axisLine={{ stroke: "#1e293b" }}
          tickFormatter={(val: string) => {
            const d = new Date(val);
            return `${d.getDate()}/${d.getMonth() + 1}`;
          }}
          interval="preserveStartEnd"
          minTickGap={40}
        />

        <YAxis
          tick={{ fill: "#6b7280", fontSize: 11 }}
          tickLine={false}
          axisLine={false}
          tickFormatter={(val: number) => `${val.toFixed(1)}%`}
          width={50}
          domain={["dataMin", 0]}
        />

        <Tooltip content={<DDTooltip />} />

        <Area
          type="monotone"
          dataKey="drawdown"
          stroke="#ef4444"
          strokeWidth={1.5}
          fill="url(#ddGradient)"
          dot={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
