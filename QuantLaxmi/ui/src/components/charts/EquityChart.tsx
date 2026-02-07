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
import type { EquityPoint } from "@/lib/types";
import { formatCompact, formatDate } from "@/lib/formatters";

// ============================================================
// Equity Curve Area Chart
// ============================================================

interface EquityChartProps {
  data: EquityPoint[];
  height?: number;
  showBenchmark?: boolean;
  showGrid?: boolean;
}

interface TooltipPayloadEntry {
  name: string;
  value: number;
  color: string;
}

function CustomTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: TooltipPayloadEntry[];
  label?: string;
}) {
  if (!active || !payload?.length || !label) return null;

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 shadow-xl">
      <p className="text-xs text-gray-400 mb-1">{formatDate(label)}</p>
      {payload.map((entry, index) => (
        <p key={index} className="text-sm font-mono" style={{ color: entry.color }}>
          {entry.name}: {formatCompact(entry.value)}
        </p>
      ))}
    </div>
  );
}

export function EquityChart({
  data,
  height = 300,
  showBenchmark = false,
  showGrid = true,
}: EquityChartProps) {
  if (!data || data.length === 0) {
    return (
      <div
        className="flex items-center justify-center text-gray-600 text-sm"
        style={{ height }}
      >
        No equity data available
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
        <defs>
          <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.3} />
            <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
          </linearGradient>
          <linearGradient id="benchmarkGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#6b7280" stopOpacity={0.2} />
            <stop offset="100%" stopColor="#6b7280" stopOpacity={0} />
          </linearGradient>
        </defs>

        {showGrid && (
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#1e293b"
            vertical={false}
          />
        )}

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
          tickFormatter={(val: number) => formatCompact(val)}
          width={70}
          domain={["auto", "auto"]}
        />

        <Tooltip content={<CustomTooltip />} />

        {showBenchmark && (
          <Area
            type="monotone"
            dataKey="benchmark"
            name="Benchmark"
            stroke="#6b7280"
            strokeWidth={1}
            fill="url(#benchmarkGradient)"
            dot={false}
          />
        )}

        <Area
          type="monotone"
          dataKey="equity"
          name="Equity"
          stroke="#3b82f6"
          strokeWidth={2}
          fill="url(#equityGradient)"
          dot={false}
          activeDot={{ r: 4, fill: "#3b82f6", stroke: "#1e293b", strokeWidth: 2 }}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
