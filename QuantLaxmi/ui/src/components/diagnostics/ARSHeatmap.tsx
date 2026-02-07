"use client";

import type { ARSHeatmap } from "@/lib/types";

interface Props {
  heatmap: ARSHeatmap | undefined;
  isLoading: boolean;
}

function convictionColor(conviction: number): string {
  if (conviction <= 0) return "bg-gray-900";
  if (conviction < 0.3) return "bg-blue-950";
  if (conviction < 0.5) return "bg-blue-900";
  if (conviction < 0.7) return "bg-blue-800";
  if (conviction < 0.9) return "bg-blue-700";
  return "bg-blue-600";
}

function statusDot(status: string): string {
  if (status === "executed") return "bg-profit";
  if (status === "blocked") return "bg-loss";
  return "";
}

export function ARSHeatmapChart({ heatmap, isLoading }: Props) {
  if (isLoading) {
    return <div className="bg-gray-900 rounded-xl p-6 animate-pulse h-64" />;
  }

  if (!heatmap || heatmap.dates.length === 0 || heatmap.strategies.length === 0) {
    return (
      <div className="bg-gray-900 rounded-xl p-6 text-center text-gray-500">
        No ARS surface data available
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-400 mb-3">
        Strategy Activation Surface (conviction intensity + execution status)
      </h3>
      <div className="overflow-x-auto">
        <table className="text-xs">
          <thead>
            <tr>
              <th className="px-2 py-1 text-left text-gray-500 sticky left-0 bg-gray-900 z-10">
                Strategy
              </th>
              {heatmap.dates.map((d) => (
                <th key={d} className="px-1 py-1 text-gray-600 font-normal whitespace-nowrap">
                  {d.slice(5)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {heatmap.strategies.map((sid, sIdx) => (
              <tr key={sid}>
                <td className="px-2 py-1 text-gray-300 font-medium sticky left-0 bg-gray-900 z-10 whitespace-nowrap">
                  {sid}
                </td>
                {heatmap.dates.map((d, dIdx) => {
                  const conviction = heatmap.matrix[sIdx]?.[dIdx] ?? 0;
                  const status = heatmap.status_matrix[sIdx]?.[dIdx] ?? "none";
                  return (
                    <td key={d} className="px-1 py-1">
                      <div
                        className={`w-8 h-6 rounded-sm ${convictionColor(conviction)} flex items-center justify-center relative`}
                        title={`${sid} ${d}: conviction=${conviction.toFixed(2)} status=${status}`}
                      >
                        {conviction > 0 && (
                          <span className="text-[8px] text-gray-300 font-mono">
                            {(conviction * 10).toFixed(0)}
                          </span>
                        )}
                        {status !== "none" && (
                          <span
                            className={`absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full ${statusDot(status)}`}
                          />
                        )}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {/* Legend */}
      <div className="flex items-center gap-4 mt-3 text-[10px] text-gray-500">
        <div className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-profit" /> Executed
        </div>
        <div className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-loss" /> Blocked
        </div>
        <div className="flex items-center gap-1">
          <span className="w-4 h-3 rounded-sm bg-blue-600" /> High conviction
        </div>
        <div className="flex items-center gap-1">
          <span className="w-4 h-3 rounded-sm bg-blue-950" /> Low conviction
        </div>
      </div>
    </div>
  );
}
