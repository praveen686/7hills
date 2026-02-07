"use client";

import { useState } from "react";
import {
  useTradeAnalytics,
  useTradeAnalyticsSummary,
  useMissedOpportunities,
  useARSSurface,
  useDiagnosticsTab,
} from "@/hooks/useDiagnostics";
import { DiagnosticsSummary } from "@/components/diagnostics/DiagnosticsSummary";
import { OpportunityCard } from "@/components/diagnostics/OpportunityCard";
import { TradeOutcomeChart } from "@/components/diagnostics/TradeOutcomeChart";
import { MissedOpportunityTable } from "@/components/diagnostics/MissedOpportunityTable";
import { ARSHeatmapChart } from "@/components/diagnostics/ARSHeatmap";

const TABS = [
  { id: "analytics" as const, label: "Trade Analytics" },
  { id: "missed" as const, label: "Missed Opportunities" },
  { id: "ars" as const, label: "ARS Surface" },
];

export default function DiagnosticsPage() {
  const { tab, setTab } = useDiagnosticsTab();
  const [strategyFilter, setStrategyFilter] = useState<string>("");
  const [missedDate, setMissedDate] = useState<string>("");

  const analyticsQuery = useTradeAnalytics(strategyFilter || undefined);
  const summaryQuery = useTradeAnalyticsSummary();
  const missedQuery = useMissedOpportunities(
    missedDate || undefined,
    undefined,
    undefined,
    strategyFilter || undefined,
  );
  const arsQuery = useARSSurface();

  // Collect strategy IDs from analytics data
  const strategyIds = analyticsQuery.data
    ? [...new Set(analyticsQuery.data.map((t) => t.strategy_id))].sort()
    : [];

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Opportunity Diagnostics</h1>
        <p className="text-sm text-gray-500 mt-1">
          Trade efficiency, missed opportunities, and strategy activation analysis
        </p>
      </div>

      {/* Summary Cards */}
      <DiagnosticsSummary summary={summaryQuery.data} isLoading={summaryQuery.isLoading} />

      {/* Tabs + Filter */}
      <div className="flex items-center justify-between">
        <div className="flex gap-1 bg-gray-900 rounded-lg p-1">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-4 py-2 text-sm rounded-md transition-colors ${
                tab === t.id
                  ? "bg-accent text-white"
                  : "text-gray-400 hover:text-gray-200"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-3">
          {tab === "missed" && (
            <input
              type="date"
              value={missedDate}
              onChange={(e) => setMissedDate(e.target.value)}
              className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-300 focus:outline-none focus:ring-1 focus:ring-accent"
            />
          )}
          <select
            value={strategyFilter}
            onChange={(e) => setStrategyFilter(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-300 focus:outline-none focus:ring-1 focus:ring-accent"
          >
            <option value="">All Strategies</option>
            {strategyIds.map((sid) => (
              <option key={sid} value={sid}>{sid}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Tab Content */}
      {tab === "analytics" && (
        <div className="space-y-6">
          <TradeOutcomeChart trades={analyticsQuery.data ?? []} />
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {analyticsQuery.isLoading ? (
              [...Array(6)].map((_, i) => (
                <div key={i} className="bg-gray-900 rounded-xl p-4 animate-pulse h-40" />
              ))
            ) : analyticsQuery.data && analyticsQuery.data.length > 0 ? (
              analyticsQuery.data.map((trade) => (
                <OpportunityCard key={trade.trade_id} trade={trade} />
              ))
            ) : (
              <div className="col-span-full bg-gray-900 rounded-xl p-6 text-center text-gray-500">
                No trade analytics available
              </div>
            )}
          </div>
        </div>
      )}

      {tab === "missed" && (
        <MissedOpportunityTable
          opportunities={missedQuery.data}
          isLoading={missedQuery.isLoading}
        />
      )}

      {tab === "ars" && (
        <ARSHeatmapChart heatmap={arsQuery.data} isLoading={arsQuery.isLoading} />
      )}
    </div>
  );
}
