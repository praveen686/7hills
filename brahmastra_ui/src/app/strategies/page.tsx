"use client";

import { useStrategies } from "@/hooks/usePortfolio";
import { StrategyCard } from "@/components/charts/StrategyCard";
import { useState } from "react";
import { clsx } from "clsx";
import type { StrategyStatus } from "@/lib/types";

// ============================================================
// Strategies List Page
// ============================================================

const STATUS_FILTERS: { label: string; value: StrategyStatus | "all" }[] = [
  { label: "All", value: "all" },
  { label: "Live", value: "live" },
  { label: "Paused", value: "paused" },
  { label: "Stopped", value: "stopped" },
  { label: "Backtest", value: "backtest" },
];

export default function StrategiesPage() {
  const { data: strategies, isLoading } = useStrategies();
  const [filter, setFilter] = useState<StrategyStatus | "all">("all");
  const [search, setSearch] = useState("");

  const filtered = strategies?.filter((s) => {
    const matchesFilter = filter === "all" || s.status === filter;
    const sid = s.strategy_id ?? s.id ?? "";
    const matchesSearch =
      search === "" ||
      s.name.toLowerCase().includes(search.toLowerCase()) ||
      sid.toLowerCase().includes(search.toLowerCase());
    return matchesFilter && matchesSearch;
  });

  return (
    <div className="space-y-6 animate-in">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-white">Strategies</h1>
          <p className="text-sm text-gray-500 mt-1">
            {strategies?.length ?? 0} strategies deployed
          </p>
        </div>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-4">
        {/* Status Filter Tabs */}
        <div className="flex gap-1 bg-gray-900 rounded-lg p-1">
          {STATUS_FILTERS.map(({ label, value }) => (
            <button
              key={value}
              onClick={() => setFilter(value)}
              className={clsx(
                "px-3 py-1.5 text-xs font-medium rounded-md transition-colors",
                filter === value
                  ? "bg-gray-800 text-white"
                  : "text-gray-500 hover:text-gray-300"
              )}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Search */}
        <input
          type="text"
          placeholder="Search strategies..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="input-field max-w-xs"
        />
      </div>

      {/* Strategy Grid */}
      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="card animate-pulse h-48" />
          ))}
        </div>
      ) : filtered && filtered.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered.map((strategy) => (
            <StrategyCard key={strategy.strategy_id ?? strategy.id} strategy={strategy} />
          ))}
        </div>
      ) : (
        <div className="card text-center py-16">
          <svg
            className="w-12 h-12 text-gray-700 mx-auto mb-3"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0112 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5"
            />
          </svg>
          <p className="text-sm text-gray-500">No strategies found</p>
          <p className="text-xs text-gray-600 mt-1">
            {search ? "Try a different search term" : "Deploy a strategy to get started"}
          </p>
        </div>
      )}
    </div>
  );
}
