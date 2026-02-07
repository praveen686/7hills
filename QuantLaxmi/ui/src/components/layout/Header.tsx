"use client";

import { useEffect, useState } from "react";
import { usePortfolio, useVIX } from "@/hooks/usePortfolio";
import { formatCompact, formatPct, pnlColor } from "@/lib/formatters";

// ============================================================
// Top Header Bar
// ============================================================

export function Header() {
  const { data: portfolio } = usePortfolio();
  const { data: vix } = useVIX();
  const [currentTime, setCurrentTime] = useState<string>("");
  const [marketOpen, setMarketOpen] = useState(false);

  useEffect(() => {
    const tick = () => {
      const now = new Date();
      setCurrentTime(
        now.toLocaleTimeString("en-IN", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: false,
          timeZone: "Asia/Kolkata",
        })
      );
      // 9:15 to 15:30 IST
      const hours = now.getHours();
      const minutes = now.getMinutes();
      const time = hours * 60 + minutes;
      setMarketOpen(time >= 555 && time <= 930);
    };
    tick();
    const interval = setInterval(tick, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="h-16 bg-gray-950 border-b border-gray-800 flex items-center justify-between px-6 sticky top-0 z-30">
      {/* Left: Market Status */}
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${
              marketOpen ? "bg-profit animate-pulse" : "bg-gray-600"
            }`}
          />
          <span className="text-xs text-gray-400 uppercase tracking-wide">
            {marketOpen ? "Market Open" : "Market Closed"}
          </span>
        </div>

        {/* VIX */}
        {vix && (
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gray-900 border border-gray-800">
            <span className="text-xs text-gray-500">VIX</span>
            <span className="text-sm font-mono font-semibold text-white">
              {vix.value.toFixed(2)}
            </span>
            <span className={`text-xs font-mono ${pnlColor(vix.change)}`}>
              {formatPct(vix.change_pct)}
            </span>
          </div>
        )}
      </div>

      {/* Center: Equity */}
      <div className="flex items-center gap-6">
        {portfolio && (
          <>
            <div className="text-center">
              <p className="text-[10px] text-gray-500 uppercase tracking-wider">
                Equity
              </p>
              <p className="text-sm font-mono font-semibold text-white">
                {formatCompact(portfolio.total_equity)}
              </p>
            </div>
            <div className="text-center">
              <p className="text-[10px] text-gray-500 uppercase tracking-wider">
                Day P&L
              </p>
              <p
                className={`text-sm font-mono font-semibold ${pnlColor(portfolio.day_pnl)}`}
              >
                {formatCompact(portfolio.day_pnl)}{" "}
                <span className="text-xs">
                  ({formatPct(portfolio.day_pnl_pct)})
                </span>
              </p>
            </div>
          </>
        )}
      </div>

      {/* Right: Time */}
      <div className="flex items-center gap-4">
        <div className="text-right">
          <p className="text-[10px] text-gray-500 uppercase tracking-wider">
            IST
          </p>
          <p className="text-sm font-mono font-semibold text-white tabular-nums">
            {currentTime}
          </p>
        </div>
      </div>
    </header>
  );
}
