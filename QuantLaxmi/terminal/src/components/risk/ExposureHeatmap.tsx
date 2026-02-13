import { useEffect, useState } from "react";
import { apiFetch } from "@/lib/api";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ExposureCell {
  strategy: string;
  symbol: string;
  exposure: number; // positive = long, negative = short
}

interface ExposureData {
  cells: ExposureCell[];
  strategies: string[];
  symbols: string[];
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

const EMPTY_DATA: ExposureData = {
  strategies: [],
  symbols: [],
  cells: [],
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getExposure(
  cells: ExposureCell[],
  strategy: string,
  symbol: string,
): number {
  return cells.find((c) => c.strategy === strategy && c.symbol === symbol)?.exposure ?? 0;
}

function cellColor(exposure: number, maxAbs: number): string {
  if (exposure === 0) return "transparent";
  const intensity = Math.min(Math.abs(exposure) / maxAbs, 1);
  const alpha = (0.15 + intensity * 0.65).toFixed(2);
  return exposure > 0
    ? `rgba(0, 212, 170, ${alpha})`
    : `rgba(255, 77, 106, ${alpha})`;
}

function formatLakhs(v: number): string {
  if (v === 0) return "--";
  const sign = v > 0 ? "+" : "";
  return `${sign}${(v / 100000).toFixed(1)}L`;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ExposureHeatmap() {
  const [matrix, setMatrix] = useState<ExposureData>(EMPTY_DATA);

  useEffect(() => {
    let active = true;

    const fetchData = () => {
      apiFetch<{
        positions: Array<{
          strategy_id: string;
          symbol: string;
          direction: string;
          weight: number;
          current_price: number;
        }>;
      }>("/api/portfolio").then((portfolio) => {
        if (!active) return;
        const positions = portfolio.positions ?? [];
        const cells: ExposureCell[] = positions.map((p) => ({
          strategy: p.strategy_id,
          symbol: p.symbol,
          exposure: (p.direction === "BUY" ? 1 : -1) * p.weight * p.current_price,
        }));
        const strategies = [...new Set(cells.map((c) => c.strategy))];
        const symbols = [...new Set(cells.map((c) => c.symbol))];
        setMatrix({ cells, strategies, symbols });
      }).catch(() => {});
    };

    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => { active = false; clearInterval(interval); };
  }, []);

  const maxAbs = Math.max(
    ...matrix.cells.map((c) => Math.abs(c.exposure)),
    1,
  );

  // Row totals
  const rowTotal = (strategy: string) =>
    matrix.cells
      .filter((c) => c.strategy === strategy)
      .reduce((s, c) => s + c.exposure, 0);

  // Column totals
  const colTotal = (symbol: string) =>
    matrix.cells
      .filter((c) => c.symbol === symbol)
      .reduce((s, c) => s + c.exposure, 0);

  const grandTotal = matrix.cells.reduce((s, c) => s + c.exposure, 0);

  return (
    <div className="flex flex-col gap-2">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-terminal-muted px-1">
        Exposure Matrix — Strategy x Symbol
      </h3>
      {matrix.strategies.length === 0 ? (
        <div className="flex items-center justify-center h-24 text-terminal-muted text-xs font-mono">
          No exposure data
        </div>
      ) : (
      <div className="overflow-x-auto rounded border border-terminal-border">
        <table className="w-full text-xs font-mono border-collapse">
          <thead>
            <tr className="bg-terminal-surface">
              <th className="sticky left-0 z-10 bg-terminal-surface px-3 py-2 text-left text-terminal-muted font-medium border-b border-r border-terminal-border">
                Strategy
              </th>
              {matrix.symbols.map((sym) => (
                <th
                  key={sym}
                  className="px-3 py-2 text-center text-terminal-muted font-medium border-b border-terminal-border whitespace-nowrap"
                >
                  {sym}
                </th>
              ))}
              <th className="px-3 py-2 text-center text-terminal-accent font-semibold border-b border-l border-terminal-border">
                Total
              </th>
            </tr>
          </thead>
          <tbody>
            {matrix.strategies.map((strat) => {
              const rt = rowTotal(strat);
              return (
                <tr key={strat} className="hover:bg-terminal-surface/50 transition-colors">
                  <td className="sticky left-0 z-10 bg-terminal-panel px-3 py-1.5 text-terminal-text-secondary font-medium border-r border-b border-terminal-border whitespace-nowrap">
                    {strat}
                  </td>
                  {matrix.symbols.map((sym) => {
                    const exp = getExposure(matrix.cells, strat, sym);
                    return (
                      <td
                        key={sym}
                        className="px-3 py-1.5 text-center border-b border-terminal-border transition-colors"
                        style={{ backgroundColor: cellColor(exp, maxAbs) }}
                      >
                        <span
                          className={cn(
                            exp === 0
                              ? "text-terminal-muted/40"
                              : exp > 0
                                ? "text-terminal-profit"
                                : "text-terminal-loss",
                          )}
                        >
                          {formatLakhs(exp)}
                        </span>
                      </td>
                    );
                  })}
                  <td
                    className={cn(
                      "px-3 py-1.5 text-center font-semibold border-b border-l border-terminal-border",
                      rt > 0 ? "text-terminal-profit" : rt < 0 ? "text-terminal-loss" : "text-terminal-muted",
                    )}
                  >
                    {formatLakhs(rt)}
                  </td>
                </tr>
              );
            })}
          </tbody>
          <tfoot>
            <tr className="bg-terminal-surface">
              <td className="sticky left-0 z-10 bg-terminal-surface px-3 py-2 text-terminal-accent font-semibold border-r border-terminal-border">
                Total
              </td>
              {matrix.symbols.map((sym) => {
                const ct = colTotal(sym);
                return (
                  <td
                    key={sym}
                    className={cn(
                      "px-3 py-2 text-center font-semibold border-terminal-border",
                      ct > 0 ? "text-terminal-profit" : ct < 0 ? "text-terminal-loss" : "text-terminal-muted",
                    )}
                  >
                    {formatLakhs(ct)}
                  </td>
                );
              })}
              <td
                className={cn(
                  "px-3 py-2 text-center font-bold border-l border-terminal-border",
                  grandTotal > 0 ? "text-terminal-profit" : grandTotal < 0 ? "text-terminal-loss" : "text-terminal-muted",
                )}
              >
                {formatLakhs(grandTotal)}
              </td>
            </tr>
          </tfoot>
        </table>
      </div>
      )}
    </div>
  );
}
