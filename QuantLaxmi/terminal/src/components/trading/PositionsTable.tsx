import { useMemo, useCallback, useState } from "react";
import { useAtomValue, useSetAtom } from "jotai";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
} from "@tanstack/react-table";
import { positionsAtom, type Position } from "@/stores/portfolio";
import { selectedSymbolAtom } from "@/stores/market";
import { cn, formatPnl, formatINR } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatPrice(p: number): string {
  return p.toLocaleString("en-IN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

// ---------------------------------------------------------------------------
// Columns
// ---------------------------------------------------------------------------

const columns: ColumnDef<Position>[] = [
  {
    accessorKey: "symbol",
    header: "Symbol",
    size: 100,
    cell: ({ getValue }) => (
      <span className="font-semibold text-gray-100">{getValue<string>()}</span>
    ),
  },
  {
    accessorKey: "side",
    header: "Side",
    size: 60,
    cell: ({ getValue }) => {
      const side = getValue<"LONG" | "SHORT">();
      return (
        <span
          className={cn(
            "text-2xs font-bold px-1.5 py-0.5 rounded",
            side === "LONG"
              ? "text-terminal-profit bg-terminal-profit/10"
              : "text-terminal-loss bg-terminal-loss/10",
          )}
        >
          {side}
        </span>
      );
    },
  },
  {
    accessorKey: "quantity",
    header: "Qty",
    size: 60,
    cell: ({ getValue }) => (
      <span className="tabular-nums">{getValue<number>().toLocaleString("en-IN")}</span>
    ),
  },
  {
    accessorKey: "avgPrice",
    header: "Avg Price",
    size: 80,
    cell: ({ getValue }) => (
      <span className="tabular-nums">{formatPrice(getValue<number>())}</span>
    ),
  },
  {
    accessorKey: "ltp",
    header: "LTP",
    size: 80,
    cell: ({ getValue }) => (
      <span className="tabular-nums text-gray-100 font-semibold">
        {formatPrice(getValue<number>())}
      </span>
    ),
  },
  {
    accessorKey: "pnl",
    header: "P&L",
    size: 90,
    cell: ({ getValue }) => {
      const pnl = getValue<number>();
      return (
        <span
          className={cn(
            "tabular-nums font-semibold",
            pnl >= 0 ? "text-terminal-profit" : "text-terminal-loss",
          )}
        >
          {formatPnl(pnl)}
        </span>
      );
    },
    sortingFn: "basic",
  },
  {
    accessorKey: "pnlPct",
    header: "P&L%",
    size: 70,
    cell: ({ getValue }) => {
      const pct = getValue<number>();
      return (
        <span
          className={cn(
            "tabular-nums",
            pct >= 0 ? "text-terminal-profit" : "text-terminal-loss",
          )}
        >
          {formatPnl(pct)}%
        </span>
      );
    },
    sortingFn: "basic",
  },
  {
    accessorKey: "strategyId",
    header: "Strategy",
    size: 80,
    cell: ({ getValue }) => (
      <span className="text-terminal-muted text-2xs">{getValue<string>() || "--"}</span>
    ),
  },
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function PositionsTable() {
  const positions = useAtomValue(positionsAtom);
  const setSelectedSymbol = useSetAtom(selectedSymbolAtom);
  const [sorting, setSorting] = useState<SortingState>([]);

  const table = useReactTable({
    data: positions,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  const handleRowClick = useCallback(
    (symbol: string) => {
      setSelectedSymbol(symbol);
    },
    [setSelectedSymbol],
  );

  // Footer totals
  const { totalPnl, totalExposure } = useMemo(() => {
    let pnl = 0;
    let exposure = 0;
    for (const p of positions) {
      pnl += p.pnl;
      exposure += Math.abs(p.quantity * p.ltp);
    }
    return { totalPnl: pnl, totalExposure: exposure };
  }, [positions]);

  return (
    <div className="flex flex-col h-full w-full bg-terminal-surface">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-terminal-border">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono font-semibold text-terminal-accent">
            Positions
          </span>
          <span className="text-2xs text-terminal-muted">
            {positions.length} open
          </span>
        </div>
      </div>

      {/* Table */}
      <div className="flex-1 min-h-0 overflow-auto">
        <table className="w-full text-xs font-mono">
          <thead className="sticky top-0 bg-terminal-panel z-10">
            {table.getHeaderGroups().map((hg) => (
              <tr key={hg.id}>
                {hg.headers.map((header) => (
                  <th
                    key={header.id}
                    className={cn(
                      "px-2 py-1.5 text-left text-2xs text-terminal-muted font-normal",
                      "border-b border-terminal-border select-none",
                      header.column.getCanSort() && "cursor-pointer hover:text-gray-300",
                    )}
                    style={{ width: header.getSize() }}
                    onClick={header.column.getToggleSortingHandler()}
                  >
                    <div className="flex items-center gap-1">
                      {flexRender(header.column.columnDef.header, header.getContext())}
                      {{
                        asc: " ^",
                        desc: " v",
                      }[header.column.getIsSorted() as string] ?? ""}
                    </div>
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.length === 0 ? (
              <tr>
                <td
                  colSpan={columns.length}
                  className="px-4 py-8 text-center text-terminal-muted text-xs"
                >
                  No open positions
                </td>
              </tr>
            ) : (
              table.getRowModel().rows.map((row) => (
                <tr
                  key={row.id}
                  onClick={() => handleRowClick(row.original.symbol)}
                  className={cn(
                    "cursor-pointer border-b border-terminal-border/50",
                    "hover:bg-terminal-panel/60 transition-colors",
                  )}
                >
                  {row.getVisibleCells().map((cell) => (
                    <td
                      key={cell.id}
                      className="px-2 py-1.5"
                      style={{ width: cell.column.getSize() }}
                    >
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Footer totals */}
      <div className="flex items-center justify-between px-3 py-1.5 border-t border-terminal-border bg-terminal-panel">
        <div className="flex items-center gap-4 text-2xs font-mono">
          <span className="text-terminal-muted">Total P&L</span>
          <span
            className={cn(
              "font-bold",
              totalPnl >= 0 ? "text-terminal-profit" : "text-terminal-loss",
            )}
          >
            {formatINR(totalPnl)}
          </span>
        </div>
        <div className="flex items-center gap-4 text-2xs font-mono">
          <span className="text-terminal-muted">Exposure</span>
          <span className="text-gray-200">{formatINR(totalExposure)}</span>
        </div>
      </div>
    </div>
  );
}
