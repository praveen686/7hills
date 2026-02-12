import { useState, useMemo, useCallback } from "react";
import { useAtomValue } from "jotai";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
} from "@tanstack/react-table";
import { filledOrdersAtom, type Order } from "@/stores/trading";
import { cn, formatPnl } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TradeRow {
  orderId: string;
  time: string;
  symbol: string;
  side: string;
  quantity: number;
  price: number;
  pnl: number;
  commission: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatTradeTime(ts: string): string {
  const d = new Date(ts);
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

function formatPrice(p: number): string {
  return p.toLocaleString("en-IN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

/** Convert filled orders into trade rows. */
function ordersToTrades(orders: Order[]): TradeRow[] {
  return orders
    .filter((o) => o.status === "FILLED")
    .map((o) => ({
      orderId: o.orderId,
      time: formatTradeTime(o.timestamp),
      symbol: o.symbol,
      side: o.side,
      quantity: o.quantity,
      price: o.price ?? 0,
      pnl: 0,
      commission: 0,
    }));
}

/** Export rows to CSV and trigger browser download. */
function exportToCSV(trades: TradeRow[]): void {
  const headers = ["Time", "Symbol", "Side", "Qty", "Price", "P&L", "Commission"];
  const rows = trades.map((t) =>
    [t.time, t.symbol, t.side, t.quantity, t.price.toFixed(2), t.pnl.toFixed(2), t.commission.toFixed(2)].join(","),
  );
  const csv = [headers.join(","), ...rows].join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `trades_${new Date().toISOString().slice(0, 10)}.csv`;
  link.click();
  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------------
// Columns
// ---------------------------------------------------------------------------

const columns: ColumnDef<TradeRow>[] = [
  {
    accessorKey: "time",
    header: "Time",
    size: 70,
    cell: ({ getValue }) => (
      <span className="text-terminal-muted tabular-nums">{getValue<string>()}</span>
    ),
  },
  {
    accessorKey: "symbol",
    header: "Symbol",
    size: 90,
    enableGlobalFilter: true,
    cell: ({ getValue }) => (
      <span className="font-semibold text-gray-100">{getValue<string>()}</span>
    ),
  },
  {
    accessorKey: "side",
    header: "Side",
    size: 50,
    cell: ({ getValue }) => {
      const side = getValue<string>();
      return (
        <span
          className={cn(
            "font-bold text-2xs px-1.5 py-0.5 rounded",
            side === "BUY"
              ? "text-terminal-profit bg-terminal-profit/15"
              : "text-terminal-loss bg-terminal-loss/15",
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
    size: 55,
    cell: ({ getValue }) => (
      <span className="tabular-nums">{getValue<number>().toLocaleString("en-IN")}</span>
    ),
  },
  {
    accessorKey: "price",
    header: "Price",
    size: 70,
    cell: ({ getValue }) => (
      <span className="tabular-nums">{formatPrice(getValue<number>())}</span>
    ),
  },
  {
    accessorKey: "pnl",
    header: "P&L",
    size: 80,
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
    accessorKey: "commission",
    header: "Comm",
    size: 60,
    cell: ({ getValue }) => (
      <span className="tabular-nums text-terminal-muted">
        {getValue<number>().toFixed(2)}
      </span>
    ),
  },
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function TradesTable() {
  const filledOrders = useAtomValue(filledOrdersAtom);
  const [sorting, setSorting] = useState<SortingState>([]);
  const [globalFilter, setGlobalFilter] = useState("");

  const trades = useMemo(() => ordersToTrades(filledOrders), [filledOrders]);

  const table = useReactTable({
    data: trades,
    columns,
    state: { sorting, globalFilter },
    onSortingChange: setSorting,
    onGlobalFilterChange: setGlobalFilter,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    globalFilterFn: "includesString",
  });

  const handleExport = useCallback(() => {
    exportToCSV(table.getFilteredRowModel().rows.map((r) => r.original));
  }, [table]);

  // Totals
  const { totalPnl, totalComm } = useMemo(() => {
    let pnl = 0;
    let comm = 0;
    for (const t of trades) {
      pnl += t.pnl;
      comm += t.commission;
    }
    return { totalPnl: pnl, totalComm: comm };
  }, [trades]);

  return (
    <div className="flex flex-col h-full w-full bg-terminal-surface">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-terminal-border">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono font-semibold text-terminal-accent">
            Trades
          </span>
          <span className="text-2xs text-terminal-muted">{trades.length} fills</span>
        </div>
        <div className="flex items-center gap-2">
          {/* Search filter */}
          <input
            type="text"
            value={globalFilter}
            onChange={(e) => setGlobalFilter(e.target.value)}
            placeholder="Filter..."
            className={cn(
              "w-24 px-2 py-0.5 rounded text-2xs font-mono",
              "bg-terminal-bg border border-terminal-border text-gray-200",
              "focus:outline-none focus:border-terminal-accent",
            )}
          />
          {/* Export CSV */}
          <button
            onClick={handleExport}
            className={cn(
              "px-2 py-0.5 rounded text-2xs font-mono transition-colors",
              "bg-terminal-panel border border-terminal-border text-terminal-muted",
              "hover:text-gray-200 hover:border-terminal-border-bright",
            )}
          >
            Export CSV
          </button>
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
                  No trades yet
                </td>
              </tr>
            ) : (
              table.getRowModel().rows.map((row) => (
                <tr
                  key={row.id}
                  className="border-b border-terminal-border/50 hover:bg-terminal-panel/60 transition-colors"
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
          <span className="text-terminal-muted">Net P&L</span>
          <span
            className={cn(
              "font-bold",
              totalPnl >= 0 ? "text-terminal-profit" : "text-terminal-loss",
            )}
          >
            {formatPnl(totalPnl)}
          </span>
        </div>
        <div className="flex items-center gap-4 text-2xs font-mono">
          <span className="text-terminal-muted">Commission</span>
          <span className="text-terminal-loss">{totalComm.toFixed(2)}</span>
        </div>
      </div>
    </div>
  );
}
