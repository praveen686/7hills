import { useState, useMemo, useCallback } from "react";
import { useAtomValue } from "jotai";
import { apiFetch } from "@/lib/api";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
} from "@tanstack/react-table";
import {
  pendingOrdersAtom,
  filledOrdersAtom,
  allOrdersAtom,
  type Order,
  type OrderStatus,
} from "@/stores/trading";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type TabId = "open" | "executed" | "all";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatOrderTime(ts: string): string {
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

function statusStyle(status: OrderStatus): string {
  switch (status) {
    case "PENDING":
    case "OPEN":
      return "text-blue-400 bg-blue-500/15";
    case "FILLED":
      return "text-terminal-profit bg-terminal-profit/15";
    case "PARTIALLY_FILLED":
      return "text-terminal-warning bg-terminal-warning/15";
    case "CANCELLED":
      return "text-gray-400 bg-gray-500/15";
    case "REJECTED":
      return "text-terminal-loss bg-terminal-loss/15";
    default:
      return "text-gray-400 bg-gray-500/15";
  }
}

// ---------------------------------------------------------------------------
// Cancel button (isolated for hook scope)
// ---------------------------------------------------------------------------

function CancelButton({ orderId }: { orderId: string }) {
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState(false);

  const handleCancel = useCallback(async () => {
    setLoading(true);
    try {
      await apiFetch(`/api/trading/order/${orderId}`, { method: "DELETE" });
      setDone(true);
    } catch {
      // Error handled by backend toast
    } finally {
      setLoading(false);
    }
  }, [orderId]);

  if (done) {
    return <span className="text-2xs text-terminal-muted">Done</span>;
  }

  return (
    <button
      onClick={(e) => {
        e.stopPropagation();
        handleCancel();
      }}
      disabled={loading}
      className={cn(
        "px-2 py-0.5 rounded text-2xs font-mono transition-colors",
        "bg-terminal-loss/10 text-terminal-loss hover:bg-terminal-loss/20",
        "disabled:opacity-50",
      )}
    >
      {loading ? "..." : "Cancel"}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Column definitions
// ---------------------------------------------------------------------------

function buildColumns(showCancel: boolean): ColumnDef<Order>[] {
  const cols: ColumnDef<Order>[] = [
    {
      accessorKey: "timestamp",
      header: "Time",
      size: 70,
      cell: ({ getValue }) => (
        <span className="text-terminal-muted tabular-nums">
          {formatOrderTime(getValue<string>())}
        </span>
      ),
    },
    {
      accessorKey: "symbol",
      header: "Symbol",
      size: 90,
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
      accessorKey: "orderType",
      header: "Type",
      size: 55,
      cell: ({ getValue }) => (
        <span className="text-terminal-muted">{getValue<string>()}</span>
      ),
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
      cell: ({ getValue }) => {
        const p = getValue<number | null>();
        return (
          <span className="tabular-nums">{p != null ? formatPrice(p) : "MKT"}</span>
        );
      },
    },
    {
      accessorKey: "status",
      header: "Status",
      size: 75,
      cell: ({ getValue }) => {
        const status = getValue<OrderStatus>();
        return (
          <span
            className={cn(
              "text-2xs font-bold px-1.5 py-0.5 rounded",
              statusStyle(status),
            )}
          >
            {status}
          </span>
        );
      },
    },
  ];

  if (showCancel) {
    cols.push({
      id: "action",
      header: "Action",
      size: 60,
      cell: ({ row }) => {
        const st = row.original.status;
        if (st === "PENDING" || st === "OPEN") {
          return <CancelButton orderId={row.original.orderId} />;
        }
        return <span className="text-terminal-muted text-2xs">--</span>;
      },
    });
  }

  return cols;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function OrdersTable() {
  const pending = useAtomValue(pendingOrdersAtom);
  const filled = useAtomValue(filledOrdersAtom);
  const all = useAtomValue(allOrdersAtom);

  const [activeTab, setActiveTab] = useState<TabId>("open");
  const [sorting, setSorting] = useState<SortingState>([]);

  const data = useMemo(() => {
    switch (activeTab) {
      case "open":
        return pending;
      case "executed":
        return filled;
      case "all":
        return all;
    }
  }, [activeTab, pending, filled, all]);

  const columns = useMemo(
    () => buildColumns(activeTab === "open" || activeTab === "all"),
    [activeTab],
  );

  const table = useReactTable({
    data,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  const tabs: { id: TabId; label: string; count: number }[] = [
    { id: "open", label: "Open Orders", count: pending.length },
    { id: "executed", label: "Executed", count: filled.length },
    { id: "all", label: "All", count: all.length },
  ];

  return (
    <div className="flex flex-col h-full w-full bg-terminal-surface">
      {/* Tab bar */}
      <div className="flex items-center gap-0 px-1 border-b border-terminal-border">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={cn(
              "px-3 py-1.5 text-2xs font-mono transition-colors border-b-2",
              activeTab === tab.id
                ? "text-terminal-accent border-terminal-accent"
                : "text-terminal-muted border-transparent hover:text-gray-300",
            )}
          >
            {tab.label}
            <span className="ml-1 text-terminal-muted">({tab.count})</span>
          </button>
        ))}
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
                  {activeTab === "open" ? "No pending orders" : "No orders"}
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
    </div>
  );
}
