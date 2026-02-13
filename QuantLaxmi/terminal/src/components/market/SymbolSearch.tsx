import { useCallback } from "react";
import { Command } from "cmdk";
import { useAtom, useSetAtom } from "jotai";
import { Search, X, TrendingUp, Landmark, Bitcoin, Globe } from "lucide-react";

import { symbolSearchOpenAtom } from "@/stores/workspace";
import { selectedSymbolAtom } from "@/stores/market";

// ---------------------------------------------------------------------------
// Symbol catalog (static — will be replaced by Tauri invoke in production)
// ---------------------------------------------------------------------------

interface SymbolEntry {
  symbol: string;
  name: string;
  exchange: "NSE" | "BSE" | "BIN";
  type: "INDEX" | "EQUITY" | "FUTURES" | "OPTIONS" | "CRYPTO";
}

const SYMBOL_CATALOG: SymbolEntry[] = [
  // Indices
  { symbol: "NIFTY", name: "Nifty 50", exchange: "NSE", type: "INDEX" },
  { symbol: "BANKNIFTY", name: "Nifty Bank", exchange: "NSE", type: "INDEX" },
  { symbol: "FINNIFTY", name: "Nifty Financial Services", exchange: "NSE", type: "INDEX" },
  { symbol: "MIDCPNIFTY", name: "Nifty Midcap Select", exchange: "NSE", type: "INDEX" },

  // Equities (top FnO stocks)
  { symbol: "RELIANCE", name: "Reliance Industries", exchange: "NSE", type: "EQUITY" },
  { symbol: "HDFCBANK", name: "HDFC Bank", exchange: "NSE", type: "EQUITY" },
  { symbol: "TCS", name: "Tata Consultancy Services", exchange: "NSE", type: "EQUITY" },
  { symbol: "INFY", name: "Infosys", exchange: "NSE", type: "EQUITY" },
  { symbol: "ICICIBANK", name: "ICICI Bank", exchange: "NSE", type: "EQUITY" },
  { symbol: "SBIN", name: "State Bank of India", exchange: "NSE", type: "EQUITY" },
  { symbol: "BHARTIARTL", name: "Bharti Airtel", exchange: "NSE", type: "EQUITY" },
  { symbol: "ITC", name: "ITC Limited", exchange: "NSE", type: "EQUITY" },
  { symbol: "KOTAKBANK", name: "Kotak Mahindra Bank", exchange: "NSE", type: "EQUITY" },
  { symbol: "LT", name: "Larsen & Toubro", exchange: "NSE", type: "EQUITY" },
  { symbol: "AXISBANK", name: "Axis Bank", exchange: "NSE", type: "EQUITY" },
  { symbol: "BAJFINANCE", name: "Bajaj Finance", exchange: "NSE", type: "EQUITY" },
  { symbol: "MARUTI", name: "Maruti Suzuki", exchange: "NSE", type: "EQUITY" },
  { symbol: "TATAMOTORS", name: "Tata Motors", exchange: "NSE", type: "EQUITY" },
  { symbol: "SUNPHARMA", name: "Sun Pharma", exchange: "NSE", type: "EQUITY" },
  { symbol: "TATASTEEL", name: "Tata Steel", exchange: "NSE", type: "EQUITY" },
  { symbol: "WIPRO", name: "Wipro", exchange: "NSE", type: "EQUITY" },
  { symbol: "HCLTECH", name: "HCL Technologies", exchange: "NSE", type: "EQUITY" },
  { symbol: "ADANIENT", name: "Adani Enterprises", exchange: "NSE", type: "EQUITY" },
  { symbol: "POWERGRID", name: "Power Grid Corp", exchange: "NSE", type: "EQUITY" },

  // Crypto
  { symbol: "BTCUSDT", name: "Bitcoin / USDT", exchange: "BIN", type: "CRYPTO" },
  { symbol: "ETHUSDT", name: "Ethereum / USDT", exchange: "BIN", type: "CRYPTO" },
];

// ---------------------------------------------------------------------------
// Icon per instrument type
// ---------------------------------------------------------------------------

function TypeIcon({ type }: { type: SymbolEntry["type"] }) {
  switch (type) {
    case "INDEX":
      return <TrendingUp size={14} className="text-terminal-accent" />;
    case "EQUITY":
      return <Landmark size={14} className="text-terminal-info" />;
    case "CRYPTO":
      return <Bitcoin size={14} className="text-terminal-warning" />;
    default:
      return <Globe size={14} className="text-terminal-muted" />;
  }
}

// ---------------------------------------------------------------------------
// Badge colours per exchange
// ---------------------------------------------------------------------------

function ExchangeBadge({ exchange }: { exchange: SymbolEntry["exchange"] }) {
  const color =
    exchange === "NSE"
      ? "text-terminal-accent bg-terminal-accent/10"
      : exchange === "BIN"
        ? "text-terminal-warning bg-terminal-warning/10"
        : "text-terminal-muted bg-terminal-panel";

  return (
    <span className={`text-2xs font-mono px-1.5 py-0.5 rounded ${color}`}>
      {exchange}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function SymbolSearch() {
  const [open, setOpen] = useAtom(symbolSearchOpenAtom);
  const setSelectedSymbol = useSetAtom(selectedSymbolAtom);

  const close = useCallback(() => setOpen(false), [setOpen]);

  const handleSelect = useCallback(
    (symbol: string) => {
      setSelectedSymbol(symbol);
      close();
    },
    [setSelectedSymbol, close],
  );

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[18vh]">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={close}
      />

      {/* Dialog */}
      <Command
        className="relative w-[480px] max-h-[400px] bg-terminal-surface border border-terminal-border-bright rounded-xl shadow-2xl overflow-hidden flex flex-col"
        loop
      >
        {/* Search input */}
        <div className="flex items-center gap-3 px-4 border-b border-terminal-border">
          <Search size={16} className="text-terminal-muted flex-shrink-0" />
          <Command.Input
            placeholder="Search symbols..."
            className="flex-1 h-11 bg-transparent text-sm text-terminal-text placeholder:text-terminal-muted outline-none"
            autoFocus
          />
          <button
            onClick={close}
            className="p-1 rounded hover:bg-terminal-border text-terminal-muted hover:text-terminal-text-secondary transition-colors"
            aria-label="Close symbol search"
          >
            <X size={14} />
          </button>
        </div>

        {/* Results */}
        <Command.List className="flex-1 overflow-y-auto p-2">
          <Command.Empty className="py-8 text-center text-sm text-terminal-muted">
            No symbols found.
          </Command.Empty>

          {/* Indices */}
          <Command.Group
            heading={
              <span className="text-2xs font-medium text-terminal-muted uppercase tracking-wider px-2">
                Indices
              </span>
            }
          >
            {SYMBOL_CATALOG.filter((s) => s.type === "INDEX").map((entry) => (
              <SymbolItem
                key={entry.symbol}
                entry={entry}
                onSelect={handleSelect}
              />
            ))}
          </Command.Group>

          {/* Equities */}
          <Command.Group
            heading={
              <span className="text-2xs font-medium text-terminal-muted uppercase tracking-wider px-2">
                Equities
              </span>
            }
          >
            {SYMBOL_CATALOG.filter((s) => s.type === "EQUITY").map((entry) => (
              <SymbolItem
                key={entry.symbol}
                entry={entry}
                onSelect={handleSelect}
              />
            ))}
          </Command.Group>

          {/* Crypto */}
          <Command.Group
            heading={
              <span className="text-2xs font-medium text-terminal-muted uppercase tracking-wider px-2">
                Crypto
              </span>
            }
          >
            {SYMBOL_CATALOG.filter((s) => s.type === "CRYPTO").map((entry) => (
              <SymbolItem
                key={entry.symbol}
                entry={entry}
                onSelect={handleSelect}
              />
            ))}
          </Command.Group>
        </Command.List>

        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-2 border-t border-terminal-border text-2xs text-terminal-muted">
          <span>
            <kbd className="kbd mr-1">&uarr;</kbd>
            <kbd className="kbd">&darr;</kbd> navigate
          </span>
          <span>
            <kbd className="kbd">Enter</kbd> select
          </span>
          <span>
            <kbd className="kbd">Ctrl+L</kbd> toggle
          </span>
        </div>
      </Command>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Single symbol row
// ---------------------------------------------------------------------------

function SymbolItem({
  entry,
  onSelect,
}: {
  entry: SymbolEntry;
  onSelect: (symbol: string) => void;
}) {
  return (
    <Command.Item
      value={`${entry.symbol} ${entry.name}`}
      onSelect={() => onSelect(entry.symbol)}
      className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm cursor-pointer
                 data-[selected=true]:bg-terminal-panel data-[selected=true]:text-terminal-text
                 hover:bg-terminal-panel/60 transition-colors"
    >
      <TypeIcon type={entry.type} />
      <span className="font-semibold text-terminal-text-secondary w-28 truncate font-mono text-xs">
        {entry.symbol}
      </span>
      <span className="flex-1 text-xs text-terminal-muted truncate">{entry.name}</span>
      <ExchangeBadge exchange={entry.exchange} />
      <span className="text-2xs text-terminal-muted uppercase w-14 text-right">
        {entry.type}
      </span>
    </Command.Item>
  );
}
