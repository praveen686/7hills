import { useCallback, useEffect } from "react";
import { Command } from "cmdk";
import { useAtom, useSetAtom } from "jotai";
import {
  Layout,
  LayoutDashboard,
  BarChart3,
  ShoppingCart,
  XCircle,
  Search,
  Shield,
  FlaskConical,
  ArrowRightLeft,
  Monitor,
  Crosshair,
  TrendingUp,
  Layers,
  Settings,
  UserCircle,
} from "lucide-react";

import { activeWorkspaceAtom, symbolSearchOpenAtom } from "@/stores/workspace";
import type { WorkspaceId } from "@/stores/workspace";
import { pageAtom, type PageId } from "@/stores/auth";
import { selectedSymbolAtom } from "@/stores/market";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface CommandPaletteProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function CommandPalette({ open, onOpenChange }: CommandPaletteProps) {
  const [, setActiveWorkspace] = useAtom(activeWorkspaceAtom);
  const setSymbolSearch = useSetAtom(symbolSearchOpenAtom);
  const setSelectedSymbol = useSetAtom(selectedSymbolAtom);
  const setPage = useSetAtom(pageAtom);

  // Close on escape
  useEffect(() => {
    if (!open) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onOpenChange(false);
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [open, onOpenChange]);

  const close = useCallback(() => onOpenChange(false), [onOpenChange]);

  const navigateTo = useCallback(
    (page: PageId, ws?: WorkspaceId) => {
      setPage(page);
      if (ws) setActiveWorkspace(ws);
      close();
    },
    [setPage, setActiveWorkspace, close],
  );

  const switchWorkspace = useCallback(
    (ws: WorkspaceId) => {
      setPage("terminal");
      setActiveWorkspace(ws);
      close();
    },
    [setPage, setActiveWorkspace, close],
  );

  const openSymbolSearch = useCallback(() => {
    close();
    // Small delay so command palette unmounts first
    requestAnimationFrame(() => setSymbolSearch(true));
  }, [close, setSymbolSearch]);

  const selectQuickSymbol = useCallback(
    (symbol: string) => {
      setSelectedSymbol(symbol);
      close();
    },
    [setSelectedSymbol, close],
  );

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[20vh]">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={close}
      />

      {/* Palette */}
      <Command
        className="relative w-[560px] max-h-[420px] bg-terminal-surface border border-terminal-border-bright rounded-xl shadow-2xl overflow-hidden flex flex-col"
        loop
      >
        {/* Search input */}
        <div className="flex items-center gap-3 px-4 border-b border-terminal-border">
          <Search size={16} className="text-terminal-muted flex-shrink-0" />
          <Command.Input
            placeholder="Type a command or search..."
            className="flex-1 h-12 bg-transparent text-sm text-terminal-text placeholder:text-terminal-muted outline-none"
            autoFocus
          />
          <kbd className="kbd">ESC</kbd>
        </div>

        {/* Results */}
        <Command.List className="flex-1 overflow-y-auto p-2">
          <Command.Empty className="py-8 text-center text-sm text-terminal-muted">
            No results found.
          </Command.Empty>

          {/* --- Navigation --- */}
          <Command.Group
            heading={
              <span className="text-2xs font-medium text-terminal-muted uppercase tracking-wider px-2">
                Navigation
              </span>
            }
          >
            <CommandItem
              icon={<LayoutDashboard size={16} />}
              label="Dashboard"
              shortcut="Alt+0"
              onSelect={() => navigateTo("dashboard")}
            />
            <CommandItem
              icon={<Crosshair size={16} />}
              label="Trading Workspace"
              shortcut="Alt+1"
              onSelect={() => switchWorkspace("trading")}
            />
            <CommandItem
              icon={<TrendingUp size={16} />}
              label="Analysis Workspace"
              shortcut="Alt+2"
              onSelect={() => switchWorkspace("analysis")}
            />
            <CommandItem
              icon={<FlaskConical size={16} />}
              label="Backtest Workspace"
              shortcut="Alt+3"
              onSelect={() => switchWorkspace("backtest")}
            />
            <CommandItem
              icon={<Monitor size={16} />}
              label="Monitor Workspace"
              shortcut="Alt+4"
              onSelect={() => switchWorkspace("monitor")}
            />
            <CommandItem
              icon={<Layers size={16} />}
              label="Strategies"
              onSelect={() => navigateTo("strategies")}
            />
            <CommandItem
              icon={<Settings size={16} />}
              label="Settings"
              onSelect={() => navigateTo("settings")}
            />
            <CommandItem
              icon={<UserCircle size={16} />}
              label="Profile"
              onSelect={() => navigateTo("profile")}
            />
          </Command.Group>

          {/* --- Trading --- */}
          <Command.Group
            heading={
              <span className="text-2xs font-medium text-terminal-muted uppercase tracking-wider px-2">
                Trading
              </span>
            }
          >
            <CommandItem
              icon={<ShoppingCart size={16} className="text-terminal-profit" />}
              label="Buy Order"
              shortcut="B"
              onSelect={close}
            />
            <CommandItem
              icon={<ArrowRightLeft size={16} className="text-terminal-loss" />}
              label="Sell Order"
              shortcut="S"
              onSelect={close}
            />
            <CommandItem
              icon={<XCircle size={16} className="text-terminal-warning" />}
              label="Cancel All Orders"
              shortcut="Ctrl+Shift+X"
              onSelect={close}
            />
          </Command.Group>

          {/* --- Symbols --- */}
          <Command.Group
            heading={
              <span className="text-2xs font-medium text-terminal-muted uppercase tracking-wider px-2">
                Symbols
              </span>
            }
          >
            <CommandItem
              icon={<Search size={16} />}
              label="Search Symbols..."
              shortcut="Ctrl+L"
              onSelect={openSymbolSearch}
            />
            <CommandItem
              icon={<BarChart3 size={16} />}
              label="NIFTY"
              onSelect={() => selectQuickSymbol("NIFTY")}
            />
            <CommandItem
              icon={<BarChart3 size={16} />}
              label="BANKNIFTY"
              onSelect={() => selectQuickSymbol("BANKNIFTY")}
            />
            <CommandItem
              icon={<BarChart3 size={16} />}
              label="FINNIFTY"
              onSelect={() => selectQuickSymbol("FINNIFTY")}
            />
          </Command.Group>

          {/* --- Tools --- */}
          <Command.Group
            heading={
              <span className="text-2xs font-medium text-terminal-muted uppercase tracking-wider px-2">
                Tools
              </span>
            }
          >
            <CommandItem
              icon={<Shield size={16} />}
              label="Risk Dashboard"
              onSelect={() => switchWorkspace("monitor")}
            />
            <CommandItem
              icon={<FlaskConical size={16} />}
              label="Run Backtest"
              onSelect={() => switchWorkspace("backtest")}
            />
            <CommandItem
              icon={<Layers size={16} />}
              label="Reset Layout"
              onSelect={close}
            />
            <CommandItem
              icon={<Layout size={16} />}
              label="Toggle Fullscreen"
              shortcut="F11"
              onSelect={close}
            />
          </Command.Group>
        </Command.List>

        {/* Footer hint */}
        <div className="flex items-center justify-between px-4 py-2 border-t border-terminal-border text-2xs text-terminal-muted">
          <span>
            <kbd className="kbd mr-1">&uarr;</kbd>
            <kbd className="kbd mr-1">&darr;</kbd>
            to navigate
          </span>
          <span>
            <kbd className="kbd mr-1">Enter</kbd>
            to select
          </span>
          <span>
            <kbd className="kbd">Ctrl+K</kbd> to toggle
          </span>
        </div>
      </Command>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Single command item
// ---------------------------------------------------------------------------

function CommandItem({
  icon,
  label,
  shortcut,
  onSelect,
}: {
  icon: React.ReactNode;
  label: string;
  shortcut?: string;
  onSelect: () => void;
}) {
  return (
    <Command.Item
      value={label}
      onSelect={onSelect}
      className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-terminal-text-secondary cursor-pointer
                 data-[selected=true]:bg-terminal-panel data-[selected=true]:text-terminal-text
                 hover:bg-terminal-panel/60 transition-colors"
    >
      <span className="text-terminal-muted flex-shrink-0">{icon}</span>
      <span className="flex-1">{label}</span>
      {shortcut && <kbd className="kbd">{shortcut}</kbd>}
    </Command.Item>
  );
}
