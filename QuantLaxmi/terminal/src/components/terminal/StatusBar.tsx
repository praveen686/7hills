import { useEffect, useRef, useState } from "react";
import { useAtomValue, useAtom } from "jotai";
import { Wifi, WifiOff, Gauge, User, LogOut } from "lucide-react";

import {
  connectionAtom,
  regimeAtom,
  activeWorkspaceAtom,
  type ConnectionStatus,
} from "@/stores/workspace";
import { selectedVpinAtom, selectedSymbolAtom } from "@/stores/market";
import { appModeAtom, type AppMode } from "@/stores/mode";
import { pageAtom } from "@/stores/auth";
import { cn } from "@/lib/utils";
import { ThemeToggle } from "@/components/terminal/ThemeToggle";
import { useAuth } from "@/hooks/useAuth";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Status dot CSS class from connection state */
function dotClass(status: ConnectionStatus): string {
  switch (status) {
    case "connected":
      return "status-dot status-connected";
    case "connecting":
      return "status-dot status-connecting";
    case "disconnected":
      return "status-dot status-disconnected";
  }
}

/** Format IST time as HH:MM:SS */
function formatIST(date: Date): string {
  return date.toLocaleTimeString("en-IN", {
    timeZone: "Asia/Kolkata",
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

/** VPIN colour based on threshold */
function vpinColor(vpin: number | null): string {
  if (vpin === null) return "text-terminal-muted";
  if (vpin >= 0.8) return "text-terminal-loss";
  if (vpin >= 0.6) return "text-terminal-warning";
  return "text-terminal-profit";
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/** Build a tooltip string summarizing user's trading access */
function userTooltip(u: import("@/stores/auth").User): string {
  const lines: string[] = [u.name];
  if (u.email) lines.push(u.email);
  if (u.provider === "zerodha") {
    if (u.exchanges?.length) lines.push(`Exchanges: ${u.exchanges.join(", ")}`);
    if (u.products?.length) lines.push(`Products: ${u.products.join(", ")}`);
    if (u.order_types?.length) lines.push(`Order types: ${u.order_types.join(", ")}`);
  } else if (u.provider === "binance") {
    if (u.permissions?.length) lines.push(`Permissions: ${u.permissions.join(", ")}`);
    if (u.account_type) lines.push(`Account: ${u.account_type}`);
    lines.push(`Trade: ${u.can_trade ? "Yes" : "No"} | Withdraw: ${u.can_withdraw ? "Yes" : "No"}`);
  }
  return lines.join("\n");
}

export function StatusBar() {
  const connection = useAtomValue(connectionAtom);
  const regime = useAtomValue(regimeAtom);
  const workspace = useAtomValue(activeWorkspaceAtom);
  const vpin = useAtomValue(selectedVpinAtom);
  const symbol = useAtomValue(selectedSymbolAtom);
  const [mode, setMode] = useAtom(appModeAtom);
  const page = useAtomValue(pageAtom);
  const { user, logout } = useAuth();
  const [clock, setClock] = useState(() => formatIST(new Date()));
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Tick the clock every second
  useEffect(() => {
    timerRef.current = setInterval(() => {
      setClock(formatIST(new Date()));
    }, 1000);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const toggleMode = () => {
    setMode((prev: AppMode) => (prev === "live" ? "backtest" : "live"));
  };

  return (
    <div className="h-7 flex items-center justify-between px-3 bg-terminal-panel border-t border-terminal-border text-2xs font-mono select-none flex-shrink-0">
      {/* Left: connection indicators */}
      <div className="flex items-center gap-4">
        {/* Zerodha */}
        <div className="flex items-center gap-1.5" title={`Zerodha: ${connection.zerodha}`}>
          <span className={dotClass(connection.zerodha)} />
          <span className="text-terminal-muted">ZRD</span>
        </div>

        {/* Binance */}
        <div className="flex items-center gap-1.5" title={`Binance: ${connection.binance}`}>
          <span className={dotClass(connection.binance)} />
          <span className="text-terminal-muted">BIN</span>
        </div>

        {/* FastAPI */}
        <div className="flex items-center gap-1.5" title={`FastAPI: ${connection.fastapi}`}>
          <span className={dotClass(connection.fastapi)} />
          <span className="text-terminal-muted">API</span>
        </div>

        {/* Latency */}
        <div className="flex items-center gap-1 text-terminal-muted" title="Round-trip latency">
          {connection.latencyMs > 0 ? (
            <Wifi size={11} className="text-terminal-profit" />
          ) : (
            <WifiOff size={11} className="text-terminal-loss" />
          )}
          <span>
            {connection.latencyMs > 0 ? `${connection.latencyMs}ms` : "--"}
          </span>
        </div>

        {/* Engine mode badge */}
        {connection.engineRunning && (
          <div className="flex items-center gap-2">
            <span
              className={cn(
                "px-1.5 py-0.5 rounded text-2xs font-mono font-bold uppercase tracking-wider",
                connection.mode === "live"
                  ? "bg-terminal-profit/20 text-terminal-profit border border-terminal-profit/40"
                  : "bg-yellow-500/20 text-yellow-400 border border-yellow-500/40",
              )}
              title={`Engine mode: ${connection.mode}`}
            >
              {connection.mode === "live" ? "LIVE" : "PAPER"}
            </span>
            <span className="text-terminal-muted tabular-nums" title="ticks / bars / signals">
              {(connection.ticksReceived ?? 0).toLocaleString()}t
              {" "}{connection.barsCompleted ?? 0}b
              {" "}{connection.signalsEmitted ?? 0}s
            </span>
          </div>
        )}
      </div>

      {/* Centre: mode toggle + VPIN + regime */}
      <div className="flex items-center gap-5">
        {/* Mode toggle */}
        <button
          onClick={toggleMode}
          className={cn(
            "px-2 py-0.5 rounded text-2xs font-mono font-bold uppercase tracking-wider transition-colors",
            mode === "live"
              ? "bg-terminal-profit/20 text-terminal-profit border border-terminal-profit/40"
              : "bg-blue-500/20 text-blue-400 border border-blue-500/40",
          )}
          title={`Mode: ${mode}. Click to toggle.`}
        >
          {mode}
        </button>

        {/* VPIN gauge */}
        <div className="flex items-center gap-1.5" title={`VPIN (${symbol})`}>
          <Gauge size={11} className={vpinColor(vpin)} />
          <span className="text-terminal-muted">VPIN</span>
          <span className={`${vpinColor(vpin)} tabular-nums`}>
            {vpin !== null ? vpin.toFixed(2) : "--"}
          </span>
        </div>

        {/* Regime */}
        <div className="flex items-center gap-1.5" title="Market regime">
          <span className="text-terminal-muted">REGIME</span>
          <span className="text-terminal-info">{regime}</span>
        </div>
      </div>

      {/* Right: user info + theme toggle + workspace name + clock */}
      <div className="flex items-center gap-4">
        {user && (
          <div
            className="flex items-center gap-1.5 cursor-default"
            title={userTooltip(user)}
          >
            <User size={11} className="text-terminal-accent" />
            <span className="text-terminal-text">{user.name.split(" ")[0]}</span>
            <span className={cn(
              "px-1 rounded text-2xs font-bold uppercase",
              user.provider === "zerodha" && "bg-[#387ed1]/20 text-[#5a9cf5]",
              user.provider === "binance" && "bg-[#F0B90B]/20 text-[#F0B90B]",
              user.provider === "google" && "bg-white/10 text-white",
            )}>
              {user.provider === "zerodha" ? "ZRD" : user.provider === "binance" ? "BIN" : "G"}
            </span>
            {user.provider === "zerodha" && user.exchanges && (
              <span className="text-terminal-muted">
                {user.exchanges.filter(e => ["NFO", "NSE", "MCX"].includes(e)).join("/")}
              </span>
            )}
            {user.provider === "binance" && user.permissions && (
              <span className="text-terminal-muted">
                {user.permissions.join("/")}
              </span>
            )}
            <button
              onClick={logout}
              className="text-terminal-muted hover:text-terminal-loss transition-colors ml-1"
              title="Sign out"
            >
              <LogOut size={10} />
            </button>
          </div>
        )}
        <ThemeToggle />
        <span className="text-terminal-accent uppercase">
          {page === "terminal" ? workspace : page}
        </span>
        <span className="text-terminal-text-secondary tabular-nums">{clock} IST</span>
      </div>
    </div>
  );
}
