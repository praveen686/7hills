import { useState, useCallback, useEffect, useRef } from "react";
import { useAtomValue } from "jotai";
import { apiFetch } from "@/lib/api";
import { selectedSymbolAtom, selectedTickAtom } from "@/stores/market";
import { cn } from "@/lib/utils";
import type { OrderSide } from "@/stores/trading";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface OrderResponse {
  orderId: string;
  status: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const LOT_SIZES: Record<string, number> = {
  NIFTY: 25,
  BANKNIFTY: 15,
  FINNIFTY: 25,
  MIDCPNIFTY: 50,
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function QuickTrade() {
  const symbol = useAtomValue(selectedSymbolAtom);
  const tick = useAtomValue(selectedTickAtom);

  const [quantity, setQuantity] = useState(1);
  const [pendingSide, setPendingSide] = useState<OrderSide | null>(null);
  const [status, setStatus] = useState<{ type: "ok" | "err"; msg: string } | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const qtyRef = useRef<HTMLInputElement>(null);

  const ltp = tick?.ltp ?? 0;
  const lotSize = LOT_SIZES[symbol] ?? 1;

  // Submit order
  const submitOrder = useCallback(
    async (side: OrderSide) => {
      if (submitting) return;
      setStatus(null);
      setSubmitting(true);
      try {
        const result = await apiFetch<OrderResponse>("/api/trading/order", {
          method: "POST",
          body: JSON.stringify({
            symbol,
            side,
            orderType: "MARKET",
            quantity: quantity * lotSize,
            price: null,
            triggerPrice: null,
          }),
        });
        setStatus({ type: "ok", msg: `${side} ${result.orderId}` });
        setPendingSide(null);
        setTimeout(() => setStatus(null), 2000);
      } catch (err: unknown) {
        setStatus({ type: "err", msg: err instanceof Error ? err.message : String(err) });
        setPendingSide(null);
      } finally {
        setSubmitting(false);
      }
    },
    [symbol, quantity, lotSize, submitting],
  );

  // Keyboard shortcuts: Shift+B -> BUY, Shift+S -> SELL, Enter -> confirm, Escape -> cancel
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Don't capture if user is typing in another input element
      const active = document.activeElement;
      const isOtherInput =
        active instanceof HTMLInputElement && active !== qtyRef.current;
      if (isOtherInput) return;

      if (e.shiftKey && e.key === "B") {
        e.preventDefault();
        setPendingSide("BUY");
        qtyRef.current?.focus();
        qtyRef.current?.select();
      } else if (e.shiftKey && e.key === "S") {
        e.preventDefault();
        setPendingSide("SELL");
        qtyRef.current?.focus();
        qtyRef.current?.select();
      } else if (e.key === "Enter" && pendingSide) {
        e.preventDefault();
        submitOrder(pendingSide);
      } else if (e.key === "Escape") {
        setPendingSide(null);
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [pendingSide, submitOrder]);

  return (
    <div
      className={cn(
        "flex items-center gap-2 px-3 py-2 bg-terminal-surface rounded",
        "border transition-colors",
        pendingSide === "BUY" && "border-terminal-profit/40",
        pendingSide === "SELL" && "border-terminal-loss/40",
        !pendingSide && "border-terminal-border",
      )}
    >
      {/* Symbol + LTP */}
      <div className="flex flex-col mr-1 min-w-[80px]">
        <span className="text-xs font-mono font-bold text-terminal-accent">
          {symbol}
        </span>
        <span
          className={cn(
            "text-sm font-mono font-bold tabular-nums",
            tick && tick.change >= 0 ? "text-terminal-profit" : "text-terminal-loss",
          )}
        >
          {ltp > 0 ? ltp.toFixed(2) : "--"}
        </span>
      </div>

      {/* Quantity input */}
      <div className="flex flex-col gap-0.5">
        <label className="text-2xs text-terminal-muted font-mono">Lots</label>
        <input
          ref={qtyRef}
          type="number"
          min={1}
          value={quantity}
          onChange={(e) => setQuantity(Math.max(1, Number(e.target.value) || 1))}
          className={cn(
            "w-16 px-1.5 py-1 rounded text-xs font-mono tabular-nums text-center",
            "bg-terminal-bg border border-terminal-border text-gray-100",
            "focus:outline-none focus:border-terminal-accent",
          )}
          onKeyDown={(e) => {
            if (e.key === "Enter" && pendingSide) {
              e.preventDefault();
              submitOrder(pendingSide);
            }
          }}
        />
      </div>

      {/* BUY button */}
      <button
        onClick={() => submitOrder("BUY")}
        disabled={submitting}
        className={cn(
          "px-4 py-2 rounded font-mono font-bold text-xs transition-all",
          "bg-terminal-profit/20 text-terminal-profit",
          "hover:bg-terminal-profit hover:text-terminal-bg",
          "disabled:opacity-50 disabled:cursor-not-allowed",
          pendingSide === "BUY" && "ring-2 ring-terminal-profit bg-terminal-profit/30",
        )}
      >
        BUY
      </button>

      {/* SELL button */}
      <button
        onClick={() => submitOrder("SELL")}
        disabled={submitting}
        className={cn(
          "px-4 py-2 rounded font-mono font-bold text-xs transition-all",
          "bg-terminal-loss/20 text-terminal-loss",
          "hover:bg-terminal-loss hover:text-terminal-bg",
          "disabled:opacity-50 disabled:cursor-not-allowed",
          pendingSide === "SELL" && "ring-2 ring-terminal-loss bg-terminal-loss/30",
        )}
      >
        SELL
      </button>

      {/* Status indicator */}
      <div className="flex-1 min-w-[60px]">
        {submitting && (
          <span className="text-2xs font-mono text-terminal-muted animate-pulse">
            Sending...
          </span>
        )}
        {status && !submitting && (
          <span
            className={cn(
              "text-2xs font-mono",
              status.type === "ok" ? "text-terminal-profit" : "text-terminal-loss",
            )}
          >
            {status.msg}
          </span>
        )}
      </div>

      {/* Keyboard hints */}
      <div className="flex flex-col items-end text-2xs text-terminal-muted font-mono gap-0.5">
        <span>
          <kbd className="px-1 py-0.5 bg-terminal-bg rounded border border-terminal-border text-terminal-muted">
            Shift+B
          </kbd>{" "}
          Buy
        </span>
        <span>
          <kbd className="px-1 py-0.5 bg-terminal-bg rounded border border-terminal-border text-terminal-muted">
            Shift+S
          </kbd>{" "}
          Sell
        </span>
      </div>
    </div>
  );
}
