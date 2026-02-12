import { useState, useCallback, useEffect, useRef, useMemo } from "react";
import { useAtomValue, useAtom } from "jotai";
import { apiFetch } from "@/lib/api";
import { selectedSymbolAtom, selectedTickAtom } from "@/stores/market";
import { selectedOrderTypeAtom, type OrderType, type OrderSide } from "@/stores/trading";
import { useTauriCommand } from "@/hooks/useTauriCommand";
import { cn, formatINR } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type ProductType = "NRML" | "MIS";

interface OrderResponse {
  orderId: string;
  status: string;
}

interface MarginEstimate {
  required: number;
  available: number;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ORDER_TYPES: OrderType[] = ["MARKET", "LIMIT", "SL", "SL-M"];
const LOT_MULTIPLIERS = [1, 2, 5, 10];

const LOT_SIZES: Record<string, number> = {
  NIFTY: 25,
  BANKNIFTY: 15,
  FINNIFTY: 25,
  MIDCPNIFTY: 50,
  RELIANCE: 250,
  HDFCBANK: 550,
  TCS: 150,
  INFY: 300,
};

function formatPrice(p: number): string {
  return p.toLocaleString("en-IN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function OrderEntry() {
  const symbol = useAtomValue(selectedSymbolAtom);
  const tick = useAtomValue(selectedTickAtom);
  const [orderType, setOrderType] = useAtom(selectedOrderTypeAtom);

  const [side, setSide] = useState<OrderSide>("BUY");
  const [lots, setLots] = useState(1);
  const [price, setPrice] = useState<string>("");
  const [triggerPrice, setTriggerPrice] = useState<string>("");
  const [product, setProduct] = useState<ProductType>("NRML");
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const formRef = useRef<HTMLFormElement>(null);

  const { data: margin, execute: fetchMargin } =
    useTauriCommand<MarginEstimate>("estimate_margin");

  const isBuy = side === "BUY";
  const lotSize = LOT_SIZES[symbol] ?? 1;
  const quantity = lots * lotSize;
  const needsPrice = orderType === "LIMIT" || orderType === "SL";
  const needsTrigger = orderType === "SL" || orderType === "SL-M";
  const ltp = tick?.ltp ?? 0;

  // Estimated margin (rough: qty * ltp * 15% for futures)
  const estimatedMargin = useMemo(() => {
    return quantity * ltp * 0.15;
  }, [quantity, ltp]);

  // Pre-fill price from LTP when switching to LIMIT
  useEffect(() => {
    if (needsPrice && ltp > 0 && price === "") {
      setPrice(ltp.toFixed(2));
    }
  }, [needsPrice, ltp, price]);

  // Fetch margin estimate on parameter change
  useEffect(() => {
    if (!symbol || quantity <= 0) return;
    const timer = setTimeout(() => {
      fetchMargin({
        symbol,
        side,
        quantity,
        orderType,
        price: needsPrice ? Number(price) || ltp : ltp,
        product,
      }).catch(() => {});
    }, 300);
    return () => clearTimeout(timer);
  }, [symbol, side, quantity, orderType, price, product, ltp, fetchMargin, needsPrice]);

  // Submit handler
  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      setError(null);
      setSuccess(null);

      const priceVal = needsPrice ? Number(price) : null;
      const triggerVal = needsTrigger ? Number(triggerPrice) : null;

      if (needsPrice && (!priceVal || priceVal <= 0)) {
        setError("Enter a valid price");
        return;
      }
      if (needsTrigger && (!triggerVal || triggerVal <= 0)) {
        setError("Enter a valid trigger price");
        return;
      }

      setSubmitting(true);
      try {
        const result = await apiFetch<OrderResponse>("/api/trading/order", {
          method: "POST",
          body: JSON.stringify({
            symbol,
            side,
            quantity,
            orderType,
            price: priceVal,
            triggerPrice: triggerVal,
            product,
          }),
        });
        setSuccess(`Order placed: ${result.orderId}`);
        setPrice("");
        setTriggerPrice("");
        setLots(1);
        setTimeout(() => setSuccess(null), 3000);
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setSubmitting(false);
      }
    },
    [symbol, side, quantity, orderType, price, triggerPrice, product, needsPrice, needsTrigger],
  );

  // Keyboard: Enter to submit when focused within form
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Enter" && !e.ctrlKey && !e.metaKey && !e.altKey) {
        if (formRef.current?.contains(document.activeElement)) {
          e.preventDefault();
          formRef.current?.requestSubmit();
        }
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  return (
    <div className="flex flex-col h-full w-full bg-terminal-surface">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-terminal-border">
        <span className="text-xs font-mono font-semibold text-terminal-accent">
          Order Entry
        </span>
        <span className="text-2xs text-terminal-muted">{symbol}</span>
      </div>

      <form
        ref={formRef}
        onSubmit={handleSubmit}
        className="flex flex-col gap-3 p-3 flex-1 overflow-y-auto"
      >
        {/* Buy / Sell toggle */}
        <div className="grid grid-cols-2 gap-1 p-0.5 bg-terminal-bg rounded">
          <button
            type="button"
            onClick={() => setSide("BUY")}
            className={cn(
              "py-2 text-xs font-mono font-bold rounded transition-colors",
              isBuy
                ? "bg-terminal-profit/15 text-terminal-profit border-b-2 border-terminal-profit"
                : "text-terminal-muted hover:text-terminal-profit",
            )}
          >
            BUY
          </button>
          <button
            type="button"
            onClick={() => setSide("SELL")}
            className={cn(
              "py-2 text-xs font-mono font-bold rounded transition-colors",
              !isBuy
                ? "bg-terminal-loss/15 text-terminal-loss border-b-2 border-terminal-loss"
                : "text-terminal-muted hover:text-terminal-loss",
            )}
          >
            SELL
          </button>
        </div>

        {/* LTP display */}
        <div className="flex items-center justify-between px-1">
          <span className="text-2xs text-terminal-muted">LTP</span>
          <span
            className={cn(
              "text-sm font-mono font-bold",
              tick && tick.change >= 0 ? "text-terminal-profit" : "text-terminal-loss",
            )}
          >
            {ltp > 0 ? formatPrice(ltp) : "--"}
          </span>
        </div>

        {/* Order type tabs */}
        <div className="flex gap-0.5 p-0.5 bg-terminal-bg rounded">
          {ORDER_TYPES.map((ot) => (
            <button
              key={ot}
              type="button"
              onClick={() => {
                setOrderType(ot);
                setError(null);
              }}
              className={cn(
                "flex-1 py-1.5 text-2xs font-mono rounded transition-colors",
                orderType === ot
                  ? "bg-terminal-panel text-gray-100 font-semibold"
                  : "text-terminal-muted hover:text-gray-300",
              )}
            >
              {ot}
            </button>
          ))}
        </div>

        {/* Quantity (lots) */}
        <div className="space-y-1">
          <label className="text-2xs text-terminal-muted font-mono">
            Quantity ({lotSize} per lot)
          </label>
          <div className="flex gap-1">
            <input
              type="number"
              value={lots}
              min={1}
              onChange={(e) => setLots(Math.max(1, parseInt(e.target.value) || 1))}
              className={cn(
                "flex-1 px-2 py-1.5 rounded text-xs font-mono tabular-nums",
                "bg-terminal-bg border border-terminal-border text-gray-100",
                "focus:outline-none focus:border-terminal-accent",
              )}
            />
            {LOT_MULTIPLIERS.map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setLots(m)}
                className={cn(
                  "px-2 py-1.5 rounded text-2xs font-mono border transition-colors",
                  lots === m
                    ? "border-terminal-accent text-terminal-accent bg-terminal-accent/10"
                    : "border-terminal-border text-terminal-muted hover:border-terminal-border-bright",
                )}
              >
                {m}x
              </button>
            ))}
          </div>
          <span className="text-2xs text-terminal-muted">
            Total: {quantity.toLocaleString("en-IN")} units
          </span>
        </div>

        {/* Price (for LIMIT / SL) */}
        {needsPrice && (
          <div className="space-y-1">
            <label className="text-2xs text-terminal-muted font-mono">Price</label>
            <input
              type="number"
              step="0.05"
              value={price}
              onChange={(e) => setPrice(e.target.value)}
              placeholder={ltp > 0 ? formatPrice(ltp) : "0.00"}
              className={cn(
                "w-full px-2 py-1.5 rounded text-xs font-mono tabular-nums",
                "bg-terminal-bg border border-terminal-border text-gray-100",
                "focus:outline-none focus:border-terminal-accent",
              )}
            />
          </div>
        )}

        {/* Trigger Price (for SL / SL-M) */}
        {needsTrigger && (
          <div className="space-y-1">
            <label className="text-2xs text-terminal-muted font-mono">
              Trigger Price
            </label>
            <input
              type="number"
              step="0.05"
              value={triggerPrice}
              onChange={(e) => setTriggerPrice(e.target.value)}
              placeholder="0.00"
              className={cn(
                "w-full px-2 py-1.5 rounded text-xs font-mono tabular-nums",
                "bg-terminal-bg border border-terminal-border text-gray-100",
                "focus:outline-none focus:border-terminal-accent",
              )}
            />
          </div>
        )}

        {/* Product type */}
        <div className="space-y-1">
          <label className="text-2xs text-terminal-muted font-mono">Product</label>
          <div className="grid grid-cols-2 gap-1 p-0.5 bg-terminal-bg rounded">
            {(["NRML", "MIS"] as const).map((p) => (
              <button
                key={p}
                type="button"
                onClick={() => setProduct(p)}
                className={cn(
                  "py-1.5 text-2xs font-mono rounded transition-colors",
                  product === p
                    ? "bg-terminal-panel text-gray-100 font-semibold"
                    : "text-terminal-muted hover:text-gray-300",
                )}
              >
                {p}
              </button>
            ))}
          </div>
        </div>

        {/* Estimated Margin */}
        <div className="flex items-center justify-between px-2 py-1.5 bg-terminal-bg rounded text-2xs font-mono border-t border-terminal-border">
          <span className="text-terminal-muted">Est. Margin</span>
          <span className="text-gray-200">
            {margin ? formatINR(margin.required) : formatINR(estimatedMargin)}
          </span>
        </div>

        {/* Error / Success messages */}
        {error && (
          <div className="px-2 py-1.5 bg-terminal-loss/10 border border-terminal-loss/30 rounded text-2xs text-terminal-loss font-mono">
            {error}
          </div>
        )}
        {success && (
          <div className="px-2 py-1.5 bg-terminal-profit/10 border border-terminal-profit/30 rounded text-2xs text-terminal-profit font-mono">
            {success}
          </div>
        )}

        {/* Spacer */}
        <div className="flex-1" />

        {/* Place order button */}
        <button
          type="submit"
          disabled={submitting}
          className={cn(
            "w-full py-2.5 rounded font-mono font-bold text-sm transition-all",
            "disabled:opacity-50 disabled:cursor-not-allowed",
            isBuy
              ? "bg-terminal-profit text-black hover:bg-terminal-profit/90"
              : "bg-terminal-loss text-white hover:bg-terminal-loss/90",
          )}
        >
          {submitting
            ? "Placing..."
            : `${side} ${quantity.toLocaleString("en-IN")} ${symbol}`}
        </button>

        <span className="text-center text-2xs text-terminal-muted">
          Press Enter to place
        </span>
      </form>
    </div>
  );
}
