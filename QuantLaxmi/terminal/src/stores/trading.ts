import { atom } from "jotai";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type OrderType = "MARKET" | "LIMIT" | "SL" | "SL-M";

export type OrderStatus =
  | "PENDING"
  | "OPEN"
  | "FILLED"
  | "PARTIALLY_FILLED"
  | "CANCELLED"
  | "REJECTED";

export type OrderSide = "BUY" | "SELL";

export interface Order {
  orderId: string;
  symbol: string;
  side: OrderSide;
  quantity: number;
  price: number | null;
  orderType: OrderType;
  status: OrderStatus;
  timestamp: string;
  strategyId?: string;
}

// ---------------------------------------------------------------------------
// Atoms
// ---------------------------------------------------------------------------

/** Orders that are pending execution (PENDING / OPEN) */
export const pendingOrdersAtom = atom<Order[]>([]);

/** Orders that have been filled */
export const filledOrdersAtom = atom<Order[]>([]);

/** Currently selected order type in the order entry panel */
export const selectedOrderTypeAtom = atom<OrderType>("MARKET");

// ---------------------------------------------------------------------------
// Derived atoms
// ---------------------------------------------------------------------------

/** All orders combined (pending + filled), most recent first */
export const allOrdersAtom = atom<Order[]>((get) => {
  const pending = get(pendingOrdersAtom);
  const filled = get(filledOrdersAtom);
  return [...pending, ...filled].sort(
    (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime(),
  );
});

/** Count of pending orders */
export const pendingOrderCountAtom = atom<number>((get) => get(pendingOrdersAtom).length);

/** Total filled today */
export const filledTodayCountAtom = atom<number>((get) => get(filledOrdersAtom).length);

/** Pending orders for a given symbol (factory) */
export const pendingOrdersForSymbolAtom = atom<(symbol: string) => Order[]>((get) => {
  const pending = get(pendingOrdersAtom);
  return (symbol: string) => pending.filter((o) => o.symbol === symbol);
});
