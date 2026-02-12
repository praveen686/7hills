import { atom } from "jotai";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type WorkspaceId = "trading" | "analysis" | "backtest" | "monitor";

export type ConnectionStatus = "connected" | "disconnected" | "connecting";

export interface LayoutItem {
  i: string;
  x: number;
  y: number;
  w: number;
  h: number;
  minW?: number;
  minH?: number;
  maxW?: number;
  maxH?: number;
  static?: boolean;
}

export interface ConnectionState {
  zerodha: ConnectionStatus;
  binance: ConnectionStatus;
  fastapi: ConnectionStatus;
  latencyMs: number;
}

export interface ToastItem {
  id: string;
  type: "signal" | "fill" | "breaker" | "info";
  title: string;
  message: string;
  timestamp: number;
}

// ---------------------------------------------------------------------------
// Atoms
// ---------------------------------------------------------------------------

/** Active workspace tab */
export const activeWorkspaceAtom = atom<WorkspaceId>("trading");

/** react-grid-layout layout configuration */
export const layoutAtom = atom<LayoutItem[]>([
  { i: "chart", x: 0, y: 0, w: 8, h: 6, minW: 4, minH: 3 },
  { i: "orderbook", x: 8, y: 0, w: 4, h: 6, minW: 3, minH: 3 },
  { i: "positions", x: 0, y: 6, w: 6, h: 4, minW: 3, minH: 2 },
  { i: "orders", x: 6, y: 6, w: 6, h: 4, minW: 3, minH: 2 },
]);

/** Currently visible panel IDs */
export const activePanelsAtom = atom<string[]>(["chart", "orderbook", "positions", "orders"]);

/** Theme — dark by default (terminal aesthetic) */
export const themeAtom = atom<"dark">("dark");

/** Command palette visibility */
export const commandPaletteOpenAtom = atom<boolean>(false);

/** Symbol search dialog open state */
export const symbolSearchOpenAtom = atom<boolean>(false);

/** Connection status for all feeds */
export const connectionAtom = atom<ConnectionState>({
  zerodha: "disconnected",
  binance: "disconnected",
  fastapi: "disconnected",
  latencyMs: 0,
});

/** Current market regime label */
export const regimeAtom = atom<string>("--");

/** Toast notification queue */
export const toastsAtom = atom<ToastItem[]>([]);

/** Derived: push a toast (max 5) */
export const pushToastAtom = atom(
  null,
  (get, set, toast: Omit<ToastItem, "id" | "timestamp">) => {
    const id = crypto.randomUUID();
    const item: ToastItem = { ...toast, id, timestamp: Date.now() };
    const current = get(toastsAtom);
    set(toastsAtom, [item, ...current].slice(0, 5));
  },
);

/** Derived: dismiss a toast by ID */
export const dismissToastAtom = atom(null, (get, set, id: string) => {
  set(
    toastsAtom,
    get(toastsAtom).filter((t) => t.id !== id),
  );
});

// ---------------------------------------------------------------------------
// Derived atoms
// ---------------------------------------------------------------------------

/** Whether a specific panel is visible */
export const isPanelVisibleAtom = atom<(panelId: string) => boolean>((get) => {
  const panels = get(activePanelsAtom);
  return (panelId: string) => panels.includes(panelId);
});
