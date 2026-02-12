import { useCallback } from "react";
import { useAtom, useAtomValue } from "jotai";
import {
  orderbookAtom,
  selectedSymbolAtom,
  selectedOrderbookAtom,
  type OrderbookData,
} from "@/stores/market";
import { useTauriStream } from "@/hooks/useTauriStream";

/**
 * Subscribe to the orderbook stream for a given symbol (defaults to selected
 * symbol). Returns the current OrderbookData from the market store.
 *
 * The backend should emit events named "orderbook_update" with payload
 * { symbol: string; bids: [number,number][]; asks: [number,number][]; spread: number; midPrice: number; }
 */
export function useOrderbook(symbolOverride?: string): OrderbookData | null {
  const selectedSymbol = useAtomValue(selectedSymbolAtom);
  const symbol = symbolOverride ?? selectedSymbol;
  const [, setOrderbook] = useAtom(orderbookAtom);

  const handleUpdate = useCallback(
    (data: OrderbookData & { symbol: string }) => {
      if (data.symbol === symbol) {
        setOrderbook((prev) => {
          const next = new Map(prev);
          next.set(data.symbol, {
            bids: data.bids,
            asks: data.asks,
            spread: data.spread,
            midPrice: data.midPrice,
          });
          return next;
        });
      }
    },
    [symbol, setOrderbook],
  );

  useTauriStream<OrderbookData & { symbol: string }>("orderbook_update", handleUpdate);

  const selectedOb = useAtomValue(selectedOrderbookAtom);

  // If a custom symbol override is provided, read directly from the map
  if (symbolOverride) {
    const obMap = useAtomValue(orderbookAtom);
    return obMap.get(symbolOverride) ?? null;
  }

  return selectedOb;
}
