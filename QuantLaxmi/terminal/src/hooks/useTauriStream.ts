import { useEffect, useRef } from "react";
import { wsManager } from "@/lib/ws";

// ---------------------------------------------------------------------------
// Event name → WebSocket channel mapping
// ---------------------------------------------------------------------------

const WS_CHANNELS: Record<string, string> = {
  portfolio_update: "/ws/portfolio",
  tick_update: "/ws/ticks",
  tick: "/ws/ticks",
  signal: "/ws/signals",
  vpin_update: "/ws/risk",
  order_update: "/ws/portfolio",
};

function resolveChannel(eventName: string): string {
  // Direct match
  if (WS_CHANNELS[eventName]) return WS_CHANNELS[eventName];
  // Prefix match: "book:NIFTY" → /ws/ticks, "bar:NIFTY" → /ws/ticks
  if (eventName.startsWith("book:") || eventName.startsWith("bar:") || eventName.startsWith("tick:")) {
    return "/ws/ticks";
  }
  // Default
  return "/ws/portfolio";
}

/**
 * Drop-in replacement for the Tauri event stream hook.
 * Subscribes to a WebSocket channel via wsManager.
 *
 * @param eventName - Event name (e.g. "tick_update", "bar:NIFTY")
 * @param onData    - Callback receiving the event payload
 */
export function useTauriStream<T>(eventName: string, onData: (data: T) => void): void {
  const callbackRef = useRef(onData);
  callbackRef.current = onData;

  useEffect(() => {
    const channel = resolveChannel(eventName);

    // Ensure the channel is connected
    wsManager.connect(channel, channel);

    // Subscribe with filtering: only forward messages matching the event
    const handler = (data: unknown) => {
      const msg = data as Record<string, unknown>;

      // If the message has a "type" or "event" field, check it matches
      if (msg.type === eventName || msg.event === eventName) {
        callbackRef.current(msg.payload as T ?? msg.data as T ?? data as T);
        return;
      }

      // For symbol-scoped events like "bar:NIFTY" — check symbol match
      if (eventName.includes(":")) {
        const [prefix, sym] = eventName.split(":", 2);
        if (
          (msg.type === prefix || msg.event === prefix) &&
          (msg.symbol === sym || msg.instrument === sym)
        ) {
          callbackRef.current(msg.payload as T ?? msg.data as T ?? data as T);
          return;
        }
      }

      // Fallback: forward all messages on the channel
      // (when channel carries a single message type)
      if (!msg.type && !msg.event) {
        callbackRef.current(data as T);
      }
    };

    const unsub = wsManager.subscribe(channel, handler);
    return unsub;
  }, [eventName]);
}
