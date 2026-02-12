// ============================================================
// WebSocket Manager
// Auto-reconnecting multi-channel WebSocket manager
// ============================================================

import { wsUrl } from "./api";

type Handler = (data: unknown) => void;

interface ChannelState {
  ws: WebSocket | null;
  url: string;
  handlers: Set<Handler>;
  reconnectTimer: ReturnType<typeof setTimeout> | null;
  reconnectDelay: number;
  connected: boolean;
}

class WSManager {
  private channels = new Map<string, ChannelState>();

  /** Connect to a WebSocket channel. */
  connect(channel: string, path: string): void {
    if (this.channels.has(channel)) return;

    const state: ChannelState = {
      ws: null,
      url: wsUrl(path),
      handlers: new Set(),
      reconnectTimer: null,
      reconnectDelay: 1000,
      connected: false,
    };
    this.channels.set(channel, state);
    this._open(channel, state);
  }

  /** Subscribe a handler to a channel. Returns unsubscribe fn. */
  subscribe(channel: string, handler: Handler): () => void {
    const state = this.channels.get(channel);
    if (!state) return () => {};
    state.handlers.add(handler);
    return () => {
      state.handlers.delete(handler);
    };
  }

  /** Disconnect a channel. */
  disconnect(channel: string): void {
    const state = this.channels.get(channel);
    if (!state) return;
    if (state.reconnectTimer) clearTimeout(state.reconnectTimer);
    if (state.ws) {
      state.ws.onclose = null;
      state.ws.close();
    }
    this.channels.delete(channel);
  }

  /** Disconnect all channels. */
  disconnectAll(): void {
    for (const ch of [...this.channels.keys()]) {
      this.disconnect(ch);
    }
  }

  /** Check if a channel is connected. */
  isConnected(channel: string): boolean {
    return this.channels.get(channel)?.connected ?? false;
  }

  /** Check if any channel is connected. */
  get anyConnected(): boolean {
    for (const state of this.channels.values()) {
      if (state.connected) return true;
    }
    return false;
  }

  private _open(channel: string, state: ChannelState): void {
    try {
      const ws = new WebSocket(state.url);
      state.ws = ws;

      ws.onopen = () => {
        state.connected = true;
        state.reconnectDelay = 1000;
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          for (const handler of state.handlers) {
            handler(data);
          }
        } catch {
          // Non-JSON message, ignore
        }
      };

      ws.onclose = () => {
        state.connected = false;
        state.ws = null;
        // Auto-reconnect with exponential backoff
        if (this.channels.has(channel)) {
          state.reconnectTimer = setTimeout(() => {
            this._open(channel, state);
          }, state.reconnectDelay);
          state.reconnectDelay = Math.min(state.reconnectDelay * 2, 30000);
        }
      };

      ws.onerror = () => {
        // onclose will fire after onerror
      };
    } catch {
      // Schedule reconnect
      state.reconnectTimer = setTimeout(() => {
        this._open(channel, state);
      }, state.reconnectDelay);
      state.reconnectDelay = Math.min(state.reconnectDelay * 2, 30000);
    }
  }
}

export const wsManager = new WSManager();
