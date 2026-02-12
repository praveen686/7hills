/**
 * All possible keyboard actions in the terminal.
 */
export type KeyAction =
  | "command_palette"
  | "buy"
  | "sell"
  | "close"
  | "workspace_trading"
  | "workspace_analysis"
  | "workspace_backtest"
  | "workspace_monitor"
  | "help"
  | "cancel_all_orders"
  | "flatten_position"
  | "toggle_orderbook"
  | "toggle_chart"
  | "focus_order_entry"
  | "next_symbol"
  | "prev_symbol";

/**
 * Default keybinding map: combo string -> action.
 *
 * Combo format: modifiers joined with "+" in order ctrl, alt, shift, then the key.
 * All lowercase. "ctrl" covers both Ctrl and Cmd (macOS).
 *
 * @example "ctrl+k" -> "command_palette"
 */
export const DEFAULT_KEYBINDINGS: Record<string, KeyAction> = {
  // Core navigation
  "ctrl+k": "command_palette",
  escape: "close",
  f1: "help",

  // Workspace switching
  "ctrl+1": "workspace_trading",
  "ctrl+2": "workspace_analysis",
  "ctrl+3": "workspace_backtest",
  "ctrl+4": "workspace_monitor",

  // Trading actions
  "shift+b": "buy",
  "shift+s": "sell",
  "ctrl+shift+x": "cancel_all_orders",
  "ctrl+shift+f": "flatten_position",

  // Panel toggles
  "ctrl+shift+o": "toggle_orderbook",
  "ctrl+shift+c": "toggle_chart",

  // Order entry
  "ctrl+e": "focus_order_entry",

  // Symbol navigation
  "alt+arrowup": "next_symbol",
  "alt+arrowdown": "prev_symbol",
};

/**
 * Reverse lookup: action -> combo string (for displaying shortcuts in UI).
 */
export const ACTION_TO_COMBO: Record<KeyAction, string> = Object.fromEntries(
  Object.entries(DEFAULT_KEYBINDINGS).map(([combo, action]) => [action, combo]),
) as Record<KeyAction, string>;

/**
 * Format a combo string for display in the UI.
 * @example formatCombo("ctrl+k") => "Ctrl+K"
 * @example formatCombo("shift+b") => "Shift+B"
 */
export function formatCombo(combo: string): string {
  return combo
    .split("+")
    .map((part) => {
      switch (part) {
        case "ctrl":
          return "Ctrl";
        case "alt":
          return "Alt";
        case "shift":
          return "Shift";
        case "escape":
          return "Esc";
        case "arrowup":
          return "\u2191";
        case "arrowdown":
          return "\u2193";
        case "arrowleft":
          return "\u2190";
        case "arrowright":
          return "\u2192";
        default:
          return part.toUpperCase();
      }
    })
    .join("+");
}
