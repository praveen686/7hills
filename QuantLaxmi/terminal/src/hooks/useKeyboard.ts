import { useEffect, useRef, useCallback } from "react";
import { useSetAtom } from "jotai";
import { commandPaletteOpenAtom, activeWorkspaceAtom } from "@/stores/workspace";
import { DEFAULT_KEYBINDINGS, type KeyAction } from "@/lib/keybindings";

type ShortcutHandler = () => void;

interface KeyboardRegistry {
  /** Register a handler for a key action. Returns an unregister function. */
  register: (action: KeyAction, handler: ShortcutHandler) => () => void;
  /** Unregister a handler for a key action. */
  unregister: (action: KeyAction) => void;
}

/**
 * Normalize a KeyboardEvent into a binding string like "ctrl+k" or "shift+b".
 */
function normalizeKeyCombo(e: KeyboardEvent): string {
  const parts: string[] = [];
  if (e.ctrlKey || e.metaKey) parts.push("ctrl");
  if (e.altKey) parts.push("alt");
  if (e.shiftKey) parts.push("shift");

  const key = e.key.toLowerCase();
  // Avoid duplicating modifier keys in the combo
  if (!["control", "alt", "shift", "meta"].includes(key)) {
    parts.push(key);
  }

  return parts.join("+");
}

/**
 * Global keyboard shortcut hook.
 *
 * Provides a registry for custom handlers and installs default shortcuts
 * (command palette, workspace switching, buy/sell, help, escape).
 *
 * @example
 * ```tsx
 * const { register } = useKeyboard();
 * useEffect(() => {
 *   const unreg = register("buy", () => console.log("Buy!"));
 *   return unreg;
 * }, [register]);
 * ```
 */
export function useKeyboard(): KeyboardRegistry {
  const setCommandPaletteOpen = useSetAtom(commandPaletteOpenAtom);
  const setWorkspace = useSetAtom(activeWorkspaceAtom);

  // Mutable map of action -> handler (custom overrides)
  const handlersRef = useRef<Map<KeyAction, ShortcutHandler>>(new Map());

  // Invert the keybindings map: combo string -> action
  const comboToAction = useRef<Map<string, KeyAction>>(new Map());
  if (comboToAction.current.size === 0) {
    for (const [combo, action] of Object.entries(DEFAULT_KEYBINDINGS)) {
      comboToAction.current.set(combo, action);
    }
  }

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't intercept if user is typing in an input/textarea
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") {
        // Allow Escape even in inputs
        if (e.key !== "Escape") return;
      }

      const combo = normalizeKeyCombo(e);
      const action = comboToAction.current.get(combo);
      if (!action) return;

      e.preventDefault();
      e.stopPropagation();

      // Check for custom handler first
      const customHandler = handlersRef.current.get(action);
      if (customHandler) {
        customHandler();
        return;
      }

      // Built-in handlers
      switch (action) {
        case "command_palette":
          setCommandPaletteOpen((prev) => !prev);
          break;
        case "close":
          setCommandPaletteOpen(false);
          break;
        case "workspace_trading":
          setWorkspace("trading");
          break;
        case "workspace_analysis":
          setWorkspace("analysis");
          break;
        case "workspace_backtest":
          setWorkspace("backtest");
          break;
        case "workspace_monitor":
          setWorkspace("monitor");
          break;
        case "help":
          // TODO: open help panel
          break;
        default:
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [setCommandPaletteOpen, setWorkspace]);

  const register = useCallback((action: KeyAction, handler: ShortcutHandler) => {
    handlersRef.current.set(action, handler);
    return () => {
      handlersRef.current.delete(action);
    };
  }, []);

  const unregister = useCallback((action: KeyAction) => {
    handlersRef.current.delete(action);
  }, []);

  return { register, unregister };
}
