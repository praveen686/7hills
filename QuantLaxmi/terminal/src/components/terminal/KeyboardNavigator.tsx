import { useEffect } from "react";
import { useSetAtom } from "jotai";

import {
  activeWorkspaceAtom,
  commandPaletteOpenAtom,
  symbolSearchOpenAtom,
  type WorkspaceId,
} from "@/stores/workspace";
import { pageAtom } from "@/stores/auth";

// ---------------------------------------------------------------------------
// Keyboard shortcut map
// ---------------------------------------------------------------------------

/**
 * Invisible component that registers global keyboard shortcuts.
 * Renders nothing -- pure side-effect via useEffect.
 *
 * Shortcuts:
 *   Ctrl+K        -- toggle command palette (handled in App.tsx, kept here for reference)
 *   Ctrl+L        -- open symbol search
 *   Alt+1..4      -- switch workspace
 *   Escape        -- close overlays (handled by individual overlay components)
 */
export function KeyboardNavigator() {
  const setWorkspace = useSetAtom(activeWorkspaceAtom);
  const setCommandPalette = useSetAtom(commandPaletteOpenAtom);
  const setSymbolSearch = useSetAtom(symbolSearchOpenAtom);
  const setPage = useSetAtom(pageAtom);

  useEffect(() => {
    const WORKSPACE_MAP: Record<string, WorkspaceId> = {
      "1": "trading",
      "2": "analysis",
      "3": "backtest",
      "4": "monitor",
    };

    function handleKeyDown(e: KeyboardEvent) {
      // Ignore events when user is typing in an input/textarea
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      // Alt+0 — dashboard
      if (e.altKey && !e.ctrlKey && !e.metaKey && e.key === "0") {
        e.preventDefault();
        setPage("dashboard");
        return;
      }

      // Alt+1..4 — workspace switching
      if (e.altKey && !e.ctrlKey && !e.metaKey) {
        const ws = WORKSPACE_MAP[e.key];
        if (ws) {
          e.preventDefault();
          setPage("terminal");
          setWorkspace(ws);
          return;
        }
      }

      // Ctrl+L — symbol search
      if ((e.ctrlKey || e.metaKey) && e.key === "l") {
        e.preventDefault();
        setSymbolSearch((prev) => !prev);
        return;
      }

      // Ctrl+K — command palette (already handled in App.tsx, but keeping as fallback)
      // Intentionally NOT duplicating — App.tsx owns this shortcut.

      // F11 — fullscreen toggle (Tauri handles natively, this is a no-op hint)
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [setWorkspace, setCommandPalette, setSymbolSearch, setPage]);

  // Render nothing
  return null;
}
