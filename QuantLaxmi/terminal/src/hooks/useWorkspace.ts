import { useCallback } from "react";
import { useAtom } from "jotai";
import {
  activeWorkspaceAtom,
  layoutAtom,
  activePanelsAtom,
  type WorkspaceId,
  type LayoutItem,
} from "@/stores/workspace";

interface UseWorkspaceReturn {
  /** Current layout items for react-grid-layout */
  layout: LayoutItem[];
  /** Update the full layout (e.g. after drag/resize) */
  setLayout: (layout: LayoutItem[]) => void;
  /** Active workspace ID */
  activeWorkspace: WorkspaceId;
  /** Switch to a different workspace */
  setActiveWorkspace: (ws: WorkspaceId) => void;
  /** Add a panel to the workspace */
  addPanel: (panel: LayoutItem) => void;
  /** Remove a panel by ID */
  removePanel: (panelId: string) => void;
  /** Currently visible panel IDs */
  activePanels: string[];
}

/**
 * Manage workspace layout state: switching workspaces, adding/removing panels,
 * and updating the grid layout.
 */
export function useWorkspace(): UseWorkspaceReturn {
  const [layout, setLayout] = useAtom(layoutAtom);
  const [activeWorkspace, setActiveWorkspace] = useAtom(activeWorkspaceAtom);
  const [activePanels, setActivePanels] = useAtom(activePanelsAtom);

  const addPanel = useCallback(
    (panel: LayoutItem) => {
      setLayout((prev) => {
        // Don't add if already exists
        if (prev.some((item) => item.i === panel.i)) return prev;
        return [...prev, panel];
      });
      setActivePanels((prev) => {
        if (prev.includes(panel.i)) return prev;
        return [...prev, panel.i];
      });
    },
    [setLayout, setActivePanels],
  );

  const removePanel = useCallback(
    (panelId: string) => {
      setLayout((prev) => prev.filter((item) => item.i !== panelId));
      setActivePanels((prev) => prev.filter((id) => id !== panelId));
    },
    [setLayout, setActivePanels],
  );

  return {
    layout,
    setLayout,
    activeWorkspace,
    setActiveWorkspace,
    activePanels,
    addPanel,
    removePanel,
  };
}
