import { useEffect } from "react";
import { useAtom } from "jotai";
import { commandPaletteOpenAtom } from "@/stores/workspace";
import { CommandPalette } from "@/components/terminal/CommandPalette";
import { TickerBar } from "@/components/market/TickerBar";
import { WorkspaceManager } from "@/components/workspace/WorkspaceManager";
import { StatusBar } from "@/components/terminal/StatusBar";
import { KeyboardNavigator } from "@/components/terminal/KeyboardNavigator";
import { AlertToast } from "@/components/terminal/AlertToast";
import { SymbolSearch } from "@/components/market/SymbolSearch";
import { useConnection } from "@/hooks/useConnection";

export default function App() {
  const [commandPaletteOpen, setCommandPaletteOpen] = useAtom(commandPaletteOpenAtom);

  // Mount connection health polling
  useConnection();

  useEffect(() => {
    // Prevent default browser shortcuts that conflict with terminal
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "k") {
        e.preventDefault();
        setCommandPaletteOpen((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [setCommandPaletteOpen]);

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-terminal-bg text-gray-100">
      <KeyboardNavigator />
      <CommandPalette open={commandPaletteOpen} onOpenChange={setCommandPaletteOpen} />
      <SymbolSearch />
      <AlertToast />
      <TickerBar />
      <main className="flex-1 overflow-hidden">
        <WorkspaceManager />
      </main>
      <StatusBar />
    </div>
  );
}
