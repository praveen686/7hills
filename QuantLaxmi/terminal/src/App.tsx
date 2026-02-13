import { lazy, Suspense, useEffect } from "react";
import { useAtom, useSetAtom } from "jotai";
import { commandPaletteOpenAtom } from "@/stores/workspace";
import { pageAtom, authTokenAtom, checkUrlToken } from "@/stores/auth";
import { CommandPalette } from "@/components/terminal/CommandPalette";
import { TickerBar } from "@/components/market/TickerBar";
import { WorkspaceManager } from "@/components/workspace/WorkspaceManager";
import { StatusBar } from "@/components/terminal/StatusBar";
import { KeyboardNavigator } from "@/components/terminal/KeyboardNavigator";
import { AlertToast } from "@/components/terminal/AlertToast";
import { SymbolSearch } from "@/components/market/SymbolSearch";
import { LandingPage } from "@/components/landing/LandingPage";
import { Sidebar } from "@/components/navigation/Sidebar";
import { useConnection } from "@/hooks/useConnection";

// Lazy-load pages that aren't always visible
const DashboardPage = lazy(() => import("@/components/dashboard/DashboardPage"));
const StrategiesPage = lazy(() => import("@/components/strategies/StrategiesPage"));
const SettingsPage = lazy(() => import("@/components/settings/SettingsPage"));
const ProfilePage = lazy(() => import("@/components/profile/ProfilePage"));

function PageSpinner() {
  return (
    <div className="flex items-center justify-center h-full text-terminal-muted text-sm">
      Loading...
    </div>
  );
}

export default function App() {
  const [commandPaletteOpen, setCommandPaletteOpen] = useAtom(commandPaletteOpenAtom);
  const [page, setPage] = useAtom(pageAtom);
  const setToken = useSetAtom(authTokenAtom);

  // Handle OAuth callback token in URL
  useEffect(() => {
    const token = checkUrlToken();
    if (token) {
      setToken(token);
      setPage("dashboard");
    }
  }, [setToken, setPage]);

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

  if (page === "landing") return <LandingPage />;

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-terminal-bg text-terminal-text">
      <KeyboardNavigator />
      <CommandPalette open={commandPaletteOpen} onOpenChange={setCommandPaletteOpen} />
      <SymbolSearch />
      <AlertToast />
      <TickerBar />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <main className="flex-1 overflow-hidden">
          <Suspense fallback={<PageSpinner />}>
            {page === "terminal" && <WorkspaceManager />}
            {page === "dashboard" && <DashboardPage />}
            {page === "strategies" && <StrategiesPage />}
            {page === "settings" && <SettingsPage />}
            {page === "profile" && <ProfilePage />}
          </Suspense>
        </main>
      </div>
      <StatusBar />
    </div>
  );
}
