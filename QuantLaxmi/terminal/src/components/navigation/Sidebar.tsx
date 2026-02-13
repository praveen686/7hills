import { useAtom } from "jotai";
import {
  LayoutDashboard,
  Crosshair,
  TrendingUp,
  FlaskConical,
  Monitor,
  Layers,
  Settings,
  UserCircle,
  ChevronLeft,
  ChevronRight,
  Sun,
  Moon,
  LogOut,
} from "lucide-react";

import { sidebarCollapsedAtom, activeWorkspaceAtom, themeAtom, type WorkspaceId } from "@/stores/workspace";
import { pageAtom, type PageId } from "@/stores/auth";
import { useAuth } from "@/hooks/useAuth";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface NavItem {
  icon: React.ReactNode;
  label: string;
  page: PageId;
  workspace?: WorkspaceId;
}

// ---------------------------------------------------------------------------
// Nav items
// ---------------------------------------------------------------------------

const topItems: NavItem[] = [
  { icon: <LayoutDashboard size={18} />, label: "Dashboard", page: "dashboard" },
  { icon: <Crosshair size={18} />, label: "Trading", page: "terminal", workspace: "trading" },
  { icon: <TrendingUp size={18} />, label: "Analysis", page: "terminal", workspace: "analysis" },
  { icon: <FlaskConical size={18} />, label: "Backtest", page: "terminal", workspace: "backtest" },
  { icon: <Monitor size={18} />, label: "Monitor", page: "terminal", workspace: "monitor" },
];

const middleItems: NavItem[] = [
  { icon: <Layers size={18} />, label: "Strategies", page: "strategies" },
];

const bottomItems: NavItem[] = [
  { icon: <Settings size={18} />, label: "Settings", page: "settings" },
  { icon: <UserCircle size={18} />, label: "Profile", page: "profile" },
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function Sidebar() {
  const [collapsed, setCollapsed] = useAtom(sidebarCollapsedAtom);
  const [page, setPage] = useAtom(pageAtom);
  const [workspace, setWorkspace] = useAtom(activeWorkspaceAtom);
  const [theme, setTheme] = useAtom(themeAtom);
  const { user, logout } = useAuth();

  const handleClick = (item: NavItem) => {
    setPage(item.page);
    if (item.workspace) {
      setWorkspace(item.workspace);
    }
  };

  const isActive = (item: NavItem) => {
    if (item.page === "terminal") {
      return page === "terminal" && item.workspace === workspace;
    }
    return page === item.page;
  };

  return (
    <div
      className={cn(
        "flex flex-col h-full bg-terminal-surface border-r border-terminal-border flex-shrink-0 transition-[width] duration-200 ease-in-out select-none",
        collapsed ? "w-12" : "w-[220px]",
      )}
    >
      {/* Top navigation */}
      <nav className="flex-1 flex flex-col pt-2 gap-0.5 px-1.5">
        {topItems.map((item) => (
          <SidebarItem
            key={item.label}
            item={item}
            collapsed={collapsed}
            active={isActive(item)}
            onClick={() => handleClick(item)}
          />
        ))}

        {/* Separator */}
        <div className="h-px bg-terminal-border mx-2 my-2" />

        {middleItems.map((item) => (
          <SidebarItem
            key={item.label}
            item={item}
            collapsed={collapsed}
            active={isActive(item)}
            onClick={() => handleClick(item)}
          />
        ))}
      </nav>

      {/* Bottom items */}
      <div className="flex flex-col gap-0.5 px-1.5 pb-2">
        <div className="h-px bg-terminal-border mx-2 my-2" />
        {bottomItems.map((item) => (
          <SidebarItem
            key={item.label}
            item={item}
            collapsed={collapsed}
            active={isActive(item)}
            onClick={() => handleClick(item)}
          />
        ))}

        {/* Theme toggle */}
        <button
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          className={cn(
            "flex items-center gap-3 rounded-md text-sm font-medium transition-colors",
            collapsed ? "justify-center h-9 w-9 mx-auto" : "px-3 h-9",
            "text-terminal-muted hover:text-terminal-text hover:bg-terminal-panel/60",
          )}
        >
          <span className="flex-shrink-0">
            {theme === "dark" ? <Sun size={18} className="text-terminal-warning" /> : <Moon size={18} className="text-terminal-accent" />}
          </span>
          {!collapsed && <span className="truncate">{theme === "dark" ? "Light Mode" : "Dark Mode"}</span>}
        </button>

        {/* Logout */}
        {user && (
          <button
            onClick={logout}
            title="Sign out"
            className={cn(
              "flex items-center gap-3 rounded-md text-sm font-medium transition-colors",
              collapsed ? "justify-center h-9 w-9 mx-auto" : "px-3 h-9",
              "text-terminal-muted hover:text-terminal-loss hover:bg-terminal-loss/10",
            )}
          >
            <span className="flex-shrink-0">
              <LogOut size={18} />
            </span>
            {!collapsed && <span className="truncate">Sign Out</span>}
          </button>
        )}

        {/* Collapse toggle */}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="flex items-center justify-center h-8 rounded-md text-terminal-muted hover:text-terminal-text hover:bg-terminal-panel/60 transition-colors mt-1"
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// SidebarItem
// ---------------------------------------------------------------------------

function SidebarItem({
  item,
  collapsed,
  active,
  onClick,
}: {
  item: NavItem;
  collapsed: boolean;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      title={collapsed ? item.label : undefined}
      className={cn(
        "flex items-center gap-3 rounded-md text-sm font-medium transition-colors",
        collapsed ? "justify-center h-9 w-9 mx-auto" : "px-3 h-9",
        active
          ? "bg-terminal-accent/15 text-terminal-accent"
          : "text-terminal-muted hover:text-terminal-text hover:bg-terminal-panel/60",
      )}
    >
      <span className="flex-shrink-0">{item.icon}</span>
      {!collapsed && <span className="truncate">{item.label}</span>}
    </button>
  );
}
