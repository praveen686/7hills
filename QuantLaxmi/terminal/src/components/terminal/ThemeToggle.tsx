import { useAtom } from "jotai";
import { Sun, Moon } from "lucide-react";
import { themeAtom } from "@/stores/workspace";

export function ThemeToggle({ size = "sm" }: { size?: "sm" | "md" }) {
  const [theme, setTheme] = useAtom(themeAtom);

  if (size === "md") {
    return (
      <button
        onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-terminal-border hover:bg-terminal-panel transition-colors text-sm"
        title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
      >
        {theme === "dark" ? (
          <Sun size={16} className="text-terminal-warning" />
        ) : (
          <Moon size={16} className="text-terminal-accent" />
        )}
        <span className="text-terminal-text-secondary text-xs">
          {theme === "dark" ? "Light" : "Dark"}
        </span>
      </button>
    );
  }

  return (
    <button
      onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
      className="flex items-center justify-center w-7 h-6 rounded hover:bg-terminal-border transition-colors"
      title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
    >
      {theme === "dark" ? (
        <Sun size={14} className="text-terminal-warning" />
      ) : (
        <Moon size={14} className="text-terminal-accent" />
      )}
    </button>
  );
}
