import { useState } from "react";
import { Eye, EyeOff } from "lucide-react";
import { cn } from "@/lib/utils";

interface SecretInputProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}

export function SecretInput({ label, value, onChange, placeholder }: SecretInputProps) {
  const [visible, setVisible] = useState(false);

  return (
    <div className="space-y-1.5">
      <label className="text-xs font-medium text-terminal-muted uppercase tracking-wider">
        {label}
      </label>
      <div className="relative">
        <input
          type={visible ? "text" : "password"}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className={cn(
            "w-full px-3 py-2 pr-10 text-sm font-mono rounded-md",
            "bg-terminal-panel border border-terminal-border",
            "text-terminal-text placeholder:text-terminal-muted/50",
            "focus:outline-none focus:ring-1 focus:ring-terminal-accent focus:border-terminal-accent",
            "transition-colors",
          )}
        />
        <button
          type="button"
          onClick={() => setVisible(!visible)}
          className={cn(
            "absolute right-2 top-1/2 -translate-y-1/2 p-1 rounded",
            "text-terminal-muted hover:text-terminal-text transition-colors",
          )}
          tabIndex={-1}
          aria-label={visible ? "Hide value" : "Show value"}
        >
          {visible ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
        </button>
      </div>
    </div>
  );
}
