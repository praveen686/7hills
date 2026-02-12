import { useEffect, useCallback } from "react";
import { useAtomValue, useSetAtom } from "jotai";
import { Zap, CheckCircle2, AlertTriangle, Info, X } from "lucide-react";

import { toastsAtom, dismissToastAtom, type ToastItem } from "@/stores/workspace";

// ---------------------------------------------------------------------------
// Icon + colour per toast type
// ---------------------------------------------------------------------------

const TOAST_CONFIG: Record<
  ToastItem["type"],
  { icon: React.ReactNode; borderColor: string; bgColor: string }
> = {
  signal: {
    icon: <Zap size={14} />,
    borderColor: "border-terminal-info",
    bgColor: "bg-terminal-info/10",
  },
  fill: {
    icon: <CheckCircle2 size={14} />,
    borderColor: "border-terminal-profit",
    bgColor: "bg-terminal-profit/10",
  },
  breaker: {
    icon: <AlertTriangle size={14} />,
    borderColor: "border-terminal-loss",
    bgColor: "bg-terminal-loss/10",
  },
  info: {
    icon: <Info size={14} />,
    borderColor: "border-terminal-border-bright",
    bgColor: "bg-terminal-panel",
  },
};

const TOAST_ICON_COLOR: Record<ToastItem["type"], string> = {
  signal: "text-terminal-info",
  fill: "text-terminal-profit",
  breaker: "text-terminal-loss",
  info: "text-terminal-muted",
};

// ---------------------------------------------------------------------------
// Auto-dismiss hook
// ---------------------------------------------------------------------------

function useAutoDismiss(id: string, durationMs: number = 5000) {
  const dismiss = useSetAtom(dismissToastAtom);
  useEffect(() => {
    const timer = setTimeout(() => dismiss(id), durationMs);
    return () => clearTimeout(timer);
  }, [id, durationMs, dismiss]);
}

// ---------------------------------------------------------------------------
// Single toast
// ---------------------------------------------------------------------------

function Toast({ item }: { item: ToastItem }) {
  useAutoDismiss(item.id);
  const dismiss = useSetAtom(dismissToastAtom);
  const cfg = TOAST_CONFIG[item.type];
  const iconColor = TOAST_ICON_COLOR[item.type];

  const handleDismiss = useCallback(() => dismiss(item.id), [dismiss, item.id]);

  const timeStr = new Date(item.timestamp).toLocaleTimeString("en-IN", {
    timeZone: "Asia/Kolkata",
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  return (
    <div
      className={`
        flex items-start gap-3 w-80 px-3 py-2.5 rounded-lg border-l-2
        bg-terminal-surface border border-terminal-border
        ${cfg.borderColor} ${cfg.bgColor}
        shadow-lg shadow-black/40 backdrop-blur-sm
        animate-in slide-in-from-right-full duration-200
      `}
      role="alert"
    >
      <span className={`flex-shrink-0 mt-0.5 ${iconColor}`}>{cfg.icon}</span>

      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs font-semibold text-gray-200 truncate">
            {item.title}
          </span>
          <span className="text-2xs text-terminal-muted tabular-nums flex-shrink-0">
            {timeStr}
          </span>
        </div>
        <p className="text-2xs text-gray-400 mt-0.5 leading-relaxed line-clamp-2">
          {item.message}
        </p>
      </div>

      <button
        onClick={handleDismiss}
        className="flex-shrink-0 p-0.5 rounded hover:bg-terminal-border text-terminal-muted hover:text-gray-300 transition-colors"
        aria-label="Dismiss"
      >
        <X size={12} />
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Toast container
// ---------------------------------------------------------------------------

export function AlertToast() {
  const toasts = useAtomValue(toastsAtom);

  if (toasts.length === 0) return null;

  return (
    <div className="fixed top-14 right-3 z-[60] flex flex-col gap-2 pointer-events-auto">
      {toasts.map((item) => (
        <Toast key={item.id} item={item} />
      ))}
    </div>
  );
}
