import type { ReactNode } from "react";
import { X, Maximize2 } from "lucide-react";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface PanelFrameProps {
  id: string;
  title: string;
  icon?: ReactNode;
  children: ReactNode;
  onClose?: () => void;
  onMaximize?: () => void;
  className?: string;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function PanelFrame({
  id,
  title,
  icon,
  children,
  onClose,
  onMaximize,
  className,
}: PanelFrameProps) {
  return (
    <div className={`panel flex flex-col h-full ${className ?? ""}`} data-panel-id={id}>
      {/* Header */}
      <div className="panel-header select-none drag-handle">
        <div className="flex items-center gap-2">
          {icon && (
            <span className="text-terminal-muted flex-shrink-0">{icon}</span>
          )}
          <span className="panel-title">{title}</span>
        </div>

        <div className="flex items-center gap-1">
          {onMaximize && (
            <button
              onClick={onMaximize}
              className="p-1 rounded hover:bg-terminal-border text-terminal-muted hover:text-gray-300 transition-colors"
              aria-label={`Maximize ${title}`}
            >
              <Maximize2 size={12} />
            </button>
          )}
          {onClose && (
            <button
              onClick={onClose}
              className="p-1 rounded hover:bg-terminal-loss/20 text-terminal-muted hover:text-terminal-loss transition-colors"
              aria-label={`Close ${title}`}
            >
              <X size={12} />
            </button>
          )}
        </div>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-auto">{children}</div>
    </div>
  );
}
