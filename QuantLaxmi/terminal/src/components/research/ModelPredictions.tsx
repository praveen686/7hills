import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { apiFetch } from "@/lib/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TopFeature {
  name: string;
  importance: number;
}

interface AssetPrediction {
  asset: string;
  position: string; // "long" | "short" | "flat"
  confidence: number;
  direction_score: number;
  top_features: TopFeature[];
}

interface TFTPredictions {
  timestamp: string;
  assets: AssetPrediction[];
  model_version: string;
}

interface ModelStatus {
  tft_loaded: boolean;
  tft_checkpoint: string;
  tft_features: number;
  rl_loaded: boolean;
  rl_agent_type: string;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ModelPredictions() {
  const [predictions, setPredictions] = useState<TFTPredictions | null>(null);
  const [status, setStatus] = useState<ModelStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const [pred, stat] = await Promise.all([
          apiFetch<TFTPredictions>("/api/models/tft/predictions"),
          apiFetch<ModelStatus>("/api/models/status"),
        ]);
        setPredictions(pred);
        setStatus(stat);
      } catch {
        // API offline
      } finally {
        setLoading(false);
      }
    }

    fetchData();
    const timer = setInterval(fetchData, 60000); // refresh every 60s
    return () => clearInterval(timer);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full text-terminal-muted text-xs font-mono">
        Loading model predictions...
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 p-4 h-full overflow-y-auto">
      {/* Header with model status */}
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-bold uppercase tracking-widest text-terminal-text">
          Model Predictions
        </h2>
        <div className="flex items-center gap-2">
          <span
            className={cn(
              "px-1.5 py-0.5 rounded text-2xs font-mono",
              status?.tft_loaded
                ? "bg-terminal-profit/20 text-terminal-profit"
                : "bg-terminal-surface text-terminal-muted",
            )}
          >
            TFT {status?.tft_loaded ? "ON" : "OFF"}
          </span>
          <span
            className={cn(
              "px-1.5 py-0.5 rounded text-2xs font-mono",
              status?.rl_loaded
                ? "bg-terminal-profit/20 text-terminal-profit"
                : "bg-terminal-surface text-terminal-muted",
            )}
          >
            RL {status?.rl_loaded ? "ON" : "OFF"}
          </span>
        </div>
      </div>

      {/* Model version */}
      {predictions && (
        <div className="text-2xs text-terminal-muted font-mono">
          v{predictions.model_version} | {new Date(predictions.timestamp).toLocaleTimeString("en-IN", { timeZone: "Asia/Kolkata" })} IST
        </div>
      )}

      {/* Asset prediction cards */}
      {!predictions || predictions.assets.length === 0 ? (
        <div className="flex items-center justify-center h-32 text-terminal-muted text-xs font-mono">
          No predictions available
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-2">
          {predictions.assets.map((asset) => (
            <AssetCard key={asset.asset} asset={asset} />
          ))}
        </div>
      )}

      {/* Feature summary */}
      {status && status.tft_features > 0 && (
        <div className="mt-2 text-2xs text-terminal-muted font-mono">
          {status.tft_features} features | checkpoint: {status.tft_checkpoint || "none"}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Asset prediction card
// ---------------------------------------------------------------------------

function AssetCard({ asset }: { asset: AssetPrediction }) {
  const posColor =
    asset.position === "long"
      ? "text-terminal-profit"
      : asset.position === "short"
        ? "text-terminal-loss"
        : "text-terminal-muted";

  const posIcon =
    asset.position === "long"
      ? "\u25B2" // up triangle
      : asset.position === "short"
        ? "\u25BC" // down triangle
        : "\u25CF"; // circle

  const confidencePct = Math.min(asset.confidence * 100, 100);
  const barWidth = Math.max(confidencePct, 2);

  return (
    <div className="bg-terminal-surface rounded p-3 border border-terminal-border">
      {/* Asset name + position */}
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-mono font-bold text-terminal-text">
          {asset.asset}
        </span>
        <span className={cn("text-xs font-mono font-bold uppercase", posColor)}>
          {posIcon} {asset.position}
        </span>
      </div>

      {/* Confidence bar */}
      <div className="flex items-center gap-2 mb-2">
        <div className="flex-1 h-2 bg-terminal-panel rounded overflow-hidden">
          <div
            className={cn(
              "h-full rounded transition-all duration-500",
              asset.position === "long"
                ? "bg-terminal-profit"
                : asset.position === "short"
                  ? "bg-terminal-loss"
                  : "bg-terminal-muted",
            )}
            style={{ width: `${barWidth}%`, opacity: 0.7 }}
          />
        </div>
        <span className="text-2xs font-mono text-terminal-muted tabular-nums w-10 text-right">
          {confidencePct.toFixed(0)}%
        </span>
      </div>

      {/* Direction score */}
      <div className="text-2xs font-mono text-terminal-muted mb-1">
        Score: {asset.direction_score.toFixed(4)}
      </div>

      {/* Top features */}
      {asset.top_features.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-1">
          {asset.top_features.slice(0, 3).map((f) => (
            <span
              key={f.name}
              className="px-1 py-0.5 rounded bg-terminal-panel text-2xs font-mono text-terminal-muted"
              title={`importance: ${f.importance.toFixed(4)}`}
            >
              {f.name}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
