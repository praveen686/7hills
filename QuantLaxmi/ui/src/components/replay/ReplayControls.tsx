"use client";

import { clsx } from "clsx";
import type { ReplayState, PlaybackSpeed } from "@/hooks/useReplay";
import { formatTime } from "@/lib/formatters";

// ============================================================
// Replay Controls — Date selector, transport buttons, speed
// ============================================================

interface ReplayControlsProps {
  state: ReplayState;
  dates: string[] | undefined;
  selectedDate: string | null;
  currentTs: string;
  speed: PlaybackSpeed;
  hasMore: boolean;
  isLoadingDates: boolean;
  onSelectDate: (date: string) => void;
  onPlay: () => void;
  onPause: () => void;
  onStep: () => void;
  onReset: () => void;
  onSetSpeed: (speed: PlaybackSpeed) => void;
}

const STATE_LABELS: Record<ReplayState, string> = {
  idle: "Select Date",
  date_selected: "Ready",
  playing: "Playing",
  paused: "Paused",
  done: "Complete",
};

const STATE_COLORS: Record<ReplayState, string> = {
  idle: "bg-gray-700 text-gray-300",
  date_selected: "bg-accent/20 text-accent",
  playing: "bg-profit/20 text-profit",
  paused: "bg-yellow-500/20 text-yellow-400",
  done: "bg-gray-600 text-gray-300",
};

const SPEEDS: PlaybackSpeed[] = [1, 2, 5, 10];

export function ReplayControls({
  state,
  dates,
  selectedDate,
  currentTs,
  speed,
  hasMore,
  isLoadingDates,
  onSelectDate,
  onPlay,
  onPause,
  onStep,
  onReset,
  onSetSpeed,
}: ReplayControlsProps) {
  const canPlay = state === "date_selected" || state === "paused";
  const canPause = state === "playing";
  const canStep = state === "date_selected" || state === "paused";
  const canReset = state !== "idle";

  return (
    <div className="card p-4">
      <div className="flex items-center gap-4 flex-wrap">
        {/* Date Selector */}
        <div className="flex items-center gap-2">
          <label className="text-xs text-gray-500 uppercase tracking-wider">
            Date
          </label>
          <select
            className="input-field text-sm py-1.5 px-3 min-w-[160px]"
            value={selectedDate || ""}
            onChange={(e) => e.target.value && onSelectDate(e.target.value)}
            disabled={isLoadingDates || state === "playing"}
          >
            <option value="">
              {isLoadingDates ? "Loading..." : "Select date"}
            </option>
            {dates?.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </div>

        {/* Transport Controls */}
        <div className="flex items-center gap-1.5">
          {/* Play / Pause */}
          {canPlay && (
            <button
              onClick={onPlay}
              className="btn-primary px-3 py-1.5 text-sm flex items-center gap-1.5"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 5v14l11-7z" />
              </svg>
              Play
            </button>
          )}
          {canPause && (
            <button
              onClick={onPause}
              className="btn-primary px-3 py-1.5 text-sm flex items-center gap-1.5"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                <path d="M6 4h4v16H6zm8 0h4v16h-4z" />
              </svg>
              Pause
            </button>
          )}

          {/* Step Forward */}
          <button
            onClick={onStep}
            disabled={!canStep}
            className={clsx(
              "px-2.5 py-1.5 text-sm rounded-lg border transition-colors",
              canStep
                ? "border-gray-700 text-gray-300 hover:bg-gray-800 hover:text-white"
                : "border-gray-800 text-gray-600 cursor-not-allowed"
            )}
            title="Step forward 1 minute"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M21 16l-7.89-5.26a2 2 0 00-2.22 0L3 16" />
            </svg>
          </button>

          {/* Reset */}
          <button
            onClick={onReset}
            disabled={!canReset}
            className={clsx(
              "px-2.5 py-1.5 text-sm rounded-lg border transition-colors",
              canReset
                ? "border-gray-700 text-gray-300 hover:bg-gray-800 hover:text-white"
                : "border-gray-800 text-gray-600 cursor-not-allowed"
            )}
            title="Reset to start"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182" />
            </svg>
          </button>
        </div>

        {/* Speed Selector */}
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-gray-500">Speed</span>
          {SPEEDS.map((s) => (
            <button
              key={s}
              onClick={() => onSetSpeed(s)}
              className={clsx(
                "px-2 py-1 text-xs rounded border transition-colors",
                speed === s
                  ? "bg-accent/20 border-accent/40 text-accent"
                  : "border-gray-700 text-gray-400 hover:border-gray-600 hover:text-gray-300"
              )}
            >
              {s}x
            </button>
          ))}
        </div>

        {/* Current Time + State Badge */}
        <div className="flex items-center gap-3 ml-auto">
          {currentTs && (
            <span className="text-sm font-mono text-gray-300">
              {formatTime(currentTs)}
            </span>
          )}
          <span
            className={clsx(
              "inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium",
              STATE_COLORS[state]
            )}
          >
            {state === "playing" && (
              <span className="w-1.5 h-1.5 rounded-full bg-profit mr-1.5 animate-pulse" />
            )}
            {STATE_LABELS[state]}
          </span>
        </div>
      </div>
    </div>
  );
}
