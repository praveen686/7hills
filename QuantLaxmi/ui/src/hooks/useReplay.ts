"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  fetchReplayDates,
  fetchReplayTimeline,
  fetchReplayStep,
} from "@/lib/api";
import type { TimelineMarker, StepEvent, ReplaySnapshot } from "@/lib/types";

// ============================================================
// Replay State Machine Hook
//
// States: idle -> date_selected -> playing <-> paused -> done
// ============================================================

export type ReplayState = "idle" | "date_selected" | "playing" | "paused" | "done";

export type PlaybackSpeed = 1 | 2 | 5 | 10;

const SPEED_INTERVAL: Record<PlaybackSpeed, number> = {
  1: 1000,
  2: 500,
  5: 200,
  10: 100,
};

// Each step advances 60 seconds of market time
const DEFAULT_DELTA_MS = 60_000;

export interface UseReplayReturn {
  /** Current state of the replay machine */
  state: ReplayState;
  /** Available dates with event data */
  dates: string[] | undefined;
  /** Currently selected date */
  selectedDate: string | null;
  /** Timeline markers for selected date */
  markers: TimelineMarker[] | undefined;
  /** Events accumulated so far */
  recentEvents: StepEvent[];
  /** Current portfolio snapshot */
  snapshot: ReplaySnapshot | null;
  /** Current playback time position */
  currentTs: string;
  /** Playback speed multiplier */
  speed: PlaybackSpeed;
  /** Whether dates are loading */
  isLoadingDates: boolean;
  /** Whether the replay has more events */
  hasMore: boolean;

  /** Select a date to replay */
  selectDate: (date: string) => void;
  /** Start or resume playback */
  play: () => void;
  /** Pause playback */
  pause: () => void;
  /** Advance one step manually */
  stepForward: () => void;
  /** Reset to beginning of selected date */
  reset: () => void;
  /** Change playback speed */
  setSpeed: (speed: PlaybackSpeed) => void;
}

export function useReplay(): UseReplayReturn {
  const [state, setState] = useState<ReplayState>("idle");
  const [selectedDate, setSelectedDate] = useState<string | null>(null);
  const [recentEvents, setRecentEvents] = useState<StepEvent[]>([]);
  const [snapshot, setSnapshot] = useState<ReplaySnapshot | null>(null);
  const [currentTs, setCurrentTs] = useState<string>("");
  const [speed, setSpeedState] = useState<PlaybackSpeed>(1);
  const [hasMore, setHasMore] = useState<boolean>(true);

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const steppingRef = useRef(false);

  // Fetch available dates
  const { data: dates, isLoading: isLoadingDates } = useQuery<string[]>({
    queryKey: ["replay-dates"],
    queryFn: fetchReplayDates,
    staleTime: 60000,
  });

  // Fetch timeline markers for selected date
  const { data: markers } = useQuery<TimelineMarker[]>({
    queryKey: ["replay-timeline", selectedDate],
    queryFn: () => fetchReplayTimeline(selectedDate!),
    enabled: !!selectedDate,
    staleTime: 60000,
  });

  // Select a date
  const selectDate = useCallback((date: string) => {
    setSelectedDate(date);
    setState("date_selected");
    setRecentEvents([]);
    setSnapshot(null);
    // Start of IST trading day: 09:15
    setCurrentTs(`${date}T09:15:00.000000Z`);
    setHasMore(true);
  }, []);

  // Execute one step
  const doStep = useCallback(async () => {
    if (!selectedDate || steppingRef.current) return;
    steppingRef.current = true;

    try {
      const result = await fetchReplayStep(currentTs, DEFAULT_DELTA_MS, selectedDate);

      if (result.events.length > 0) {
        setRecentEvents((prev) => [...prev, ...result.events]);
      }
      if (result.snapshot) {
        setSnapshot(result.snapshot as ReplaySnapshot);
      }
      setCurrentTs(result.next_ts);
      setHasMore(result.has_more);

      if (!result.has_more) {
        setState("done");
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
      }
    } finally {
      steppingRef.current = false;
    }
  }, [selectedDate, currentTs]);

  // Play
  const play = useCallback(() => {
    if (!selectedDate || state === "done") return;
    setState("playing");
  }, [selectedDate, state]);

  // Pause
  const pause = useCallback(() => {
    setState("paused");
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  // Step forward (manual)
  const stepForward = useCallback(() => {
    if (!selectedDate || state === "done") return;
    if (state === "idle") return;
    doStep();
  }, [selectedDate, state, doStep]);

  // Reset
  const reset = useCallback(() => {
    if (!selectedDate) return;
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setState("date_selected");
    setRecentEvents([]);
    setSnapshot(null);
    setCurrentTs(`${selectedDate}T09:15:00.000000Z`);
    setHasMore(true);
  }, [selectedDate]);

  // Speed change
  const setSpeed = useCallback((newSpeed: PlaybackSpeed) => {
    setSpeedState(newSpeed);
  }, []);

  // Auto-stepping when playing
  useEffect(() => {
    if (state === "playing") {
      // Clear any existing interval
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      intervalRef.current = setInterval(() => {
        doStep();
      }, SPEED_INTERVAL[speed]);

      return () => {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
      };
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state, speed, doStep]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  return {
    state,
    dates,
    selectedDate,
    markers,
    recentEvents,
    snapshot,
    currentTs,
    speed,
    isLoadingDates,
    hasMore,
    selectDate,
    play,
    pause,
    stepForward,
    reset,
    setSpeed,
  };
}
