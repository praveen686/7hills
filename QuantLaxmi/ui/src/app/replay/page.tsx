"use client";

import { useCallback } from "react";
import { useReplay } from "@/hooks/useReplay";
import { useWhyPanel } from "@/hooks/useWhyPanel";
import { ReplayControls } from "@/components/replay/ReplayControls";
import { ReplayTimeline } from "@/components/replay/ReplayTimeline";
import { ReplaySnapshotPanel } from "@/components/replay/ReplaySnapshotPanel";
import { ReplayEventLog } from "@/components/replay/ReplayEventLog";
import { WhyPanel } from "@/components/WhyPanel";
import type { TimelineMarker, StepEvent } from "@/lib/types";

// ============================================================
// Replay Page — Time-travel through a trading day
//
// Operator selects a date, then plays through the day's event
// stream with controllable playback. Portfolio snapshots update
// and clicking trade markers opens the WhyPanel.
// ============================================================

export default function ReplayPage() {
  const replay = useReplay();
  const whyPanel = useWhyPanel();

  // Handle marker click from timeline — open WhyPanel
  const handleMarkerClick = useCallback(
    (marker: TimelineMarker) => {
      if (marker.strategy_id && marker.symbol && replay.selectedDate) {
        whyPanel.open({
          strategy_id: marker.strategy_id,
          symbol: marker.symbol,
          date: replay.selectedDate,
        });
      }
    },
    [whyPanel, replay.selectedDate]
  );

  // Handle event click from event log — open WhyPanel
  const handleEventClick = useCallback(
    (event: StepEvent) => {
      if (event.strategy_id && event.symbol && replay.selectedDate) {
        whyPanel.open({
          strategy_id: event.strategy_id,
          symbol: event.symbol,
          date: replay.selectedDate,
        });
      }
    },
    [whyPanel, replay.selectedDate]
  );

  return (
    <div className="space-y-4 animate-in">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-white">Replay</h1>
        <p className="text-sm text-gray-500 mt-1">
          Time-travel through a trading day — step through events and inspect every decision
        </p>
      </div>

      {/* Controls */}
      <ReplayControls
        state={replay.state}
        dates={replay.dates}
        selectedDate={replay.selectedDate}
        currentTs={replay.currentTs}
        speed={replay.speed}
        hasMore={replay.hasMore}
        isLoadingDates={replay.isLoadingDates}
        onSelectDate={replay.selectDate}
        onPlay={replay.play}
        onPause={replay.pause}
        onStep={replay.stepForward}
        onReset={replay.reset}
        onSetSpeed={replay.setSpeed}
      />

      {/* Timeline */}
      <ReplayTimeline
        markers={replay.markers}
        currentTs={replay.currentTs}
        selectedDate={replay.selectedDate}
        onMarkerClick={handleMarkerClick}
      />

      {/* Snapshot + Event Log Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ReplaySnapshotPanel snapshot={replay.snapshot} />
        <ReplayEventLog
          events={replay.recentEvents}
          onEventClick={handleEventClick}
        />
      </div>

      {/* Why Panel (side drawer) */}
      <WhyPanel
        isOpen={whyPanel.isOpen}
        chain={whyPanel.chain}
        isLoading={whyPanel.isLoading}
        error={whyPanel.error}
        onClose={whyPanel.close}
      />
    </div>
  );
}
