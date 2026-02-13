import type { LayoutItem } from "@/stores/workspace";

// ---------------------------------------------------------------------------
// Workspace IDs — re-export for convenience
// ---------------------------------------------------------------------------

export type WorkspaceId = "trading" | "analysis" | "backtest" | "monitor";

// ---------------------------------------------------------------------------
// Panel type registry — every valid panel ID and its display name
// ---------------------------------------------------------------------------

export const PANEL_TITLES: Record<string, string> = {
  chart: "Chart",
  orderbook: "Order Book",
  dom: "DOM Ladder",
  orderEntry: "Order Entry",
  positions: "Positions",
  tape: "Time & Sales",
  signalFeed: "Signal Feed",
  strategy: "Strategy",
  equityCurve: "Equity Curve",
  featureImportance: "Feature Importance",
  walkForward: "Walk-Forward",
  risk: "Risk",
  backtestConfig: "Backtest Config",
  backtestResults: "Backtest Results",
  monthlyHeatmap: "Monthly Heatmap",
  tradeList: "Trade List",
  compare: "Compare",
  allStrategies: "All Strategies",
  allPositions: "All Positions",
  riskDashboard: "Risk Dashboard",
  signals: "Signals",
  orders: "Orders",
  modelPredictions: "Model Predictions",
};

// ---------------------------------------------------------------------------
// Preset layouts (12-column grid)
// ---------------------------------------------------------------------------

export const WORKSPACE_PRESETS: Record<WorkspaceId, LayoutItem[]> = {
  /**
   * Trading workspace
   * Large chart, orderbook, DOM ladder, order entry, positions, tape, signal feed
   */
  trading: [
    { i: "chart", x: 0, y: 0, w: 6, h: 6, minW: 4, minH: 3 },
    { i: "orderbook", x: 6, y: 0, w: 3, h: 6, minW: 2, minH: 3 },
    { i: "dom", x: 9, y: 0, w: 3, h: 6, minW: 2, minH: 3 },
    { i: "orderEntry", x: 9, y: 6, w: 3, h: 4, minW: 2, minH: 3 },
    { i: "positions", x: 0, y: 6, w: 5, h: 4, minW: 3, minH: 2 },
    { i: "tape", x: 5, y: 6, w: 4, h: 4, minW: 2, minH: 2 },
    { i: "signalFeed", x: 0, y: 10, w: 12, h: 3, minW: 4, minH: 2 },
  ],

  /**
   * Analysis workspace
   * Strategy panel, equity curves, feature importance, walk-forward, risk
   */
  analysis: [
    { i: "strategy", x: 0, y: 0, w: 4, h: 5, minW: 3, minH: 3 },
    { i: "equityCurve", x: 4, y: 0, w: 8, h: 5, minW: 4, minH: 3 },
    { i: "featureImportance", x: 0, y: 5, w: 4, h: 5, minW: 3, minH: 3 },
    { i: "modelPredictions", x: 4, y: 5, w: 4, h: 5, minW: 3, minH: 3 },
    { i: "risk", x: 8, y: 5, w: 4, h: 5, minW: 3, minH: 3 },
  ],

  /**
   * Backtest workspace
   * Config, results, monthly heatmap, trade list, compare
   */
  backtest: [
    { i: "backtestConfig", x: 0, y: 0, w: 3, h: 6, minW: 2, minH: 4 },
    { i: "backtestResults", x: 3, y: 0, w: 5, h: 6, minW: 3, minH: 3 },
    { i: "monthlyHeatmap", x: 8, y: 0, w: 4, h: 6, minW: 3, minH: 3 },
    { i: "tradeList", x: 0, y: 6, w: 8, h: 4, minW: 4, minH: 2 },
    { i: "compare", x: 8, y: 6, w: 4, h: 4, minW: 3, minH: 2 },
  ],

  /**
   * Monitor workspace
   * All strategies, all positions, risk dashboard, signals
   */
  monitor: [
    { i: "allStrategies", x: 0, y: 0, w: 6, h: 5, minW: 4, minH: 3 },
    { i: "allPositions", x: 6, y: 0, w: 6, h: 5, minW: 4, minH: 3 },
    { i: "riskDashboard", x: 0, y: 5, w: 6, h: 5, minW: 4, minH: 3 },
    { i: "signals", x: 6, y: 5, w: 6, h: 5, minW: 4, minH: 3 },
  ],
};
