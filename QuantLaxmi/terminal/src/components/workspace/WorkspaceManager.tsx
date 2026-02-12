import { useCallback, useMemo } from "react";
import { Responsive, WidthProvider } from "react-grid-layout";
import { useAtom, useAtomValue } from "jotai";
import type { Layout } from "react-grid-layout";

import "react-grid-layout/css/styles.css";

import { activeWorkspaceAtom, layoutAtom } from "@/stores/workspace";
import { WORKSPACE_PRESETS, PANEL_TITLES } from "@/components/workspace/WorkspacePresets";
import { PanelFrame } from "@/components/workspace/PanelFrame";

// Panel components
import { ChartPanel } from "@/components/market/ChartPanel";
import { OrderbookPanel } from "@/components/market/OrderbookPanel";
import { DomLadder } from "@/components/market/DomLadder";
import { TapePanel } from "@/components/market/TapePanel";
import { OrderEntry } from "@/components/trading/OrderEntry";
import { PositionsTable } from "@/components/trading/PositionsTable";
import { OrdersTable } from "@/components/trading/OrdersTable";
import { TradesTable } from "@/components/trading/TradesTable";
import { SignalFeed } from "@/components/strategy/SignalFeed";
import { StrategyPanel } from "@/components/strategy/StrategyPanel";
import { RiskDashboard } from "@/components/risk/RiskDashboard";
import { DrawdownChart } from "@/components/risk/DrawdownChart";
import { ExposureHeatmap } from "@/components/risk/ExposureHeatmap";
import { FeatureImportance } from "@/components/research/FeatureImportance";
import { WalkForwardView } from "@/components/backtest/WalkForwardView";
import { BacktestRunner } from "@/components/backtest/BacktestRunner";
import { BacktestResults } from "@/components/backtest/BacktestResults";
import { BacktestCompare } from "@/components/backtest/BacktestCompare";

import {
  BarChart3,
  BookOpen,
  Layers,
  ClipboardList,
  Activity,
  Radio,
  FlaskConical,
  TrendingUp,
  BarChart2,
  ArrowLeftRight,
  Shield,
  Settings,
  FileText,
  CalendarDays,
  List,
  GitCompare,
  LayoutGrid,
  Wallet,
  AlertTriangle,
  Zap,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Responsive grid with auto width
// ---------------------------------------------------------------------------

const ResponsiveGridLayout = WidthProvider(Responsive);

// ---------------------------------------------------------------------------
// Panel icon map
// ---------------------------------------------------------------------------

const PANEL_ICONS: Record<string, React.ReactNode> = {
  chart: <BarChart3 size={14} />,
  orderbook: <BookOpen size={14} />,
  dom: <Layers size={14} />,
  orderEntry: <ClipboardList size={14} />,
  positions: <Activity size={14} />,
  tape: <Radio size={14} />,
  signalFeed: <Zap size={14} />,
  strategy: <FlaskConical size={14} />,
  equityCurve: <TrendingUp size={14} />,
  featureImportance: <BarChart2 size={14} />,
  walkForward: <ArrowLeftRight size={14} />,
  risk: <Shield size={14} />,
  backtestConfig: <Settings size={14} />,
  backtestResults: <FileText size={14} />,
  monthlyHeatmap: <CalendarDays size={14} />,
  tradeList: <List size={14} />,
  compare: <GitCompare size={14} />,
  allStrategies: <LayoutGrid size={14} />,
  allPositions: <Wallet size={14} />,
  riskDashboard: <AlertTriangle size={14} />,
  signals: <Zap size={14} />,
  orders: <ClipboardList size={14} />,
};

// ---------------------------------------------------------------------------
// Placeholder panel content — replaced as real panels are built
// ---------------------------------------------------------------------------

function PanelPlaceholder({ panelId }: { panelId: string }) {
  return (
    <div className="flex items-center justify-center h-full text-terminal-muted text-xs font-mono">
      <span className="opacity-50">{PANEL_TITLES[panelId] ?? panelId}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Panel component registry
// Components are lazy-resolved here. As real panel components are built,
// replace PanelPlaceholder with the real import.
// ---------------------------------------------------------------------------

function renderPanelContent(panelId: string): React.ReactNode {
  switch (panelId) {
    case "chart":
    case "equityCurve":
      return <ChartPanel />;
    case "orderbook":
      return <OrderbookPanel />;
    case "dom":
      return <DomLadder />;
    case "tape":
      return <TapePanel />;
    case "orderEntry":
      return <OrderEntry />;
    case "positions":
    case "allPositions":
      return <PositionsTable />;
    case "orders":
      return <OrdersTable />;
    case "tradeList":
      return <TradesTable />;
    case "signalFeed":
    case "signals":
      return <SignalFeed />;
    case "strategy":
    case "allStrategies":
      return <StrategyPanel />;
    case "risk":
      return <DrawdownChart />;
    case "riskDashboard":
      return <RiskDashboard />;
    case "featureImportance":
      return <FeatureImportance />;
    case "walkForward":
      return <WalkForwardView />;
    case "backtestConfig":
      return <BacktestRunner />;
    case "backtestResults":
      return <BacktestResults />;
    case "monthlyHeatmap":
      return <ExposureHeatmap />;
    case "compare":
      return <BacktestCompare />;
    default:
      return <PanelPlaceholder panelId={panelId} />;
  }
}

// ---------------------------------------------------------------------------
// WorkspaceManager
// ---------------------------------------------------------------------------

export function WorkspaceManager() {
  const activeWorkspace = useAtomValue(activeWorkspaceAtom);
  const [layout, setLayout] = useAtom(layoutAtom);

  // When workspace changes, load the corresponding preset
  const currentLayout = useMemo(() => {
    const preset = WORKSPACE_PRESETS[activeWorkspace];
    return preset ?? layout;
  }, [activeWorkspace, layout]);

  // Persist layout changes from drag/resize
  const handleLayoutChange = useCallback(
    (newLayout: Layout[]) => {
      setLayout(
        newLayout.map((item) => ({
          i: item.i,
          x: item.x,
          y: item.y,
          w: item.w,
          h: item.h,
          minW: item.minW,
          minH: item.minH,
        })),
      );
    },
    [setLayout],
  );

  return (
    <ResponsiveGridLayout
      className="layout"
      layouts={{ lg: currentLayout }}
      breakpoints={{ lg: 1200, md: 996, sm: 768, xs: 480 }}
      cols={{ lg: 12, md: 12, sm: 6, xs: 4 }}
      rowHeight={60}
      margin={[4, 4]}
      containerPadding={[4, 4]}
      draggableHandle=".drag-handle"
      onLayoutChange={handleLayoutChange}
      compactType="vertical"
      isResizable
      isDraggable
    >
      {currentLayout.map((item) => (
        <div key={item.i} className="overflow-hidden">
          <PanelFrame
            id={item.i}
            title={PANEL_TITLES[item.i] ?? item.i}
            icon={PANEL_ICONS[item.i]}
          >
            {renderPanelContent(item.i)}
          </PanelFrame>
        </div>
      ))}
    </ResponsiveGridLayout>
  );
}
