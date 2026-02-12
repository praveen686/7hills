# Plan: BRAHMASTRA Terminal — Institutional-Grade Trading Frontend

## Context

Build a commercial-grade, Bloomberg-class trading terminal for India FnO + Crypto. Must be blazingly fast, tightly coupled to the Rust execution engine, and designed for professional traders. The system already has:
- **Rust workspace**: 22 crates (executor, risk, data, L2 book, VPIN, circuit breakers, Zerodha/Binance connectors, protobuf contracts)
- **Python backend**: FastAPI with 10 route modules, 4 WebSocket channels, strategies, backtests
- **Existing frontend**: Next.js 14, React 18, 9 pages, TanStack Query — functional but not commercial-grade

## Architecture: Tauri 2.0 Desktop + Web Fallback

```
┌─────────────────────────────────────────────────────────────────┐
│  BRAHMASTRA Terminal (Tauri 2.0)                                │
│                                                                 │
│  ┌─────────────────────────┐   ┌─────────────────────────────┐  │
│  │  Rust Sidecar Process   │   │  WebView (Frontend)         │  │
│  │                         │   │                             │  │
│  │  quantlaxmi-terminal    │   │  React 19 + shadcn/ui       │  │
│  │  ├─ MarketDataHub       │◄─►│  ├─ WorkspaceManager        │  │
│  │  │  ├─ L2 Book mgmt    │IPC│  ├─ WebGL OrderbookRenderer │  │
│  │  │  ├─ Tick→Bar agg    │   │  ├─ lightweight-charts       │  │
│  │  │  └─ VPIN calc       │   │  ├─ Canvas DOM Ladder       │  │
│  │  ├─ ExecutionBridge     │   │  ├─ CommandPalette (⌘K)     │  │
│  │  │  ├─ SimExchange     │   │  ├─ KeyboardNavigator       │  │
│  │  │  └─ ZerodhaLive     │   │  ├─ Jotai atoms (state)     │  │
│  │  ├─ RiskEngine          │   │  └─ react-grid-layout       │  │
│  │  │  ├─ PreTrade gates  │   │                             │  │
│  │  │  ├─ CircuitBreaker  │   │  Rendering tiers:           │  │
│  │  │  └─ PostTrade DD    │   │  ├─ WebGL: orderbook, heat  │  │
│  │  ├─ StrategyManager     │   │  ├─ Canvas: tape, ladder    │  │
│  │  └─ Proto serializer    │   │  └─ DOM: menus, forms, tabs │  │
│  └─────────────────────────┘   └─────────────────────────────┘  │
│         ↕ Tauri IPC (~1μs)        ↕ Tauri Events (streaming)    │
└─────────────────────────────────────────────────────────────────┘
         ↕ WebSocket                    ↕ REST
   ┌─────────────────┐          ┌──────────────────┐
   │ Zerodha Kite WS │          │ Python FastAPI    │
   │ Binance WS      │          │ (backtest, research│
   └─────────────────┘          │  strategies, ML)  │
                                └──────────────────┘
```

### Why Tauri 2.0

| Property | Tauri | Electron | Pure Web |
|----------|-------|----------|----------|
| Binary size | ~10 MB | ~200 MB | N/A |
| Memory | ~30 MB | ~300 MB | Browser tab |
| Rust integration | In-process | IPC/FFI | WebSocket |
| Execution latency | ~1μs IPC | ~1ms IPC | ~5-50ms WS |
| Broker connection | Native TCP | Node child | Browser sandbox |
| Distribution | Signed installer | Signed installer | URL |

### Data Flow

```
Zerodha WS ──► quantlaxmi-data (Rust) ──► Tauri Event ──► WebGL Renderer
                 ├─ L2Book.apply_update()                    (orderbook)
                 ├─ BarAggregator.on_tick()     ──►         (candlestick)
                 └─ VpinCalculator.on_trade()   ──►         (VPIN gauge)

User clicks "Buy" ──► Tauri Command ──► quantlaxmi-risk ──► quantlaxmi-executor
                                         (4ns gate check)    (Zerodha API)
                                              ↓
                                         GateDecision ──► UI feedback (<1ms)
```

---

## Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Desktop shell | **Tauri 2.0** | Rust-native, 10MB binary, in-process IPC |
| Frontend framework | **React 19** | Ecosystem, shadcn/ui, proven for terminals |
| Component library | **shadcn/ui + Radix** | Accessible, composable, Bloomberg clone validated |
| State management | **Jotai** | Atomic, minimal rerenders, perfect for streaming data |
| CSS | **Tailwind CSS 4** | Utility-first, dark mode, JetBrains Mono |
| Charts | **lightweight-charts** | TradingView quality, existing dep, fast Canvas renderer |
| Orderbook/heatmap | **Custom WebGL** | 100K+ updates/sec, depth visualization, footprint |
| DOM ladder | **Custom Canvas** | One-click trading, price ladder with volume profile |
| Layout | **react-grid-layout** | Draggable, resizable panels, save/load workspaces |
| Command palette | **cmdk** | ⌘K palette, Bloomberg-style search, Vercel uses it |
| Data tables | **TanStack Table** | Virtual scrolling, sorting, filtering, 100K rows |
| Serialization | **protobuf-es** | Match Rust proto schemas, zero-copy where possible |
| Icons | **Lucide** | Consistent, tree-shakeable |

---

## Rust Crate: `quantlaxmi-terminal`

New Tauri plugin crate that wraps existing crates for the desktop app:

```rust
// New crate: rust/crates/quantlaxmi-terminal/
quantlaxmi-terminal/
├── Cargo.toml          // deps: tauri, quantlaxmi-data, -risk, -executor, -connectors-*
├── src/
│   ├── lib.rs          // Tauri plugin registration
│   ├── commands.rs     // #[tauri::command] handlers (sync request/response)
│   │   ├── get_portfolio()
│   │   ├── get_orderbook(symbol) → L2 snapshot
│   │   ├── place_order(intent) → OrderResult
│   │   ├── cancel_order(order_id)
│   │   ├── get_positions()
│   │   ├── get_risk_state()
│   │   ├── run_backtest(config) → backtest_id
│   │   ├── get_strategies()
│   │   └── search_symbols(query) → Symbol[]
│   ├── streams.rs      // Tauri event emitters (async push)
│   │   ├── stream_ticks(symbol)       → "tick:{symbol}"
│   │   ├── stream_orderbook(symbol)   → "book:{symbol}"
│   │   ├── stream_trades(symbol)      → "trade:{symbol}"
│   │   ├── stream_vpin(symbol)        → "vpin:{symbol}"
│   │   ├── stream_bars(symbol, tf)    → "bar:{symbol}:{tf}"
│   │   ├── stream_signals()           → "signal"
│   │   ├── stream_portfolio()         → "portfolio"
│   │   └── stream_risk()              → "risk"
│   ├── market_hub.rs   // Wraps quantlaxmi-data (L2Book, BarAgg, VPIN)
│   ├── exec_bridge.rs  // Wraps quantlaxmi-executor + risk pre-check
│   └── config.rs       // App configuration, workspace persistence
```

---

## Frontend Structure

```
terminal/                          (Tauri project root)
├── src-tauri/
│   ├── Cargo.toml                 // deps: quantlaxmi-terminal, tauri
│   ├── tauri.conf.json            // window config, permissions, auto-update
│   └── src/main.rs                // Tauri entry point
│
├── src/                           (React frontend)
│   ├── app/
│   │   ├── layout.tsx             // Root: fonts, theme, providers, keyboard handler
│   │   └── page.tsx               // Workspace renderer (no traditional routing — SPA)
│   │
│   ├── components/
│   │   ├── workspace/
│   │   │   ├── WorkspaceManager.tsx    // react-grid-layout orchestrator
│   │   │   ├── PanelFrame.tsx          // Generic panel wrapper (title, close, maximize)
│   │   │   ├── WorkspacePresets.ts     // Default layouts: Trading, Analysis, Backtest, Monitor
│   │   │   └── WorkspacePersistence.ts // Save/load to Tauri fs
│   │   │
│   │   ├── market/
│   │   │   ├── OrderbookPanel.tsx      // WebGL depth chart + bid/ask columns
│   │   │   ├── OrderbookRenderer.ts    // WebGL shader: heatmap + depth visualization
│   │   │   ├── DomLadder.tsx           // Canvas price ladder with one-click trading
│   │   │   ├── TapePanel.tsx           // Time & sales (virtual scrolling)
│   │   │   ├── ChartPanel.tsx          // lightweight-charts wrapper with indicators
│   │   │   ├── TickerBar.tsx           // Top bar: watchlist symbols with live prices
│   │   │   └── SymbolSearch.tsx        // Fuzzy symbol search (cmdk)
│   │   │
│   │   ├── trading/
│   │   │   ├── OrderEntry.tsx          // Buy/Sell panel (market, limit, SL, bracket)
│   │   │   ├── PositionsTable.tsx      // Open positions with live P&L
│   │   │   ├── OrdersTable.tsx         // Pending + executed orders
│   │   │   ├── TradesTable.tsx         // Filled trades history
│   │   │   └── QuickTrade.tsx          // Keyboard-driven rapid order entry
│   │   │
│   │   ├── risk/
│   │   │   ├── RiskDashboard.tsx       // Greeks, VaR, drawdown, exposure, breakers
│   │   │   ├── VpinGauge.tsx           // Real-time VPIN toxicity meter
│   │   │   ├── DrawdownChart.tsx       // Live drawdown curve
│   │   │   └── ExposureHeatmap.tsx     // Strategy × symbol exposure matrix
│   │   │
│   │   ├── strategy/
│   │   │   ├── StrategyPanel.tsx       // Strategy cards with live equity curves
│   │   │   ├── SignalFeed.tsx          // Real-time signal stream with Why drill-down
│   │   │   ├── StrategyDetail.tsx      // Single strategy deep-dive
│   │   │   └── WhyPanel.tsx            // Signal explanation drawer (gates, risk, fills)
│   │   │
│   │   ├── backtest/
│   │   │   ├── BacktestRunner.tsx      // Config form + launch + progress
│   │   │   ├── BacktestResults.tsx     // Equity curve, drawdown, monthly heatmap
│   │   │   ├── WalkForwardView.tsx     // Fold-by-fold OOS vs IS comparison
│   │   │   └── BacktestCompare.tsx     // Side-by-side strategy comparison
│   │   │
│   │   ├── research/
│   │   │   ├── FeatureImportance.tsx   // VSN weights, fANOVA, feature IC
│   │   │   ├── TFTDashboard.tsx        // Training progress, fold metrics, live loss
│   │   │   └── AlphaDecay.tsx          // Signal degradation tracking
│   │   │
│   │   ├── terminal/
│   │   │   ├── CommandPalette.tsx      // ⌘K Bloomberg-style command search
│   │   │   ├── KeyboardNavigator.tsx   // Global hotkeys (Shift+B=buy, Shift+S=sell, etc.)
│   │   │   ├── StatusBar.tsx           // Bottom: connection status, latency, clock, regime
│   │   │   └── AlertToast.tsx          // Non-blocking notifications (fills, breakers, signals)
│   │   │
│   │   └── ui/                        // shadcn/ui components (auto-generated)
│   │       ├── button.tsx, badge.tsx, card.tsx, dialog.tsx, ...
│   │       └── (standard shadcn components)
│   │
│   ├── hooks/
│   │   ├── useTauriStream.ts          // Subscribe to Tauri events (tick, book, signal, etc.)
│   │   ├── useTauriCommand.ts         // Call Tauri commands (place_order, get_positions, etc.)
│   │   ├── useOrderbook.ts            // L2 book state from stream
│   │   ├── usePortfolio.ts            // Portfolio state from stream
│   │   ├── useKeyboard.ts             // Global keyboard shortcut registry
│   │   └── useWorkspace.ts            // Layout state, panel registry
│   │
│   ├── stores/
│   │   ├── market.ts                  // Jotai atoms: orderbook, ticks, bars, VPIN
│   │   ├── portfolio.ts               // Jotai atoms: positions, equity, drawdown
│   │   ├── trading.ts                 // Jotai atoms: orders, fills, selected symbol
│   │   └── workspace.ts              // Jotai atoms: layout, active panels, theme
│   │
│   ├── renderers/
│   │   ├── orderbook-webgl/
│   │   │   ├── shaders.ts            // GLSL vertex + fragment shaders
│   │   │   ├── OrderbookRenderer.ts   // WebGL context, buffer management, draw loop
│   │   │   └── HeatmapRenderer.ts     // Historical orderbook heatmap overlay
│   │   └── dom-ladder/
│   │       └── LadderRenderer.ts      // Canvas-based price ladder with click zones
│   │
│   ├── lib/
│   │   ├── proto.ts                   // protobuf-es generated types (from .proto files)
│   │   ├── format.ts                  // Number formatting (Indian lakhs, crypto decimals)
│   │   └── keybindings.ts             // Default keybinding map
│   │
│   └── styles/
│       ├── globals.css                // Tailwind base + terminal theme
│       └── webgl.css                  // Canvas/WebGL container styles
│
├── package.json
├── tailwind.config.ts
├── tsconfig.json
└── next.config.mjs                    // (or vite.config.ts if we go Vite)
```

---

## Workspace Presets

### Trading Workspace (Default)
```
┌──────────────────────────────────────────────────────────────────────┐
│ TickerBar: NIFTY 24,150 ▲0.3% | BANKNIFTY 51,200 ▼0.1%            │
├──────────┬───────────────────┬───────────┬───────────────────────────┤
│          │                   │ Orderbook │  Order Entry              │
│   DOM    │   Chart (lwc)     │  (WebGL)  │  ┌─────────┐             │
│  Ladder  │   + indicators    │  depth +  │  │ BUY/SELL│             │
│          │                   │  heatmap  │  │ Qty/Px  │             │
│          │                   │           │  └─────────┘             │
├──────────┼───────────────────┴───────────┤  Positions               │
│  Signal  │      Tape (Time & Sales)      │  (live P&L)              │
│  Feed    │      filtered by size         │                          │
├──────────┴───────────────────────────────┴───────────────────────────┤
│ StatusBar: ● Connected | Latency 2ms | VPIN 0.42 | Regime: Trend    │
└──────────────────────────────────────────────────────────────────────┘
```

### Analysis Workspace
```
┌───────────────────────────────────────────────────┐
│ TickerBar                                         │
├─────────────────────┬─────────────────────────────┤
│  Strategy Panel     │  Equity Curves (overlay)    │
│  (cards + equity)   │  + Drawdown chart below     │
├─────────────────────┼─────────────────────────────┤
│  Feature Importance │  Walk-Forward Validation    │
│  (VSN + fANOVA)     │  (fold-by-fold OOS Sharpe)  │
├─────────────────────┼─────────────────────────────┤
│  Signal History     │  Risk Dashboard             │
│  (Why drill-down)   │  (Greeks, VaR, exposure)    │
└─────────────────────┴─────────────────────────────┘
```

### Backtest Workspace
```
┌───────────────────────────────────────────────────┐
│ TickerBar                                         │
├──────────────┬────────────────────────────────────┤
│  Backtest    │  Results: Equity + Drawdown        │
│  Config      │  ──────────────────────────────    │
│  ┌────────┐  │  Monthly Returns Heatmap           │
│  │Strategy│  │  ──────────────────────────────    │
│  │Dates   │  │  Trade List (sortable, filterable) │
│  │Params  │  │  ──────────────────────────────    │
│  │[RUN]   │  │  Compare: Strategy A vs B vs C     │
│  └────────┘  │                                    │
└──────────────┴────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
**Goal**: Tauri app boots, Rust IPC works, basic layout renders

**Files to create**:
- `terminal/` project scaffold (Tauri 2.0 + React 19 + Vite)
- `rust/crates/quantlaxmi-terminal/` crate
- Basic Tauri commands: `get_portfolio`, `search_symbols`
- Basic Tauri events: `stream_portfolio`
- `WorkspaceManager` + `PanelFrame` with react-grid-layout
- `CommandPalette` with cmdk
- `StatusBar` with connection status
- shadcn/ui theme (dark, JetBrains Mono, terminal colors)
- Jotai stores for portfolio + workspace

**Verification**: App launches, shows portfolio data from Rust, panels are draggable

### Phase 2: Market Data + Charts (Week 2-3)
**Goal**: Real-time price charts, orderbook, tape

**Files to create**:
- Wire `quantlaxmi-data` (L2Book, BarAggregator, VPIN) into terminal crate
- Tauri streams: `stream_ticks`, `stream_orderbook`, `stream_bars`, `stream_vpin`
- `ChartPanel` (lightweight-charts with OHLCV from Rust bars)
- `OrderbookPanel` + `OrderbookRenderer` (WebGL depth + heatmap)
- `DomLadder` (Canvas price ladder)
- `TapePanel` (virtual-scrolled time & sales)
- `TickerBar` (watchlist with live prices)
- `VpinGauge` (real-time toxicity)

**Verification**: Connect to Zerodha/Binance WS, see live orderbook + charts + tape

### Phase 3: Trading + Risk (Week 3-4)
**Goal**: Place orders, see positions, risk gates visible

**Files to create**:
- Wire `quantlaxmi-executor` + `quantlaxmi-risk` into terminal crate
- Tauri commands: `place_order`, `cancel_order`, `get_positions`
- `OrderEntry` (market, limit, SL, bracket orders)
- `QuickTrade` (keyboard: Shift+B buy, Shift+S sell, Enter confirm)
- `PositionsTable` (live P&L, unrealized + realized)
- `OrdersTable` + `TradesTable`
- `RiskDashboard` (Greeks, VaR, drawdown, circuit breakers)
- `ExposureHeatmap`
- `KeyboardNavigator` (global hotkeys)

**Verification**: Place simulated order, see it fill, position appears with live P&L, risk gates block if limits hit

### Phase 4: Strategy + Signals (Week 4-5)
**Goal**: Strategy monitoring, signal feed, Why panel

**Files to create**:
- Bridge to Python FastAPI for strategy data (Tauri sidecar or HTTP)
- `StrategyPanel` (cards with mini equity curves)
- `SignalFeed` (real-time signal stream)
- `WhyPanel` (drill into gate decisions, fills, risk alerts)
- `StrategyDetail` (full strategy deep-dive)
- `AlertToast` (non-blocking signal/fill/breaker notifications)

**Verification**: See live signals, drill into Why panel, strategy equity updates in real-time

### Phase 5: Backtest Platform (Week 5-6)
**Goal**: Run and visualize backtests from the terminal

**Files to create**:
- Bridge to Python backtest engine (HTTP to FastAPI)
- `BacktestRunner` (config form, strategy selection, date range)
- `BacktestResults` (equity curve, drawdown, monthly heatmap, trade table)
- `WalkForwardView` (fold-by-fold IS vs OOS)
- `BacktestCompare` (overlay multiple strategies)

**Verification**: Launch backtest from terminal, see progress, review results with full drill-down

### Phase 6: Research + Polish (Week 6-7)
**Goal**: TFT dashboard, feature importance, final polish

**Files to create**:
- `FeatureImportance` (VSN weights visualization, fANOVA chart)
- `TFTDashboard` (training progress, fold metrics)
- `AlphaDecay` (signal degradation over time)
- Workspace save/load to disk
- Window management (multi-monitor support via Tauri)
- Auto-update configuration
- Performance profiling and optimization pass

**Verification**: Full end-to-end workflow: monitor market → receive signal → inspect Why → place order → track P&L → review backtest → analyze features

---

## Key Design Principles

1. **WebSocket-first, not polling** — All real-time data flows through Tauri events (Rust → Frontend), not HTTP polling
2. **Three rendering tiers** — WebGL (orderbook, heatmap), Canvas (ladder, tape), DOM (everything else)
3. **SPA, not pages** — Single workspace with customizable panels, not separate routes
4. **Keyboard-first** — Every action has a hotkey. ⌘K command palette for discovery
5. **Zero network hop for execution** — Order placement goes Rust IPC → risk check → broker. No Python in the hot path
6. **Protobuf everywhere** — Same `.proto` schemas for Rust↔Frontend, Rust↔Python, Rust↔Broker
7. **Deterministic replay** — Every event is WAL-logged, UI can replay any trading day

---

## Files Modified in Existing Codebase

| File | Change |
|------|--------|
| `rust/Cargo.toml` | Add `quantlaxmi-terminal` to workspace members |
| `rust/crates/quantlaxmi-terminal/` | NEW: entire Tauri plugin crate |
| `terminal/` | NEW: entire Tauri desktop app (React frontend) |
| `quantlaxmi/engine/api/app.py` | Add CORS for Tauri origin, ensure backtest/research routes work |

The existing `ui/` directory is **NOT modified** — it continues to work as a standalone web dashboard. The new `terminal/` is a separate Tauri desktop app that reuses the Rust crates directly and bridges to Python FastAPI for strategy/backtest/research data.

---

## Verification Plan

1. **Unit**: Each Tauri command has a Rust test (`cargo test -p quantlaxmi-terminal`)
2. **Integration**: Tauri app launches, connects to simulated feed, places/cancels orders
3. **Visual**: Orderbook renders 20 levels at 60fps with live updates
4. **Latency**: Order placement < 5ms end-to-end (click → broker API call)
5. **Stress**: 10K orderbook updates/sec without frame drops
6. **E2E**: Full workflow — market data → signal → risk check → order → fill → P&L update
