"""QuantLaxmi Engine FastAPI application.

Main entry point for the API server.  Initialises shared services
(MarketDataStore, PortfolioState, StrategyRegistry, StrategyReader)
on startup and tears them down on shutdown via the ASGI lifespan
protocol.

Usage:
    uvicorn apps.engine.api.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from quantlaxmi.engine.state import PortfolioState, DEFAULT_STATE_FILE
from quantlaxmi.engine.services.strategy_reader import StrategyReader
from quantlaxmi.engine.services.market_data import MarketDataService
from quantlaxmi.engine.api.routes import portfolio, strategies, backtest, risk, market, research, signals, why_panel, replay, diagnostics
from quantlaxmi.engine.api.routes.backtest import BacktestTracker
from quantlaxmi.engine.api import ws as ws_module
from quantlaxmi.engine.services.wal_query import WalQueryService
from quantlaxmi.engine.services.replay_service import ReplayService
from quantlaxmi.engine.services.trade_analytics import TradeAnalyticsService
from quantlaxmi.engine.services.missed_opportunity import MissedOpportunityService
from quantlaxmi.engine.services.ars_surface import ARSSurfaceService

from quantlaxmi.data._paths import BACKTEST_RESULTS_DIR, EVENTS_DIR
from quantlaxmi.data.store import MarketDataStore
from quantlaxmi.strategies.registry import StrategyRegistry

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Lifespan — startup / shutdown
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise shared state on startup; clean up on shutdown."""
    logger.info("QuantLaxmi Engine API starting up...")

    # 1. Load portfolio state
    state_path = Path(
        app.state.state_file
        if hasattr(app.state, "state_file")
        else DEFAULT_STATE_FILE
    )
    app.state.engine = PortfolioState.load(state_path)
    app.state.state_path = state_path
    logger.info(
        "Loaded state: equity=%.4f  positions=%d  trades=%d",
        app.state.engine.equity,
        len(app.state.engine.positions),
        len(app.state.engine.closed_trades),
    )

    # 2. MarketDataStore (DuckDB)
    store = MarketDataStore()
    market_svc = MarketDataService(store)
    app.state.market_data_service = market_svc
    logger.info("MarketDataStore ready: %s", store.summary())

    # 3. Strategy registry (discover strategies)
    registry = StrategyRegistry()
    try:
        registry.discover()
        logger.info(
            "Discovered %d strategies: %s", len(registry), registry.list_ids()
        )
    except Exception as exc:
        logger.warning("Strategy discovery failed (non-fatal): %s", exc)
    app.state.strategy_registry = registry

    # 4. Strategy state reader
    app.state.strategy_reader = StrategyReader()

    # 5. Backtest tracker (persisted to data/results/backtest_results/)
    bt_persist_dir = BACKTEST_RESULTS_DIR
    app.state.backtest_tracker = BacktestTracker(persist_dir=bt_persist_dir)
    logger.info("BacktestTracker ready: persist_dir=%s, loaded=%d jobs",
                bt_persist_dir, len(app.state.backtest_tracker._jobs))

    # 6. WAL query service (Why Panel reads directly from event logs)
    wal_dir = EVENTS_DIR
    app.state.wal_query = WalQueryService(base_dir=wal_dir)
    logger.info("WalQueryService ready: base_dir=%s", wal_dir)

    # 6b. Replay service (time-travel playback over event logs)
    app.state.replay_service = ReplayService(base_dir=wal_dir)
    logger.info("ReplayService ready: base_dir=%s", wal_dir)

    # 6c. Trade analytics
    app.state.trade_analytics = TradeAnalyticsService(store=store)
    # 6d. Missed opportunity
    app.state.missed_opportunity = MissedOpportunityService(base_dir=wal_dir, store=store)
    # 6e. ARS surface
    app.state.ars_surface = ARSSurfaceService(base_dir=wal_dir)
    logger.info("Diagnostics services ready (TradeAnalytics, MissedOpportunity, ARSSurface)")

    # 7. WS connection manager
    app.state.ws_manager = ws_module.manager

    logger.info("QuantLaxmi Engine API ready.")

    yield  # ---- application runs ----

    # Shutdown
    logger.info("QuantLaxmi Engine API shutting down...")
    market_svc.close()
    logger.info("Shutdown complete.")


# ------------------------------------------------------------------
# App factory
# ------------------------------------------------------------------

def create_app(
    state_file: str | Path | None = None,
    allowed_origins: list[str] | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Parameters
    ----------
    state_file : str | Path, optional
        Path to the portfolio state JSON file.
        Defaults to ``data/state/portfolio.json``.
    allowed_origins : list[str], optional
        CORS allowed origins.  Defaults to permissive local development
        origins.
    """
    application = FastAPI(
        title="QuantLaxmi Engine",
        description=(
            "QuantLaxmi Engine API -- India FnO portfolio management, "
            "strategy monitoring, backtesting, risk analytics, and real-time "
            "market data."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # Store config on app.state before lifespan runs
    if state_file is not None:
        application.state.state_file = str(state_file)

    # CORS
    if allowed_origins is None:
        allowed_origins = ["*"]

    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    application.include_router(portfolio.router)
    application.include_router(strategies.router)
    application.include_router(backtest.router)
    application.include_router(risk.router)
    application.include_router(market.router)
    application.include_router(research.router)
    application.include_router(signals.router)
    application.include_router(why_panel.router)
    application.include_router(replay.router)
    application.include_router(diagnostics.router)
    application.include_router(ws_module.router)

    # Health check
    @application.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "engine"}

    return application


# Default app instance for ``uvicorn apps.engine.api.app:app``
app = create_app()
