"""BRAHMASTRA FastAPI application.

Main entry point for the API server.  Initialises shared services
(MarketDataStore, BrahmastraState, StrategyRegistry, StrategyReader)
on startup and tears them down on shutdown via the ASGI lifespan
protocol.

Usage:
    uvicorn apps.brahmastra.api.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from brahmastra.state import BrahmastraState, DEFAULT_STATE_FILE
from brahmastra.services.strategy_reader import StrategyReader
from brahmastra.services.market_data import MarketDataService
from brahmastra.api.routes import portfolio, strategies, backtest, risk, market, research, signals, why_panel
from brahmastra.api.routes.backtest import BacktestTracker
from brahmastra.api import ws as ws_module
from brahmastra.services.wal_query import WalQueryService

from qlx.data.store import MarketDataStore
from qlx.strategy.registry import StrategyRegistry

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Lifespan — startup / shutdown
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise shared state on startup; clean up on shutdown."""
    logger.info("BRAHMASTRA API starting up...")

    # 1. Load portfolio state
    state_path = Path(
        app.state.state_file
        if hasattr(app.state, "state_file")
        else DEFAULT_STATE_FILE
    )
    app.state.brahmastra = BrahmastraState.load(state_path)
    app.state.state_path = state_path
    logger.info(
        "Loaded state: equity=%.4f  positions=%d  trades=%d",
        app.state.brahmastra.equity,
        len(app.state.brahmastra.positions),
        len(app.state.brahmastra.closed_trades),
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

    # 5. Backtest tracker
    app.state.backtest_tracker = BacktestTracker()

    # 6. WAL query service (Why Panel reads directly from event logs)
    wal_dir = Path("data/events")
    app.state.wal_query = WalQueryService(base_dir=wal_dir)
    logger.info("WalQueryService ready: base_dir=%s", wal_dir)

    # 7. WS connection manager
    app.state.ws_manager = ws_module.manager

    logger.info("BRAHMASTRA API ready.")

    yield  # ---- application runs ----

    # Shutdown
    logger.info("BRAHMASTRA API shutting down...")
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
        Path to the Brahmastra state JSON file.
        Defaults to ``data/brahmastra_state.json``.
    allowed_origins : list[str], optional
        CORS allowed origins.  Defaults to permissive local development
        origins.
    """
    application = FastAPI(
        title="BRAHMASTRA",
        description=(
            "BRAHMASTRA Trading System API -- India FnO portfolio management, "
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
        allowed_origins = [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:8080",
        ]

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
    application.include_router(ws_module.router)

    # Health check
    @application.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "brahmastra"}

    return application


# Default app instance for ``uvicorn apps.brahmastra.api.app:app``
app = create_app()
