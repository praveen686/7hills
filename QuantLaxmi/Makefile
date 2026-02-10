.PHONY: help install test test-unit lint format run-api run-ui run-engine collect-nse collect-telegram rust-build rust-test rust-check

help:                     ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:                  ## Install Python package + deps
	pip install -e ".[dev,ml,zerodha,telegram]"

test:                     ## Run all Python tests
	pytest tests/ -x -q

test-unit:                ## Run unit tests only
	pytest tests/ -x -q -m "not integration"

lint:                     ## Check code quality (ruff + mypy)
	ruff check quantlaxmi/ tests/
	mypy quantlaxmi/ --ignore-missing-imports

format:                   ## Auto-format code
	ruff format quantlaxmi/ tests/
	ruff check --fix quantlaxmi/ tests/

run-api:                  ## Start FastAPI backend
	uvicorn quantlaxmi.engine.api.app:app --host 0.0.0.0 --port 8000 --reload

run-ui:                   ## Start Next.js frontend
	cd ui && npm run dev

run-engine:               ## Run paper trading engine
	python -m quantlaxmi.engine paper --once

collect-nse:              ## Run NSE daily data collection
	python -m quantlaxmi.data.collectors.nse_daily collect

collect-telegram:         ## Download Telegram tick data
	python quantlaxmi/data/collectors/runner/tasks.py

rust-build:               ## Build Rust workspace
	cd rust && cargo build --release

rust-test:                ## Run Rust tests
	cd rust && cargo test

rust-check:               ## Check Rust compilation
	cd rust && cargo check
