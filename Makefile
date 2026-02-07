# QuantLaxmi — Production Makefile
# Usage: make test | make test-phase7 | make deploy-gate

SHELL := /bin/bash
PYTHON := source QuantLaxmi/env/bin/activate && python
PYTEST := source QuantLaxmi/env/bin/activate && python -m pytest
SRC := QuantLaxmi
TESTS := $(SRC)/tests

.PHONY: test test-fast test-phase1 test-phase2 test-phase3 test-phase4 \
        test-phase5 test-phase6 test-phase7 test-missingness test-parity \
        test-regime test-sanos test-ci deploy-gate lint

# -----------------------------------------------------------------------
# Full suite
# -----------------------------------------------------------------------

test:
	$(PYTEST) $(TESTS) -v --tb=short -q

test-fast:
	$(PYTEST) $(TESTS) -v --tb=short -q -m "not slow"

# -----------------------------------------------------------------------
# Phase-specific targets
# -----------------------------------------------------------------------

test-phase1:
	$(PYTEST) $(TESTS)/determinism $(TESTS)/risk $(TESTS)/causality $(TESTS)/costs -v --tb=short -q

test-phase2:
	$(PYTEST) $(TESTS)/events -v --tb=short -q

test-phase3:
	$(PYTEST) $(TESTS)/replay -v --tb=short -q

test-phase4:
	$(PYTEST) $(TESTS)/why_panel -v --tb=short -q

test-phase5:
	$(PYTEST) $(TESTS)/replay_ui_api -v --tb=short -q

test-phase6:
	$(PYTEST) $(TESTS)/phase6_diagnostics -v --tb=short -q

test-phase7:
	$(PYTEST) $(TESTS)/phase7_production -v --tb=short -q

# -----------------------------------------------------------------------
# Phase 7 sub-targets
# -----------------------------------------------------------------------

test-missingness:
	$(PYTEST) $(TESTS)/phase7_production/test_missingness.py -v --tb=short -q

test-parity:
	$(PYTEST) $(TESTS)/phase7_production/test_e2e_determinism.py -v --tb=short -q

test-regime:
	$(PYTEST) $(TESTS)/phase7_production/test_regime_state_machine.py -v --tb=short -q

test-sanos:
	$(PYTEST) $(TESTS)/phase7_production/test_sanos_determinism.py -v --tb=short -q

test-ci:
	$(PYTEST) $(TESTS)/phase7_production/test_ci_gates.py -v --tb=short -q

# -----------------------------------------------------------------------
# CI gates
# -----------------------------------------------------------------------

deploy-gate: test-parity test-missingness test-regime test-sanos test-ci
	@echo "=== ALL DEPLOY GATES PASSED ==="

# -----------------------------------------------------------------------
# Lint
# -----------------------------------------------------------------------

lint:
	$(PYTHON) -m ruff check $(SRC) --select=E,F,W --ignore=E501
