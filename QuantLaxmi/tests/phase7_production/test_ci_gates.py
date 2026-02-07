"""Phase 7 — CI/CD gate meta-tests.

Verify that the CI/CD infrastructure (Makefile, GitHub Actions workflow,
test structure) is correctly configured. These are structural / meta-tests
that check file existence and deploy-gate blocking semantics.

10 tests across 3 classes:
  - TestCIGateConfiguration (5): file/dir existence
  - TestDeployGateBlocking  (3): gate logic simulation
  - TestRegressionGuard     (2): guard-rail directory existence
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.replay.comparator import ComparisonResult
from engine.services.data_quality import DQGateResult, DQCheckResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path("/home/ubuntu/Desktop/7hills")
PYTHON_SRC_ROOT = PROJECT_ROOT / "QuantLaxmi"


# ---------------------------------------------------------------------------
# Helper — deploy gate logic
# ---------------------------------------------------------------------------

def deploy_allowed(comparison: ComparisonResult, dq: DQGateResult) -> bool:
    """Deploy is allowed only when ALL gates pass."""
    return comparison.identical and dq.passed


# ===================================================================
# 1. TestCIGateConfiguration — file & directory existence (5 tests)
# ===================================================================

class TestCIGateConfiguration:
    """Verify that essential CI/CD artefacts exist in the repo."""

    def test_makefile_exists(self) -> None:
        makefile = PROJECT_ROOT / "Makefile"
        assert makefile.is_file(), f"Makefile not found at {makefile}"

    def test_workflow_exists(self) -> None:
        workflow = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
        assert workflow.is_file(), f"CI workflow not found at {workflow}"

    def test_phase7_test_dir_exists(self) -> None:
        phase7_dir = PYTHON_SRC_ROOT / "tests" / "phase7_production"
        assert phase7_dir.is_dir(), f"phase7_production dir missing: {phase7_dir}"

    def test_init_files_present(self) -> None:
        init = PYTHON_SRC_ROOT / "tests" / "phase7_production" / "__init__.py"
        assert init.is_file(), f"__init__.py missing in phase7_production: {init}"

    def test_conftest_present(self) -> None:
        conftest = PYTHON_SRC_ROOT / "tests" / "phase7_production" / "conftest.py"
        assert conftest.is_file(), f"conftest.py missing in phase7_production: {conftest}"


# ===================================================================
# 2. TestDeployGateBlocking — gate semantics simulation (3 tests)
# ===================================================================

class TestDeployGateBlocking:
    """Simulate deploy-gate blocking logic using real dataclasses."""

    def test_parity_failure_blocks(self) -> None:
        """If replay parity fails (identical=False), deploy must be blocked."""
        comparison = ComparisonResult(identical=False, total_compared=100)
        dq = DQGateResult(
            passed=True,
            checks=[DQCheckResult(passed=True, check_type="index_close", detail="ok")],
        )
        assert not deploy_allowed(comparison, dq), (
            "Deploy should be BLOCKED when replay parity fails"
        )

    def test_missingness_failure_blocks(self) -> None:
        """If any DQ gate check fails (passed=False), deploy must be blocked."""
        comparison = ComparisonResult(identical=True, total_compared=50)
        dq = DQGateResult(
            passed=False,
            checks=[
                DQCheckResult(passed=False, check_type="min_strikes",
                              detail="Only 2 strikes", severity="block"),
            ],
        )
        assert not deploy_allowed(comparison, dq), (
            "Deploy should be BLOCKED when DQ gate fails"
        )

    def test_all_pass_allows(self) -> None:
        """When all gates pass, deploy should be allowed."""
        comparison = ComparisonResult(identical=True, total_compared=200)
        dq = DQGateResult(
            passed=True,
            checks=[
                DQCheckResult(passed=True, check_type="index_close", detail="ok"),
                DQCheckResult(passed=True, check_type="min_strikes", detail="22 strikes"),
                DQCheckResult(passed=True, check_type="min_oi", detail="Max OI 5000"),
            ],
        )
        assert deploy_allowed(comparison, dq), (
            "Deploy should be ALLOWED when all gates pass"
        )


# ===================================================================
# 3. TestRegressionGuard — guard-rail directory existence (2 tests)
# ===================================================================

class TestRegressionGuard:
    """Verify that regression guard-rail test directories exist."""

    def test_cost_model_tests_exist(self) -> None:
        """The costs/ test directory must exist (CostModel regression tests)."""
        costs_dir = PYTHON_SRC_ROOT / "tests" / "costs"
        assert costs_dir.is_dir(), (
            f"costs test directory missing: {costs_dir}. "
            "CostModel regression tests are required for deploy safety."
        )

    def test_causality_poison_test_exists(self) -> None:
        """The causality/ test directory must exist (poison-future tests)."""
        causality_dir = PYTHON_SRC_ROOT / "tests" / "causality"
        assert causality_dir.is_dir(), (
            f"causality test directory missing: {causality_dir}. "
            "Poison-future causality tests are required for deploy safety."
        )
