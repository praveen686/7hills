# Contributing

Guidelines for contributing to QuantLaxmi.

## Branch Naming

Use prefixed branch names:

```
feat/tft-attention-heads       New feature or strategy
fix/cost-model-slippage        Bug fix
research/s26-order-flow        Research exploration
refactor/feature-builder       Code restructuring
```

## Code Style

- **Formatter/Linter**: [ruff](https://github.com/astral-sh/ruff), configured at 100 character line width
- **Type hints**: Encouraged on all public functions and class attributes
- **Docstrings**: Required for public classes and functions (Google style)
- **Imports**: Sorted by ruff (isort-compatible)

Run formatting and linting before committing:

```bash
make format    # Auto-format with ruff
make lint      # Check with ruff + mypy
```

## Testing

### Requirements

- All new code must have corresponding tests
- All tests must pass before opening a PR
- Test files go in `tests/` and mirror the source structure

### Running Tests

```bash
make test          # Full suite (~1,290 tests)
make test-unit     # Unit tests only (no live API calls)

# Run a specific test file
pytest tests/test_my_feature.py -v

# Run tests matching a pattern
pytest tests/ -k "test_sharpe" -v
```

### Test Conventions

- Use `pytest` (not unittest)
- Fixtures go in `tests/conftest.py` or local `conftest.py` files
- Mark integration tests with `@pytest.mark.integration`
- Use `pytest.approx()` for floating-point comparisons
- For IEEE 754 boundary values, use explicit tolerances (e.g., `rtol=1e-10, atol=1e-12`)

## Pre-Commit Hooks

Install pre-commit hooks after cloning:

```bash
pip install pre-commit
pre-commit install
```

This runs ruff formatting and linting automatically on every commit.

## Pull Request Process

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feat/my-feature main
   ```

2. **Make changes** and ensure tests pass:
   ```bash
   make test
   make lint
   ```

3. **Push and create PR**:
   ```bash
   git push -u origin feat/my-feature
   gh pr create --title "feat: description" --body "Summary of changes"
   ```

4. **PR review checklist**:
   - All tests pass
   - No ruff/mypy warnings
   - New features have tests
   - Backtests include realistic costs
   - No look-ahead bias (TimeGuard enforced)

## Backtest Integrity Rules

These rules are non-negotiable. Violations will block a PR.

### 1. No Future Data

All features and signals must be strictly causal. A signal generated at time `t` may only use data from time `t` and earlier. Use `TimeGuard` to enforce this.

### 2. Always Specify Costs

Every backtest must use a `CostModel` with realistic per-leg costs:

```python
# Index options (per-leg, in index points)
cost_nifty = CostModel(fixed_cost_per_leg=3.0)       # NIFTY
cost_bnf = CostModel(fixed_cost_per_leg=5.0)         # BANKNIFTY

# NOT this (bps of spot is wrong for index options):
# CostModel(commission_bps=5)  # WRONG
```

### 3. Sharpe Ratio Protocol

- Use `ddof=1` for standard deviation (unbiased estimator)
- Annualize with `sqrt(252)` (trading days)
- Include all calendar days (flat days count as 0 return)
- Apply T+1 signal lag (signal at close of day `t`, entry at close of day `t+1`)

### 4. Walk-Forward Validation

Strategies must be validated out-of-sample using walk-forward analysis. In-sample Sharpe alone is insufficient for production consideration.

## Adding a New Strategy

1. Create a directory: `quantlaxmi/strategies/s{N}_{name}/`
2. Implement `strategy.py` extending `BaseStrategy`
3. Register in `quantlaxmi/strategies/registry.py`
4. Add tests in `tests/test_s{N}_{name}.py`
5. Run walk-forward validation in `research/`
6. Document in `docs/strategies/`

## Adding New Features

1. Create or extend a feature builder in `quantlaxmi/features/`
2. If it's a new group, register it in `MegaFeatureBuilder` (`quantlaxmi/features/mega.py`)
3. Ensure all computations are causal (no lookahead)
4. Add tests verifying correctness and edge cases (NaN handling, empty inputs)
