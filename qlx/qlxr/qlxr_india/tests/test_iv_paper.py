"""Comprehensive tests for IV mean-reversion paper trading module.

Tests cover data types, serialization, state management, signal logic,
position management, performance metrics, and full trading cycle simulation.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from _archive.india_fno_legacy.paper_state import (
    ClosedTrade,
    IVObservation,
    PaperPosition,
    PaperState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_obs(
    day: int,
    spot: float = 24000.0,
    atm_iv: float = 0.14,
    atm_var: float | None = None,
    forward: float | None = None,
    sanos_ok: bool = True,
) -> IVObservation:
    """Build a deterministic IVObservation for testing.

    `day` is used to generate a YYYY-MM-DD date string (2026-01-01 + day offset).
    """
    d = f"2026-01-{day:02d}"
    return IVObservation(
        date=d,
        spot=spot,
        atm_iv=atm_iv,
        atm_var=atm_var if atm_var is not None else atm_iv ** 2,
        forward=forward if forward is not None else spot * 1.001,
        sanos_ok=sanos_ok,
    )


def make_obs_iso(
    date_str: str,
    spot: float = 24000.0,
    atm_iv: float = 0.14,
    sanos_ok: bool = True,
) -> IVObservation:
    """Build an IVObservation with an explicit ISO date string."""
    return IVObservation(
        date=date_str,
        spot=spot,
        atm_iv=atm_iv,
        atm_var=atm_iv ** 2,
        forward=spot * 1.001,
        sanos_ok=sanos_ok,
    )


def build_history(n: int, base_iv: float = 0.12, iv_step: float = 0.001) -> list[IVObservation]:
    """Build a list of n observations with linearly increasing IV."""
    obs_list = []
    for i in range(1, n + 1):
        iv = base_iv + iv_step * (i - 1)
        obs_list.append(make_obs(day=i, atm_iv=iv))
    return obs_list


def state_with_history(n: int, **kwargs) -> PaperState:
    """Return a PaperState pre-loaded with n observations."""
    state = PaperState(**kwargs)
    for obs in build_history(n):
        state.append_observation(obs)
    return state


# ---------------------------------------------------------------------------
# 1. IVObservation: to_dict / from_dict round-trip
# ---------------------------------------------------------------------------

class TestIVObservation:
    def test_round_trip(self):
        obs = make_obs(day=5, spot=24500.0, atm_iv=0.18, sanos_ok=False)
        d = obs.to_dict()
        restored = IVObservation.from_dict(d)
        assert restored.date == obs.date
        assert restored.spot == obs.spot
        assert restored.atm_iv == obs.atm_iv
        assert restored.atm_var == obs.atm_var
        assert restored.forward == obs.forward
        assert restored.sanos_ok == obs.sanos_ok

    def test_to_dict_keys(self):
        obs = make_obs(day=1)
        d = obs.to_dict()
        expected_keys = {"date", "spot", "atm_iv", "atm_var", "forward", "sanos_ok", "symbol"}
        assert set(d.keys()) == expected_keys

    def test_from_dict_coerces_types(self):
        """Numeric strings should be coerced to float."""
        d = {
            "date": "2026-01-01",
            "spot": "24000",
            "atm_iv": "0.14",
            "atm_var": "0.0196",
            "forward": "24024",
            "sanos_ok": 1,
        }
        obs = IVObservation.from_dict(d)
        assert isinstance(obs.spot, float)
        assert isinstance(obs.atm_iv, float)
        assert isinstance(obs.sanos_ok, bool)


# ---------------------------------------------------------------------------
# 2. PaperPosition: to_dict / from_dict round-trip
# ---------------------------------------------------------------------------

class TestPaperPosition:
    def test_round_trip(self):
        pos = PaperPosition(
            symbol="NIFTY",
            entry_date="2026-01-10",
            entry_spot=24100.0,
            entry_iv=0.16,
            iv_pctile=0.85,
            hold_days=3,
        )
        d = pos.to_dict()
        restored = PaperPosition.from_dict(d)
        assert restored.symbol == pos.symbol
        assert restored.entry_date == pos.entry_date
        assert restored.entry_spot == pos.entry_spot
        assert restored.entry_iv == pos.entry_iv
        assert restored.iv_pctile == pos.iv_pctile
        assert restored.hold_days == pos.hold_days

    def test_default_hold_days(self):
        pos = PaperPosition(
            symbol="NIFTY",
            entry_date="2026-01-01",
            entry_spot=24000.0,
            entry_iv=0.14,
            iv_pctile=0.80,
        )
        assert pos.hold_days == 0


# ---------------------------------------------------------------------------
# 3. ClosedTrade: to_dict / from_dict round-trip
# ---------------------------------------------------------------------------

class TestClosedTrade:
    def test_round_trip(self):
        trade = ClosedTrade(
            symbol="NIFTY",
            entry_date="2026-01-10",
            exit_date="2026-01-15",
            entry_spot=24000.0,
            exit_spot=24200.0,
            entry_iv=0.18,
            exit_iv=0.13,
            iv_pctile=0.85,
            pnl_pct=0.0078,
            hold_days=5,
            exit_reason="iv_normalised",
        )
        d = trade.to_dict()
        restored = ClosedTrade.from_dict(d)
        assert restored.symbol == trade.symbol
        assert restored.entry_date == trade.entry_date
        assert restored.exit_date == trade.exit_date
        assert restored.entry_spot == trade.entry_spot
        assert restored.exit_spot == trade.exit_spot
        assert restored.entry_iv == trade.entry_iv
        assert restored.exit_iv == trade.exit_iv
        assert restored.iv_pctile == trade.iv_pctile
        assert math.isclose(restored.pnl_pct, trade.pnl_pct)
        assert restored.hold_days == trade.hold_days
        assert restored.exit_reason == trade.exit_reason

    def test_to_dict_keys(self):
        trade = ClosedTrade(
            symbol="NIFTY",
            entry_date="2026-01-10",
            exit_date="2026-01-15",
            entry_spot=24000.0,
            exit_spot=24200.0,
            entry_iv=0.18,
            exit_iv=0.13,
            iv_pctile=0.85,
            pnl_pct=0.008,
            hold_days=5,
            exit_reason="max_hold",
        )
        d = trade.to_dict()
        expected_keys = {
            "symbol", "entry_date", "exit_date", "entry_spot", "exit_spot",
            "entry_iv", "exit_iv", "iv_pctile", "pnl_pct",
            "hold_days", "exit_reason",
        }
        assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# 4. PaperState save / load: JSON round-trip via temp file
# ---------------------------------------------------------------------------

class TestPaperStateSaveLoad:
    def test_round_trip_empty_state(self, tmp_path: Path):
        path = tmp_path / "state.json"
        state = PaperState()
        state.save(path)
        loaded = PaperState.load(path)
        assert loaded.iv_history == []
        assert loaded.position is None
        assert loaded.closed_trades == []
        assert loaded.equity == 1.0

    def test_round_trip_with_data(self, tmp_path: Path):
        path = tmp_path / "state.json"
        state = PaperState(
            iv_history=[make_obs(1), make_obs(2)],
            position=PaperPosition(
                symbol="NIFTY",
                entry_date="2026-01-02",
                entry_spot=24000.0,
                entry_iv=0.14,
                iv_pctile=0.82,
                hold_days=1,
            ),
            closed_trades=[
                ClosedTrade(
                    symbol="NIFTY",
                    entry_date="2025-12-20",
                    exit_date="2025-12-25",
                    entry_spot=23800.0,
                    exit_spot=24000.0,
                    entry_iv=0.17,
                    exit_iv=0.12,
                    iv_pctile=0.90,
                    pnl_pct=0.0079,
                    hold_days=5,
                    exit_reason="iv_normalised",
                ),
            ],
            equity=1.0079,
            started_at="2025-12-01T00:00:00+00:00",
            last_scan_date="2026-01-02",
            iv_lookback=20,
            entry_pctile=0.85,
            exit_pctile=0.45,
            max_hold_days=7,
            cost_bps=8.0,
        )
        state.save(path)
        loaded = PaperState.load(path)

        assert len(loaded.iv_history) == 2
        assert loaded.iv_history[0].date == "2026-01-01"
        assert loaded.position is not None
        assert loaded.position.entry_spot == 24000.0
        assert len(loaded.closed_trades) == 1
        assert math.isclose(loaded.equity, 1.0079)
        assert loaded.started_at == "2025-12-01T00:00:00+00:00"
        assert loaded.last_scan_date == "2026-01-02"
        # Config
        assert loaded.iv_lookback == 20
        assert loaded.entry_pctile == 0.85
        assert loaded.exit_pctile == 0.45
        assert loaded.max_hold_days == 7
        assert loaded.cost_bps == 8.0

    def test_load_missing_file_returns_fresh(self, tmp_path: Path):
        path = tmp_path / "nonexistent.json"
        state = PaperState.load(path)
        assert state.iv_history == []
        assert state.equity == 1.0

    def test_load_corrupt_json_returns_fresh(self, tmp_path: Path):
        path = tmp_path / "corrupt.json"
        path.write_text("{bad json!!!")
        state = PaperState.load(path)
        assert state.iv_history == []
        assert state.equity == 1.0

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "deep" / "nested" / "dir" / "state.json"
        state = PaperState()
        state.save(path)
        assert path.exists()
        loaded = PaperState.load(path)
        assert loaded.equity == 1.0

    def test_atomic_write_file_content(self, tmp_path: Path):
        """Verify the JSON on disk matches to_dict()."""
        path = tmp_path / "state.json"
        state = PaperState(equity=1.05)
        state.save(path)
        raw = json.loads(path.read_text())
        assert raw == state.to_dict()


# ---------------------------------------------------------------------------
# 5. append_observation
# ---------------------------------------------------------------------------

class TestAppendObservation:
    def test_normal_append(self):
        state = PaperState()
        state.append_observation(make_obs(1))
        state.append_observation(make_obs(2))
        state.append_observation(make_obs(3))
        assert len(state.iv_history) == 3
        assert state.iv_history[0].date == "2026-01-01"
        assert state.iv_history[2].date == "2026-01-03"

    def test_duplicate_date_updates(self):
        state = PaperState()
        state.append_observation(make_obs(1, spot=24000.0))
        state.append_observation(make_obs(2, spot=24100.0))
        # Duplicate date with different spot
        state.append_observation(make_obs(2, spot=24200.0))
        assert len(state.iv_history) == 2
        assert state.iv_history[-1].spot == 24200.0

    def test_out_of_order_skip(self):
        state = PaperState()
        state.append_observation(make_obs(3))
        state.append_observation(make_obs(5))
        # Day 4 is out-of-order (before day 5), should be skipped
        state.append_observation(make_obs(4))
        assert len(state.iv_history) == 2
        assert state.iv_history[-1].date == "2026-01-05"

    def test_append_to_empty(self):
        state = PaperState()
        state.append_observation(make_obs(15))
        assert len(state.iv_history) == 1


# ---------------------------------------------------------------------------
# 6. percentile_rank
# ---------------------------------------------------------------------------

class TestPercentileRank:
    def test_single_observation_returns_one(self):
        state = PaperState()
        state.append_observation(make_obs(1, atm_iv=0.12))
        assert state.percentile_rank() == 1.0

    def test_multiple_observations(self):
        """With linearly increasing IVs, the last should have rank 1.0."""
        state = PaperState(iv_lookback=5)
        for i, iv in enumerate([0.10, 0.11, 0.12, 0.13, 0.14], start=1):
            state.append_observation(make_obs(i, atm_iv=iv))
        # All 5 values <= 0.14, so rank = 5/5 = 1.0
        assert state.percentile_rank() == 1.0

    def test_lowest_value_rank(self):
        """If current IV is the lowest, rank = 1/n."""
        state = PaperState(iv_lookback=5)
        # Add decreasing IVs so last one is smallest
        for i, iv in enumerate([0.18, 0.16, 0.14, 0.12, 0.10], start=1):
            state.append_observation(make_obs(i, atm_iv=iv))
        # Only 0.10 <= 0.10, so rank = 1/5 = 0.2
        assert math.isclose(state.percentile_rank(), 0.2)

    def test_manual_computation(self):
        """Hand-verified percentile rank computation."""
        state = PaperState(iv_lookback=4)
        ivs = [0.10, 0.15, 0.12, 0.14]
        for i, iv in enumerate(ivs, start=1):
            state.append_observation(make_obs(i, atm_iv=iv))
        # current = 0.14
        # lookback (last 4): [0.10, 0.15, 0.12, 0.14]
        # values <= 0.14: 0.10, 0.12, 0.14 => 3 out of 4
        assert math.isclose(state.percentile_rank(), 3 / 4)

    def test_lookback_window_truncation(self):
        """With lookback=3, only last 3 values should be used."""
        state = PaperState(iv_lookback=3)
        ivs = [0.20, 0.10, 0.12, 0.14, 0.11]
        for i, iv in enumerate(ivs, start=1):
            state.append_observation(make_obs(i, atm_iv=iv))
        # window is last 3: [0.12, 0.14, 0.11]
        # current = 0.11, values <= 0.11: just 0.11 => 1/3
        assert math.isclose(state.percentile_rank(), 1 / 3)

    def test_empty_history_returns_zero(self):
        state = PaperState()
        assert state.percentile_rank() == 0.0

    def test_custom_window_parameter(self):
        state = PaperState(iv_lookback=100)
        ivs = [0.10, 0.12, 0.14, 0.11, 0.13]
        for i, iv in enumerate(ivs, start=1):
            state.append_observation(make_obs(i, atm_iv=iv))
        # Override with window=3: last 3 = [0.14, 0.11, 0.13]
        # current = 0.13, values <= 0.13: 0.11, 0.13 => 2/3
        assert math.isclose(state.percentile_rank(window=3), 2 / 3)


# ---------------------------------------------------------------------------
# 7. have_enough_history
# ---------------------------------------------------------------------------

class TestHaveEnoughHistory:
    def test_false_when_too_few(self):
        state = PaperState(iv_lookback=30)
        for obs in build_history(29):
            state.append_observation(obs)
        assert not state.have_enough_history()

    def test_true_when_exactly_enough(self):
        state = PaperState(iv_lookback=30)
        for obs in build_history(30):
            state.append_observation(obs)
        assert state.have_enough_history()

    def test_true_when_more_than_enough(self):
        state = state_with_history(50, iv_lookback=30)
        assert state.have_enough_history()

    def test_empty_history(self):
        state = PaperState(iv_lookback=5)
        assert not state.have_enough_history()


# ---------------------------------------------------------------------------
# 8. enter_position
# ---------------------------------------------------------------------------

class TestEnterPosition:
    def test_creates_position(self):
        state = PaperState()
        obs = make_obs(10, spot=24500.0, atm_iv=0.18)
        pos = state.enter_position(obs, pctile=0.85)
        assert pos is state.position
        assert state.position is not None

    def test_fields_set_correctly(self):
        state = PaperState()
        obs = make_obs(10, spot=24500.0, atm_iv=0.18)
        pos = state.enter_position(obs, pctile=0.85)
        assert pos.entry_date == "2026-01-10"
        assert pos.entry_spot == 24500.0
        assert pos.entry_iv == 0.18
        assert pos.iv_pctile == 0.85
        assert pos.hold_days == 0

    def test_raises_if_already_in_position(self):
        state = PaperState()
        obs1 = make_obs(10, spot=24500.0)
        state.enter_position(obs1, pctile=0.85)
        obs2 = make_obs(11, spot=24600.0)
        with pytest.raises(ValueError, match="Already in a position"):
            state.enter_position(obs2, pctile=0.90)


# ---------------------------------------------------------------------------
# 9. exit_position
# ---------------------------------------------------------------------------

class TestExitPosition:
    def test_pnl_computation_with_costs(self):
        state = PaperState(cost_bps=10.0)
        entry_obs = make_obs(10, spot=24000.0, atm_iv=0.18)
        state.enter_position(entry_obs, pctile=0.85)

        exit_obs = make_obs(15, spot=24240.0, atm_iv=0.12)
        trade = state.exit_position(exit_obs, reason="iv_normalised")

        # Expected PnL: (24240 - 24000) / 24000 - 10/10000
        # = 240/24000 - 0.001 = 0.01 - 0.001 = 0.009
        expected_pnl = (24240.0 - 24000.0) / 24000.0 - 10.0 / 10_000
        assert math.isclose(trade.pnl_pct, expected_pnl, rel_tol=1e-9)

    def test_records_trade(self):
        state = PaperState(cost_bps=5.0)
        state.enter_position(make_obs(10, spot=24000.0, atm_iv=0.18), 0.85)
        state.exit_position(make_obs(15, spot=24100.0, atm_iv=0.12), "max_hold")
        assert len(state.closed_trades) == 1
        trade = state.closed_trades[0]
        assert trade.entry_date == "2026-01-10"
        assert trade.exit_date == "2026-01-15"
        assert trade.entry_spot == 24000.0
        assert trade.exit_spot == 24100.0
        assert trade.entry_iv == 0.18
        assert trade.exit_iv == 0.12
        assert trade.exit_reason == "max_hold"

    def test_clears_position(self):
        state = PaperState()
        state.enter_position(make_obs(10, spot=24000.0), 0.85)
        state.exit_position(make_obs(15, spot=24100.0), "iv_normalised")
        assert state.position is None

    def test_updates_equity(self):
        state = PaperState(cost_bps=0.0)
        state.enter_position(make_obs(10, spot=24000.0), 0.85)
        state.exit_position(make_obs(15, spot=24480.0), "max_hold")
        # PnL = 480/24000 = 0.02, equity = 1.0 * (1 + 0.02) = 1.02
        assert math.isclose(state.equity, 1.02)

    def test_negative_pnl(self):
        state = PaperState(cost_bps=5.0)
        state.enter_position(make_obs(10, spot=24000.0), 0.85)
        trade = state.exit_position(make_obs(15, spot=23900.0), "max_hold")
        # PnL = (23900-24000)/24000 - 5/10000 = -100/24000 - 0.0005
        expected = -100.0 / 24000.0 - 0.0005
        assert math.isclose(trade.pnl_pct, expected, rel_tol=1e-9)
        assert state.equity < 1.0

    def test_raises_if_no_position(self):
        state = PaperState()
        with pytest.raises(ValueError, match="No position to exit"):
            state.exit_position(make_obs(15, spot=24100.0), "test")


# ---------------------------------------------------------------------------
# 10. increment_hold
# ---------------------------------------------------------------------------

class TestIncrementHold:
    def test_increments_hold_days(self):
        state = PaperState()
        state.enter_position(make_obs(10, spot=24000.0), 0.85)
        assert state.position.hold_days == 0
        state.increment_hold()
        assert state.position.hold_days == 1
        state.increment_hold()
        assert state.position.hold_days == 2

    def test_noop_if_no_position(self):
        state = PaperState()
        # Should not raise
        state.increment_hold()
        assert state.position is None


# ---------------------------------------------------------------------------
# 11. check_signal
# ---------------------------------------------------------------------------

class TestCheckSignal:
    def test_warmup(self):
        state = PaperState(iv_lookback=30)
        for obs in build_history(20):
            state.append_observation(obs)
        assert state.check_signal() == "warmup"

    def test_wait_when_iv_below_threshold(self):
        """All identical IVs => percentile = 1.0 for the last value.
        We need percentile < entry_pctile to get 'wait'.
        Use decreasing IVs so last is the smallest.
        """
        state = PaperState(iv_lookback=5, entry_pctile=0.80)
        # 5 decreasing IVs: 0.20, 0.18, 0.16, 0.14, 0.12
        ivs = [0.20, 0.18, 0.16, 0.14, 0.12]
        for i, iv in enumerate(ivs, start=1):
            state.append_observation(make_obs(i, atm_iv=iv))
        # rank = 1/5 = 0.20 < 0.80 => wait
        assert state.check_signal() == "wait"

    def test_enter_when_iv_above_threshold(self):
        state = PaperState(iv_lookback=5, entry_pctile=0.80)
        # Increasing IVs: last one has rank 5/5 = 1.0 >= 0.80
        ivs = [0.10, 0.11, 0.12, 0.13, 0.14]
        for i, iv in enumerate(ivs, start=1):
            state.append_observation(make_obs(i, atm_iv=iv))
        assert state.check_signal() == "enter"

    def test_hold(self):
        state = PaperState(iv_lookback=5, entry_pctile=0.80, exit_pctile=0.50, max_hold_days=10)
        ivs = [0.10, 0.11, 0.12, 0.13, 0.14]
        for i, iv in enumerate(ivs, start=1):
            state.append_observation(make_obs(i, atm_iv=iv))
        # Enter position
        state.enter_position(state.iv_history[-1], 1.0)
        state.increment_hold()
        # Percentile is 1.0 >= exit_pctile(0.50), hold_days(1) < max_hold(10)
        assert state.check_signal() == "hold"

    def test_exit_hold(self):
        state = PaperState(iv_lookback=5, max_hold_days=3)
        for obs in build_history(5):
            state.append_observation(obs)
        state.enter_position(state.iv_history[-1], 0.90)
        # Increment hold to reach max
        for _ in range(3):
            state.increment_hold()
        assert state.position.hold_days == 3
        assert state.check_signal() == "exit_hold"

    def test_exit_iv(self):
        state = PaperState(iv_lookback=5, exit_pctile=0.50, max_hold_days=10)
        # Build history with high IVs, then add a very low IV at the end
        ivs = [0.20, 0.18, 0.16, 0.14, 0.12]
        for i, iv in enumerate(ivs, start=1):
            state.append_observation(make_obs(i, atm_iv=iv))
        # Enter at day 3 (some position, doesn't matter for signal)
        state.enter_position(state.iv_history[2], 0.85)
        state.increment_hold()
        # Now add a very low IV observation that drops percentile below exit
        state.append_observation(make_obs(6, atm_iv=0.08))
        # lookback = last 5: [0.18, 0.16, 0.14, 0.12, 0.08]
        # current = 0.08, values <= 0.08: [0.08] => 1/5 = 0.2 < 0.50
        assert state.check_signal() == "exit_iv"


# ---------------------------------------------------------------------------
# 12. Performance metrics
# ---------------------------------------------------------------------------

class TestPerformanceMetrics:
    def _build_state_with_trades(self) -> PaperState:
        """Build a state with 4 trades: 3 wins, 1 loss."""
        state = PaperState(cost_bps=0.0)
        trades = [
            ClosedTrade("NIFTY", "2026-01-01", "2026-01-05", 24000, 24240, 0.18, 0.12, 0.85, 0.01, 4, "iv"),
            ClosedTrade("NIFTY", "2026-01-10", "2026-01-14", 24200, 24400, 0.17, 0.11, 0.82, 0.00826, 4, "iv"),
            ClosedTrade("NIFTY", "2026-01-20", "2026-01-24", 24500, 24255, 0.19, 0.14, 0.90, -0.01, 4, "hold"),
            ClosedTrade("NIFTY", "2026-02-01", "2026-02-05", 24300, 24543, 0.16, 0.10, 0.88, 0.01, 4, "iv"),
        ]
        state.closed_trades = trades
        # Compute equity from trades
        equity = 1.0
        for t in trades:
            equity *= (1 + t.pnl_pct)
        state.equity = equity
        return state

    def test_win_rate(self):
        state = self._build_state_with_trades()
        # 3 wins out of 4
        assert math.isclose(state.win_rate(), 0.75)

    def test_win_rate_empty(self):
        state = PaperState()
        assert state.win_rate() == 0.0

    def test_total_return_pct(self):
        state = self._build_state_with_trades()
        expected = (state.equity - 1.0) * 100
        assert math.isclose(state.total_return_pct(), expected)

    def test_total_return_pct_fresh(self):
        state = PaperState()
        assert state.total_return_pct() == 0.0

    def test_avg_pnl_pct(self):
        state = self._build_state_with_trades()
        pnls = [t.pnl_pct for t in state.closed_trades]
        expected = sum(pnls) / len(pnls) * 100
        assert math.isclose(state.avg_pnl_pct(), expected)

    def test_avg_pnl_pct_empty(self):
        state = PaperState()
        assert state.avg_pnl_pct() == 0.0

    def test_single_winning_trade(self):
        state = PaperState()
        state.closed_trades = [
            ClosedTrade("NIFTY", "2026-01-01", "2026-01-05", 24000, 24240, 0.18, 0.12, 0.85, 0.01, 4, "iv"),
        ]
        state.equity = 1.01
        assert state.win_rate() == 1.0
        assert math.isclose(state.total_return_pct(), 1.0)
        assert math.isclose(state.avg_pnl_pct(), 1.0)

    def test_all_losing_trades(self):
        state = PaperState()
        state.closed_trades = [
            ClosedTrade("NIFTY", "2026-01-01", "2026-01-05", 24000, 23800, 0.18, 0.12, 0.85, -0.0083, 4, "hold"),
            ClosedTrade("NIFTY", "2026-01-10", "2026-01-15", 24100, 23900, 0.17, 0.11, 0.82, -0.0083, 4, "hold"),
        ]
        assert state.win_rate() == 0.0


# ---------------------------------------------------------------------------
# 13. Config persistence
# ---------------------------------------------------------------------------

class TestConfigPersistence:
    def test_config_saved_and_restored(self, tmp_path: Path):
        path = tmp_path / "config_state.json"
        state = PaperState(
            iv_lookback=20,
            entry_pctile=0.90,
            exit_pctile=0.40,
            max_hold_days=10,
            cost_bps=12.0,
        )
        state.save(path)
        loaded = PaperState.load(path)
        assert loaded.iv_lookback == 20
        assert loaded.entry_pctile == 0.90
        assert loaded.exit_pctile == 0.40
        assert loaded.max_hold_days == 10
        assert loaded.cost_bps == 12.0

    def test_default_config_when_missing(self):
        """from_dict should fill in defaults when config block is absent."""
        d = {
            "iv_history": [],
            "position": None,
            "closed_trades": [],
            "equity": 1.0,
            "started_at": "",
            "last_scan_date": "",
            # No "config" key
        }
        state = PaperState.from_dict(d)
        assert state.iv_lookback == 30
        assert state.entry_pctile == 0.80
        assert state.exit_pctile == 0.50
        assert state.max_hold_days == 5
        assert state.cost_bps == 5.0

    def test_config_in_json_structure(self, tmp_path: Path):
        path = tmp_path / "state.json"
        state = PaperState(iv_lookback=25, cost_bps=7.5)
        state.save(path)
        raw = json.loads(path.read_text())
        assert "config" in raw
        assert raw["config"]["iv_lookback"] == 25
        assert raw["config"]["cost_bps"] == 7.5


# ---------------------------------------------------------------------------
# 14. Full paper trading cycle
# ---------------------------------------------------------------------------

class TestFullPaperTradingCycle:
    """Simulate 40 observations with an IV spike to trigger entry and exit."""

    def _build_iv_series(self) -> list[float]:
        """Create 40 IV values:
        - Days 1-30: stable around 0.12 (warmup period)
        - Day 31-33: spike up to 0.22 (should trigger entry)
        - Day 34-37: fall back towards 0.12 (should trigger exit)
        - Day 38-40: stable at 0.11
        """
        ivs = []
        # Warmup: 30 days of gentle variation around 0.12
        for i in range(30):
            iv = 0.12 + 0.002 * math.sin(i * 0.3)
            ivs.append(round(iv, 6))
        # Spike: days 31-33
        ivs.append(0.18)   # day 31
        ivs.append(0.20)   # day 32
        ivs.append(0.22)   # day 33 -- peak
        # Decline: days 34-37
        ivs.append(0.16)   # day 34
        ivs.append(0.13)   # day 35
        ivs.append(0.11)   # day 36
        ivs.append(0.10)   # day 37
        # Stable: days 38-40
        ivs.append(0.11)
        ivs.append(0.11)
        ivs.append(0.11)
        return ivs

    def _build_spot_series(self) -> list[float]:
        """Create 40 spot prices. Start at 24000, small daily moves."""
        spots = []
        spot = 24000.0
        for i in range(40):
            spot += 10.0 * math.sin(i * 0.5)
            spots.append(round(spot, 2))
        return spots

    def test_full_cycle(self):
        ivs = self._build_iv_series()
        spots = self._build_spot_series()

        state = PaperState(
            iv_lookback=30,
            entry_pctile=0.80,
            exit_pctile=0.50,
            max_hold_days=5,
            cost_bps=5.0,
        )

        entered = False
        exited = False
        entry_day = None
        exit_day = None

        for day_idx in range(40):
            day_num = day_idx + 1
            # Use February dates to avoid overlap with helper make_obs
            date_str = f"2026-02-{day_num:02d}" if day_num <= 28 else f"2026-03-{day_num - 28:02d}"
            obs = IVObservation(
                date=date_str,
                spot=spots[day_idx],
                atm_iv=ivs[day_idx],
                atm_var=ivs[day_idx] ** 2,
                forward=spots[day_idx] * 1.001,
                sanos_ok=True,
            )
            state.append_observation(obs)

            if state.position is not None:
                state.increment_hold()

            sig = state.check_signal()

            if sig == "enter":
                pctile = state.percentile_rank()
                state.enter_position(obs, pctile)
                entered = True
                entry_day = day_num

            elif sig in ("exit_hold", "exit_iv"):
                reason = "max_hold" if sig == "exit_hold" else "iv_normalised"
                state.exit_position(obs, reason)
                exited = True
                exit_day = day_num

        # Verify that we entered and exited at least once
        assert entered, "Expected at least one entry signal"
        assert exited, "Expected at least one exit signal"

        # Verify the entry happened during the spike (days 31-33)
        assert entry_day is not None
        assert 31 <= entry_day <= 35, f"Entry on day {entry_day}, expected during IV spike"

        # Verify trade was recorded
        assert len(state.closed_trades) >= 1

        # Verify the first trade has correct structure
        trade = state.closed_trades[0]
        assert trade.entry_spot > 0
        assert trade.exit_spot > 0
        assert trade.hold_days >= 0
        assert trade.exit_reason in ("max_hold", "iv_normalised")

        # Verify equity was updated
        assert state.equity != 1.0

        # Verify position is cleared after exit
        # (there might be a second entry after exit; check there was at least one closed trade)
        assert len(state.closed_trades) >= 1

    def test_full_cycle_max_hold_exit(self):
        """Simulate a scenario where IV stays high and exit is via max_hold."""
        state = PaperState(
            iv_lookback=5,
            entry_pctile=0.80,
            exit_pctile=0.50,
            max_hold_days=3,
            cost_bps=5.0,
        )

        # 5 warmup days with low IV
        for i in range(1, 6):
            state.append_observation(make_obs(i, atm_iv=0.10))

        # Day 6: spike that triggers entry (rank will be 1.0 >= 0.80)
        state.append_observation(make_obs(6, atm_iv=0.20, spot=24000.0))
        assert state.check_signal() == "enter"
        pctile = state.percentile_rank()
        state.enter_position(state.iv_history[-1], pctile)

        # Days 7-9: IV stays high, hold_days increments until max_hold
        for day in range(7, 10):
            state.append_observation(make_obs(day, atm_iv=0.20 + 0.01 * (day - 6), spot=24100.0))
            state.increment_hold()
            sig = state.check_signal()
            if sig in ("exit_hold", "exit_iv"):
                state.exit_position(state.iv_history[-1], "max_hold")
                break

        assert len(state.closed_trades) == 1
        assert state.closed_trades[0].exit_reason == "max_hold"
        assert state.closed_trades[0].hold_days == 3
        assert state.position is None

    def test_full_cycle_iv_normalised_exit(self):
        """Simulate a scenario where IV drops and exit is via iv_normalised."""
        state = PaperState(
            iv_lookback=5,
            entry_pctile=0.80,
            exit_pctile=0.50,
            max_hold_days=10,
            cost_bps=5.0,
        )

        # 5 warmup days with moderate IV
        for i in range(1, 6):
            state.append_observation(make_obs(i, atm_iv=0.12))

        # Day 6: spike to trigger entry
        state.append_observation(make_obs(6, atm_iv=0.20, spot=24000.0))
        sig = state.check_signal()
        assert sig == "enter"
        pctile = state.percentile_rank()
        state.enter_position(state.iv_history[-1], pctile)

        # Day 7: IV still high, hold
        state.append_observation(make_obs(7, atm_iv=0.18, spot=24050.0))
        state.increment_hold()
        sig = state.check_signal()
        assert sig == "hold"

        # Day 8: IV drops sharply — should trigger exit_iv
        state.append_observation(make_obs(8, atm_iv=0.08, spot=24100.0))
        state.increment_hold()
        sig = state.check_signal()
        assert sig == "exit_iv"
        state.exit_position(state.iv_history[-1], "iv_normalised")

        assert len(state.closed_trades) == 1
        trade = state.closed_trades[0]
        assert trade.exit_reason == "iv_normalised"
        assert trade.hold_days == 2
        assert trade.entry_spot == 24000.0
        assert trade.exit_spot == 24100.0
        # PnL = (24100-24000)/24000 - 5/10000 = 100/24000 - 0.0005
        expected_pnl = 100.0 / 24000.0 - 0.0005
        assert math.isclose(trade.pnl_pct, expected_pnl, rel_tol=1e-9)

    def test_pnl_matches_equity(self):
        """Verify that equity equals product of (1 + pnl) for all trades."""
        state = PaperState(
            iv_lookback=5,
            entry_pctile=0.80,
            exit_pctile=0.40,
            max_hold_days=3,
            cost_bps=5.0,
        )

        # Run two complete trade cycles
        # Cycle 1: warmup + spike + exit via max_hold
        for i in range(1, 6):
            state.append_observation(make_obs(i, atm_iv=0.10))

        state.append_observation(make_obs(6, atm_iv=0.20, spot=24000.0))
        state.enter_position(state.iv_history[-1], state.percentile_rank())

        for day in range(7, 10):
            state.append_observation(make_obs(day, atm_iv=0.19, spot=24000.0 + 50.0 * (day - 6)))
            state.increment_hold()
            sig = state.check_signal()
            if sig in ("exit_hold", "exit_iv"):
                state.exit_position(state.iv_history[-1], "max_hold")
                break

        # Cycle 2: another spike and exit
        state.append_observation(make_obs(10, atm_iv=0.10, spot=24200.0))
        state.append_observation(make_obs(11, atm_iv=0.22, spot=24200.0))
        if state.check_signal() == "enter":
            state.enter_position(state.iv_history[-1], state.percentile_rank())

        for day in range(12, 16):
            state.append_observation(make_obs(day, atm_iv=0.22, spot=24200.0 + 30.0 * (day - 11)))
            state.increment_hold()
            sig = state.check_signal()
            if sig in ("exit_hold", "exit_iv"):
                state.exit_position(state.iv_history[-1], "max_hold")
                break

        # Verify equity matches
        expected_equity = 1.0
        for t in state.closed_trades:
            expected_equity *= (1 + t.pnl_pct)
        assert math.isclose(state.equity, expected_equity, rel_tol=1e-12)
