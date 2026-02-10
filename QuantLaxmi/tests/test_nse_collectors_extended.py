"""Tests for new NSE collectors (pre-open, OI spurts) + feature groups 21-24.

Phase A: Parser tests for nse_convert.py (pre-open, OI spurts)
Phase B: Feature group tests for mega.py (contract delta, delta-eq OI, pre-open OFI, OI spurts)
Phase C: Integration tests (builder group count, strategy wiring)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Phase A: Parser tests
# ---------------------------------------------------------------------------


def _make_preopen_json() -> dict:
    """Create realistic pre-open FnO JSON."""
    return {
        "data": [
            {
                "metadata": {
                    "symbol": "RELIANCE",
                    "iep": "2450.50",
                    "finalQuantity": "123456",
                    "finalPrice": "302500000",
                },
                "detail": {
                    "preOpenMarket": {
                        "preopen": [
                            {"price": "2448.00", "buyQty": "500", "sellQty": "0"},
                            {"price": "2449.00", "buyQty": "300", "sellQty": "0"},
                            {"price": "2450.00", "buyQty": "200", "sellQty": "0"},
                            {"price": "2451.00", "buyQty": "0", "sellQty": "400"},
                            {"price": "2452.00", "buyQty": "0", "sellQty": "250"},
                            {"price": "2453.00", "buyQty": "0", "sellQty": "100"},
                        ]
                    }
                },
            },
            {
                "metadata": {
                    "symbol": "TCS",
                    "iep": "3800.00",
                    "finalQuantity": "50000",
                    "finalPrice": "190000000",
                },
                "detail": {
                    "preOpenMarket": {
                        "preopen": [
                            {"price": "3798.00", "buyQty": "100", "sellQty": "0"},
                            {"price": "3799.00", "buyQty": "200", "sellQty": "0"},
                            {"price": "3801.00", "buyQty": "0", "sellQty": "150"},
                            {"price": "3802.00", "buyQty": "0", "sellQty": "50"},
                        ]
                    }
                },
            },
        ]
    }


def _make_oi_spurts_json() -> dict:
    """Create realistic OI spurts JSON with 4 categories."""
    return {
        "long_build_up": {
            "data": [
                {
                    "symbol": "NIFTY",
                    "instrument": "OPTIDX",
                    "expiryDate": "27-Feb-2026",
                    "optionType": "CE",
                    "strikePrice": "23000",
                    "ltp": "150.5",
                    "pChange": "12.5",
                    "latestOI": "500000",
                    "previousOI": "400000",
                    "changeInOI": "100000",
                    "volume": "250000",
                    "underlyingValue": "22850",
                },
            ]
        },
        "short_build_up": {
            "data": [
                {
                    "symbol": "BANKNIFTY",
                    "instrument": "OPTIDX",
                    "expiryDate": "27-Feb-2026",
                    "optionType": "PE",
                    "strikePrice": "48000",
                    "ltp": "200.0",
                    "pChange": "-8.3",
                    "latestOI": "300000",
                    "previousOI": "250000",
                    "changeInOI": "50000",
                    "volume": "100000",
                    "underlyingValue": "48200",
                },
            ]
        },
        "long_unwinding": {
            "data": [
                {
                    "symbol": "RELIANCE",
                    "instrument": "OPTSTK",
                    "expiryDate": "27-Feb-2026",
                    "optionType": "CE",
                    "strikePrice": "2500",
                    "ltp": "30.0",
                    "pChange": "-5.0",
                    "latestOI": "80000",
                    "previousOI": "100000",
                    "changeInOI": "-20000",
                    "volume": "50000",
                    "underlyingValue": "2450",
                },
            ]
        },
        "short_covering": {
            "data": [
                {
                    "symbol": "TCS",
                    "instrument": "OPTSTK",
                    "expiryDate": "27-Feb-2026",
                    "optionType": "PE",
                    "strikePrice": "3700",
                    "ltp": "15.0",
                    "pChange": "20.0",
                    "latestOI": "60000",
                    "previousOI": "75000",
                    "changeInOI": "-15000",
                    "volume": "30000",
                    "underlyingValue": "3800",
                },
            ]
        },
    }


class TestParsePreopenJson:
    """Tests for parse_preopen_json in nse_convert.py."""

    def test_basic_parse(self, tmp_path: Path):
        from quantlaxmi.data.nse_convert import parse_preopen_json

        path = tmp_path / "pre_open_fo.json"
        path.write_text(json.dumps(_make_preopen_json()))

        df = parse_preopen_json(path)
        assert df is not None
        assert len(df) == 2
        assert "symbol" in df.columns
        assert "iep" in df.columns
        assert "ofi" in df.columns

    def test_symbols_present(self, tmp_path: Path):
        from quantlaxmi.data.nse_convert import parse_preopen_json

        path = tmp_path / "pre_open_fo.json"
        path.write_text(json.dumps(_make_preopen_json()))

        df = parse_preopen_json(path)
        assert set(df["symbol"]) == {"RELIANCE", "TCS"}

    def test_ofi_calculation(self, tmp_path: Path):
        """OFI = sum(bid_qty) - sum(ask_qty)."""
        from quantlaxmi.data.nse_convert import parse_preopen_json

        path = tmp_path / "pre_open_fo.json"
        path.write_text(json.dumps(_make_preopen_json()))

        df = parse_preopen_json(path)
        # RELIANCE: bids=500+300+200=1000, asks=400+250+100=750 → OFI=250
        rel = df[df["symbol"] == "RELIANCE"].iloc[0]
        assert rel["ofi"] == pytest.approx(250.0)

        # TCS: bids=100+200=300, asks=150+50=200 → OFI=100
        tcs = df[df["symbol"] == "TCS"].iloc[0]
        assert tcs["ofi"] == pytest.approx(100.0)

    def test_bid_ask_levels(self, tmp_path: Path):
        from quantlaxmi.data.nse_convert import parse_preopen_json

        path = tmp_path / "pre_open_fo.json"
        path.write_text(json.dumps(_make_preopen_json()))

        df = parse_preopen_json(path)
        rel = df[df["symbol"] == "RELIANCE"].iloc[0]
        # Bids sorted descending by price
        assert rel["bid1_price"] == pytest.approx(2450.0)
        assert rel["bid1_qty"] == pytest.approx(200.0)
        # Asks sorted ascending by price
        assert rel["ask1_price"] == pytest.approx(2451.0)
        assert rel["ask1_qty"] == pytest.approx(400.0)

    def test_empty_json(self, tmp_path: Path):
        from quantlaxmi.data.nse_convert import parse_preopen_json

        path = tmp_path / "pre_open_fo.json"
        path.write_text(json.dumps({"data": []}))

        df = parse_preopen_json(path)
        assert df is None

    def test_schema_columns(self, tmp_path: Path):
        from quantlaxmi.data.nse_convert import parse_preopen_json

        path = tmp_path / "pre_open_fo.json"
        path.write_text(json.dumps(_make_preopen_json()))

        df = parse_preopen_json(path)
        expected_cols = {
            "symbol", "iep", "final_quantity", "total_turnover", "ofi",
            "bid1_price", "bid1_qty", "bid2_price", "bid2_qty",
            "bid3_price", "bid3_qty", "bid4_price", "bid4_qty",
            "bid5_price", "bid5_qty",
            "ask1_price", "ask1_qty", "ask2_price", "ask2_qty",
            "ask3_price", "ask3_qty", "ask4_price", "ask4_qty",
            "ask5_price", "ask5_qty",
        }
        assert expected_cols.issubset(set(df.columns))


class TestParseOiSpurtsJson:
    """Tests for parse_oi_spurts_json in nse_convert.py."""

    def test_basic_parse(self, tmp_path: Path):
        from quantlaxmi.data.nse_convert import parse_oi_spurts_json

        path = tmp_path / "oi_spurts.json"
        path.write_text(json.dumps(_make_oi_spurts_json()))

        df = parse_oi_spurts_json(path)
        assert df is not None
        assert len(df) == 4  # one per category

    def test_four_categories(self, tmp_path: Path):
        from quantlaxmi.data.nse_convert import parse_oi_spurts_json

        path = tmp_path / "oi_spurts.json"
        path.write_text(json.dumps(_make_oi_spurts_json()))

        df = parse_oi_spurts_json(path)
        cats = set(df["category"])
        expected = {"oi_build_call", "oi_build_put", "oi_unwind_call", "oi_unwind_put"}
        assert cats == expected

    def test_numeric_columns(self, tmp_path: Path):
        from quantlaxmi.data.nse_convert import parse_oi_spurts_json

        path = tmp_path / "oi_spurts.json"
        path.write_text(json.dumps(_make_oi_spurts_json()))

        df = parse_oi_spurts_json(path)
        # All numeric columns should be float
        for col in ("strike_price", "ltp", "pchange", "latest_oi", "prev_oi",
                     "change_in_oi", "volume", "underlying_value"):
            assert df[col].dtype in (np.float64, np.int64), f"{col} is {df[col].dtype}"

    def test_change_in_oi_values(self, tmp_path: Path):
        from quantlaxmi.data.nse_convert import parse_oi_spurts_json

        path = tmp_path / "oi_spurts.json"
        path.write_text(json.dumps(_make_oi_spurts_json()))

        df = parse_oi_spurts_json(path)
        # long_build_up has +100000 change
        build_call = df[df["category"] == "oi_build_call"].iloc[0]
        assert build_call["change_in_oi"] == pytest.approx(100000.0)
        # long_unwinding has -20000 change
        unwind_call = df[df["category"] == "oi_unwind_call"].iloc[0]
        assert unwind_call["change_in_oi"] == pytest.approx(-20000.0)

    def test_empty_json(self, tmp_path: Path):
        from quantlaxmi.data.nse_convert import parse_oi_spurts_json

        path = tmp_path / "oi_spurts.json"
        path.write_text(json.dumps({"data": []}))

        df = parse_oi_spurts_json(path)
        assert df is None

    def test_schema_columns(self, tmp_path: Path):
        from quantlaxmi.data.nse_convert import parse_oi_spurts_json

        path = tmp_path / "oi_spurts.json"
        path.write_text(json.dumps(_make_oi_spurts_json()))

        df = parse_oi_spurts_json(path)
        expected_cols = {
            "category", "symbol", "instrument", "expiry_date", "option_type",
            "strike_price", "ltp", "pchange", "latest_oi", "prev_oi",
            "change_in_oi", "volume", "underlying_value",
        }
        assert expected_cols.issubset(set(df.columns))


# ---------------------------------------------------------------------------
# Phase B: Feature group tests
# ---------------------------------------------------------------------------


def _mock_store_sql(category_data: dict):
    """Create a mock store whose sql() returns dataframes based on table name."""
    def sql_fn(query, params=None):
        for key, df in category_data.items():
            if key in query:
                return df.copy()
        return pd.DataFrame()
    store = MagicMock()
    store.sql = MagicMock(side_effect=sql_fn)
    return store


class TestContractDeltaFeatures:
    """Tests for _build_contract_delta_features (group 21)."""

    def _make_contract_delta_df(self) -> pd.DataFrame:
        """Create mock nse_contract_delta data (actual schema)."""
        dates = pd.date_range("2025-12-01", "2025-12-21", freq="B")
        rows = []
        for d in dates:
            for i in range(20):
                opt_type = "CE" if i % 2 == 0 else "PE"
                delta = 0.5 + (i - 10) * 0.03 if opt_type == "CE" else -(0.5 + (i - 10) * 0.03)
                rows.append({
                    "date": d,
                    "symbol": "NIFTY",
                    "delta_factor": delta,
                    "option_type": opt_type,
                    "strike_price": float(23000 + i * 50),
                })
        return pd.DataFrame(rows)

    def test_basic_output(self):
        from quantlaxmi.features.mega import MegaFeatureBuilder

        builder = MegaFeatureBuilder()
        builder._store = _mock_store_sql({"nse_contract_delta": self._make_contract_delta_df()})

        result = builder._build_contract_delta_features("NIFTY", "2025-12-01", "2025-12-21")
        assert not result.empty
        assert "cd_mean_delta" in result.columns
        assert "cd_delta_z21" in result.columns

    def test_feature_count(self):
        from quantlaxmi.features.mega import MegaFeatureBuilder

        builder = MegaFeatureBuilder()
        builder._store = _mock_store_sql({"nse_contract_delta": self._make_contract_delta_df()})

        result = builder._build_contract_delta_features("NIFTY", "2025-12-01", "2025-12-21")
        expected_features = {
            "cd_mean_delta", "cd_mean_delta_chg", "cd_put_call_delta_ratio",
            "cd_atm_delta_near", "cd_delta_skew",
            "cd_n_contracts", "cd_delta_dispersion", "cd_delta_z21",
        }
        assert expected_features.issubset(set(result.columns))

    def test_no_look_ahead(self):
        """Each row should only use data from that date (no future dates)."""
        from quantlaxmi.features.mega import MegaFeatureBuilder

        builder = MegaFeatureBuilder()
        builder._store = _mock_store_sql({"nse_contract_delta": self._make_contract_delta_df()})

        result = builder._build_contract_delta_features("NIFTY", "2025-12-01", "2025-12-21")
        # z21 requires 21 days — first 20 should be NaN
        assert result.index.is_monotonic_increasing


class TestDeltaeqOiFeatures:
    """Tests for _build_deltaeq_oi_features (group 22)."""

    def _make_deltaeq_df(self) -> pd.DataFrame:
        dates = pd.date_range("2025-11-01", "2025-12-21", freq="B")
        rows = []
        for d in dates:
            for sym in ["RELIANCE", "TCS", "INFY", "HDFC", "ICICI", "SBI"]:
                rows.append({
                    "date": d,
                    "symbol": sym,
                    "oi": float(np.random.uniform(1e4, 1e6)),
                    "deltaeq_cw": float(np.random.uniform(-1e8, 1e8)),
                })
        return pd.DataFrame(rows)

    def test_basic_output(self):
        from quantlaxmi.features.mega import MegaFeatureBuilder

        builder = MegaFeatureBuilder()
        builder._store = _mock_store_sql({"nse_combined_oi_deleq": self._make_deltaeq_df()})

        result = builder._build_deltaeq_oi_features("2025-11-01", "2025-12-21")
        assert not result.empty

    def test_feature_names(self):
        from quantlaxmi.features.mega import MegaFeatureBuilder

        builder = MegaFeatureBuilder()
        builder._store = _mock_store_sql({"nse_combined_oi_deleq": self._make_deltaeq_df()})

        result = builder._build_deltaeq_oi_features("2025-11-01", "2025-12-21")
        expected = {
            "deleq_total_oi_z21", "deleq_total_deltaeq_z21",
            "deleq_concentration", "deleq_chg_pct", "deleq_breadth",
            "deleq_oi_vs_deltaeq",
        }
        assert expected.issubset(set(result.columns))

    def test_concentration_bounded(self):
        from quantlaxmi.features.mega import MegaFeatureBuilder

        builder = MegaFeatureBuilder()
        builder._store = _mock_store_sql({"nse_combined_oi_deleq": self._make_deltaeq_df()})

        result = builder._build_deltaeq_oi_features("2025-11-01", "2025-12-21")
        conc = result["deleq_concentration"].dropna()
        assert (conc >= 0).all()
        assert (conc <= 1).all()

    def test_breadth_bounded(self):
        from quantlaxmi.features.mega import MegaFeatureBuilder

        builder = MegaFeatureBuilder()
        builder._store = _mock_store_sql({"nse_combined_oi_deleq": self._make_deltaeq_df()})

        result = builder._build_deltaeq_oi_features("2025-11-01", "2025-12-21")
        breadth = result["deleq_breadth"].dropna()
        assert (breadth >= 0).all()
        assert (breadth <= 1).all()


class TestPreopenOfiFeatures:
    """Tests for _build_preopen_ofi_features (group 23)."""

    def _make_preopen_df(self) -> pd.DataFrame:
        dates = pd.date_range("2025-11-01", "2025-12-21", freq="B")
        rows = []
        for d in dates:
            for sym in ["RELIANCE", "TCS", "INFY"]:
                rows.append({
                    "date": d,
                    "symbol": sym,
                    "iep": float(np.random.uniform(100, 5000)),
                    "final_quantity": float(np.random.randint(0, 100000)),
                    "total_turnover": float(np.random.uniform(1e6, 1e9)),
                    "ofi": float(np.random.uniform(-1000, 1000)),
                    "bid1_price": 100.0,
                    "ask1_price": 100.5,
                    "bid1_qty": 500.0, "bid2_qty": 400.0, "bid3_qty": 300.0,
                    "bid4_qty": 200.0, "bid5_qty": 100.0,
                    "ask1_qty": 450.0, "ask2_qty": 350.0, "ask3_qty": 250.0,
                    "ask4_qty": 150.0, "ask5_qty": 50.0,
                })
        return pd.DataFrame(rows)

    def test_basic_output(self):
        from quantlaxmi.features.mega import MegaFeatureBuilder

        builder = MegaFeatureBuilder()
        builder._store = _mock_store_sql({"nse_preopen": self._make_preopen_df()})

        result = builder._build_preopen_ofi_features("2025-11-01", "2025-12-21")
        assert not result.empty

    def test_feature_names(self):
        from quantlaxmi.features.mega import MegaFeatureBuilder

        builder = MegaFeatureBuilder()
        builder._store = _mock_store_sql({"nse_preopen": self._make_preopen_df()})

        result = builder._build_preopen_ofi_features("2025-11-01", "2025-12-21")
        expected = {
            "preopen_ofi_mean", "preopen_ofi_skew", "preopen_spread_mean",
            "preopen_spread_std", "preopen_iep_gap", "preopen_depth_imb",
            "preopen_vol_surprise", "preopen_participation",
        }
        assert expected.issubset(set(result.columns))

    def test_causal_preopen(self):
        """Pre-open data from day D should produce features for day D (same day, causal)."""
        from quantlaxmi.features.mega import MegaFeatureBuilder

        builder = MegaFeatureBuilder()
        builder._store = _mock_store_sql({"nse_preopen": self._make_preopen_df()})

        result = builder._build_preopen_ofi_features("2025-11-01", "2025-12-21")
        assert result.index.is_monotonic_increasing
        # No future dates should appear
        assert result.index.max() <= pd.Timestamp("2025-12-21")


class TestOiSpurtsFeatures:
    """Tests for _build_oi_spurts_features (group 24)."""

    def _make_oi_spurts_df(self) -> pd.DataFrame:
        dates = pd.date_range("2025-12-01", "2025-12-21", freq="B")
        categories = ["oi_build_call", "oi_build_put", "oi_unwind_call", "oi_unwind_put"]
        rows = []
        for d in dates:
            for cat in categories:
                for _ in range(5):
                    rows.append({
                        "date": d,
                        "category": cat,
                        "change_in_oi": float(np.random.uniform(-50000, 50000)),
                        "pchange": float(np.random.uniform(-20, 20)),
                        "latest_oi": float(np.random.uniform(10000, 500000)),
                        "prev_oi": float(np.random.uniform(10000, 500000)),
                    })
        return pd.DataFrame(rows)

    def test_basic_output(self):
        from quantlaxmi.features.mega import MegaFeatureBuilder

        builder = MegaFeatureBuilder()
        builder._store = _mock_store_sql({"nse_oi_spurts": self._make_oi_spurts_df()})

        result = builder._build_oi_spurts_features("2025-12-01", "2025-12-21")
        assert not result.empty

    def test_feature_names(self):
        from quantlaxmi.features.mega import MegaFeatureBuilder

        builder = MegaFeatureBuilder()
        builder._store = _mock_store_sql({"nse_oi_spurts": self._make_oi_spurts_df()})

        result = builder._build_oi_spurts_features("2025-12-01", "2025-12-21")
        expected = {
            "oisp_build_count", "oisp_unwind_count", "oisp_build_unwind_ratio",
            "oisp_net_oi_change", "oisp_call_put_build", "oisp_max_oi_change_pct",
        }
        assert expected.issubset(set(result.columns))

    def test_ratio_bounded(self):
        from quantlaxmi.features.mega import MegaFeatureBuilder

        builder = MegaFeatureBuilder()
        builder._store = _mock_store_sql({"nse_oi_spurts": self._make_oi_spurts_df()})

        result = builder._build_oi_spurts_features("2025-12-01", "2025-12-21")
        ratio = result["oisp_build_unwind_ratio"].dropna()
        assert (ratio >= 0).all()
        assert (ratio <= 1).all()


# ---------------------------------------------------------------------------
# Phase C: Integration / structure tests
# ---------------------------------------------------------------------------


class TestMegaBuilderGroupCount:
    """Verify total builder groups = 25."""

    def test_builders_list_has_25_entries(self):
        """The builders list in build() should have 25 entries."""
        from quantlaxmi.features.mega import MegaFeatureBuilder

        builder = MegaFeatureBuilder()

        # Inspect build() source to count builder entries
        import inspect
        source = inspect.getsource(builder.build)
        # Count lines that match the pattern ("name", self._build_
        count = source.count("self._build_")
        assert count == 25, f"Expected 25 builder groups, found {count}"


class TestNseFileRegistryCount:
    """Verify the NSE file registry includes new entries."""

    def test_all_files_count(self):
        from quantlaxmi.data.collectors.nse_daily.files import ALL_FILES
        # Was 23, now 25 with pre_open_fo.json and oi_spurts.json
        assert len(ALL_FILES) == 25

    def test_preopen_in_files(self):
        from quantlaxmi.data.collectors.nse_daily.files import ALL_FILES
        names = [f.name for f in ALL_FILES]
        assert "pre_open_fo.json" in names

    def test_oi_spurts_in_files(self):
        from quantlaxmi.data.collectors.nse_daily.files import ALL_FILES
        names = [f.name for f in ALL_FILES]
        assert "oi_spurts.json" in names

    def test_preopen_is_optional(self):
        from quantlaxmi.data.collectors.nse_daily.files import ALL_FILES
        pre_open = [f for f in ALL_FILES if f.name == "pre_open_fo.json"][0]
        assert pre_open.optional is True
        assert pre_open.tier == 2

    def test_oi_spurts_is_optional(self):
        from quantlaxmi.data.collectors.nse_daily.files import ALL_FILES
        oi_spurts = [f for f in ALL_FILES if f.name == "oi_spurts.json"][0]
        assert oi_spurts.optional is True
        assert oi_spurts.tier == 2


class TestNseConvertRegistryCount:
    """Verify nse_convert registry includes new parsers."""

    def test_registry_includes_preopen(self):
        from quantlaxmi.data.nse_convert import REGISTRY
        cats = [s.category for s in REGISTRY]
        assert "nse_preopen" in cats

    def test_registry_includes_oi_spurts(self):
        from quantlaxmi.data.nse_convert import REGISTRY
        cats = [s.category for s in REGISTRY]
        assert "nse_oi_spurts" in cats

    def test_parsers_registered(self):
        from quantlaxmi.data.nse_convert import _PARSERS
        assert "preopen_json" in _PARSERS
        assert "oi_spurts_json" in _PARSERS


class TestNseFileUrls:
    """Verify NSE file URL generation for new entries."""

    def test_preopen_url(self):
        from datetime import date
        from quantlaxmi.data.collectors.nse_daily.files import ALL_FILES

        pre_open = [f for f in ALL_FILES if f.name == "pre_open_fo.json"][0]
        url = pre_open.url_for_date(date(2026, 2, 10))
        assert "www.nseindia.com" in url
        assert "market-data-pre-open" in url

    def test_oi_spurts_url(self):
        from datetime import date
        from quantlaxmi.data.collectors.nse_daily.files import ALL_FILES

        oi_spurts = [f for f in ALL_FILES if f.name == "oi_spurts.json"][0]
        url = oi_spurts.url_for_date(date(2026, 2, 10))
        assert "www.nseindia.com" in url
        assert "oi-spurts" in url
