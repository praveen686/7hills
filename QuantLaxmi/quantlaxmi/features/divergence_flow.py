"""Divergence Flow Field (DFF) Feature Builder — S25 Strategy.

Applies conservation-law and Helmholtz decomposition to NSE 4-party
participant OI to extract information flow signals.

The zero-sum constraint Σ_i N_i^k(t) = 0 for each instrument class k
creates a conserved flow field. We decompose daily changes into:
- Divergence (d_hat): net informed accumulation rate
- Rotation (r_hat): instrument-class preference of informed money
- Composite signal: weighted combination with interaction term

Usage
-----
    from quantlaxmi.features.divergence_flow import DivergenceFlowBuilder, DFFConfig

    builder = DivergenceFlowBuilder()
    features = builder.build("2025-08-01", "2026-02-06")
    print(features.shape, features.columns.tolist())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Instrument class keys (order matters for consistent iteration)
_INSTRUMENT_CLASSES: list[str] = [
    "fut_idx",       # Index Futures
    "fut_stk",       # Stock Futures
    "opt_idx_call",  # Index Options Call
    "opt_idx_put",   # Index Options Put
    "opt_stk_call",  # Stock Options Call
    "opt_stk_put",   # Stock Options Put
]

# Mapping from instrument class key to DuckDB column name pairs (Long, Short)
_OI_COLUMNS: dict[str, tuple[str, str]] = {
    "fut_idx":      ("Future Index Long",       "Future Index Short"),
    "fut_stk":      ("Future Stock Long",       "Future Stock Short"),
    "opt_idx_call": ("Option Index Call Long",   "Option Index Call Short"),
    "opt_idx_put":  ("Option Index Put Long",    "Option Index Put Short"),
    "opt_stk_call": ("Option Stock Call Long",   "Option Stock Call Short"),
    "opt_stk_put":  ("Option Stock Put Long",    "Option Stock Put Short"),
}

# Participant types
_PARTICIPANTS: list[str] = ["FII", "DII", "CLIENT", "PRO"]

# Informed vs uninformed grouping
_INFORMED: list[str] = ["FII", "PRO"]
_UNINFORMED: list[str] = ["CLIENT", "DII"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DFFConfig:
    """Configuration for Divergence Flow Field features."""

    # Delta-equivalent weights per instrument class
    w_fut_idx: float = 1.0        # Index futures: full delta
    w_fut_stk: float = 0.5        # Stock futures: half (diversification)
    w_opt_idx_call: float = 0.4   # Index calls: ~ATM delta
    w_opt_idx_put: float = -0.4   # Index puts: negative delta
    w_opt_stk_call: float = 0.2   # Stock calls: smaller
    w_opt_stk_put: float = -0.2   # Stock puts: smaller negative

    # Signal parameters
    zscore_window: int = 21       # Rolling z-score lookback
    ema_span: int = 5             # EMA span for smoothed features
    momentum_lag: int = 5         # Lag for momentum feature

    # Composite signal weights
    alpha: float = 0.6            # Divergence weight
    beta: float = 0.25            # Rotation weight
    gamma: float = 0.15           # Interaction weight

    # Energy floor (prevent division by zero)
    energy_floor: float = 1.0


# ---------------------------------------------------------------------------
# DivergenceFlowBuilder
# ---------------------------------------------------------------------------


class DivergenceFlowBuilder:
    """Build DFF features from NSE participant OI data.

    Implements the Helmholtz decomposition of participant positioning
    flows into divergence (irrotational) and rotation (solenoidal)
    components, plus derived trading signals.

    Parameters
    ----------
    config : DFFConfig, optional
        Feature configuration. Uses defaults if not provided.
    """

    def __init__(self, config: DFFConfig | None = None) -> None:
        self.config: DFFConfig = config or DFFConfig()
        self._weights: dict[str, float] = {
            "fut_idx": self.config.w_fut_idx,
            "fut_stk": self.config.w_fut_stk,
            "opt_idx_call": self.config.w_opt_idx_call,
            "opt_idx_put": self.config.w_opt_idx_put,
            "opt_stk_call": self.config.w_opt_stk_call,
            "opt_stk_put": self.config.w_opt_stk_put,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        start_date: str,
        end_date: str,
        store: "MarketDataStore | None" = None,
    ) -> pd.DataFrame:
        """Build DFF features for a date range.

        Parameters
        ----------
        start_date, end_date : str
            Date range in ``"YYYY-MM-DD"`` format.
        store : MarketDataStore, optional
            If ``None``, creates a new one.

        Returns
        -------
        pd.DataFrame
            Features indexed by datetime, columns prefixed with ``"dff_"``.
        """
        if store is None:
            from quantlaxmi.data.store import MarketDataStore
            store = MarketDataStore()

        # Step 1: Load raw participant OI (with lookback buffer)
        raw = self._load_participant_oi(store, start_date, end_date)
        if raw is None or raw.empty:
            logger.warning("No participant OI data for %s to %s", start_date, end_date)
            return pd.DataFrame()

        return self._build_from_raw(raw, start_date, end_date)

    @classmethod
    def build_from_dataframe(
        cls,
        raw: pd.DataFrame,
        start_date: str,
        end_date: str,
        config: DFFConfig | None = None,
    ) -> pd.DataFrame:
        """Build DFF features from a pre-loaded DataFrame (for testing).

        Parameters
        ----------
        raw : pd.DataFrame
            Raw participant OI data with columns matching ``nse_participant_oi``.
            Must include: ``date``, ``"Client Type"``, and all OI columns.
            Should include lookback buffer days before ``start_date``.
        start_date, end_date : str
            Date range in ``"YYYY-MM-DD"`` format for the output window.
        config : DFFConfig, optional
            Feature configuration.

        Returns
        -------
        pd.DataFrame
            Features indexed by datetime, columns prefixed with ``"dff_"``.
        """
        builder = cls(config=config)
        return builder._build_from_raw(raw, start_date, end_date)

    def verify_conservation(
        self,
        net_positions: pd.DataFrame,
        tolerance: float = 1e-6,
    ) -> pd.DataFrame:
        """Check the zero-sum constraint: Sigma_i N_i^k(t) approx 0.

        The four participants (FII, DII, CLIENT, PRO) should sum to zero
        for each instrument class on each date (since every long has a
        corresponding short across participants).

        Note: In practice, the TOTAL row may differ from the sum of the
        four participants due to rounding or classification differences.
        This method checks the sum of the four named participants only.

        Parameters
        ----------
        net_positions : pd.DataFrame
            Output of ``_compute_net_positions``: MultiIndex columns
            ``(participant, instrument_class)``, indexed by date.
        tolerance : float
            Absolute tolerance for the zero-sum check.

        Returns
        -------
        pd.DataFrame
            One column per instrument class showing the sum across
            participants. Values near zero indicate conservation holds.
        """
        results: dict[str, pd.Series] = {}
        for ic in _INSTRUMENT_CLASSES:
            cols_for_ic = [
                (p, ic) for p in _PARTICIPANTS
                if (p, ic) in net_positions.columns
            ]
            if cols_for_ic:
                total = net_positions[cols_for_ic].sum(axis=1)
                results[ic] = total
            else:
                results[ic] = pd.Series(0.0, index=net_positions.index)

        check_df = pd.DataFrame(results, index=net_positions.index)
        max_violation = check_df.abs().max().max()
        if max_violation > tolerance:
            logger.warning(
                "Conservation violation: max |Sigma_i N_i^k| = %.4f "
                "(tolerance = %.2e)",
                max_violation,
                tolerance,
            )
        else:
            logger.info(
                "Conservation check passed: max violation = %.6f", max_violation
            )
        return check_df

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _build_from_raw(
        self,
        raw: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Run the full pipeline on pre-loaded raw data.

        Parameters
        ----------
        raw : pd.DataFrame
            Raw participant OI (includes lookback buffer).
        start_date, end_date : str
            Output date range.

        Returns
        -------
        pd.DataFrame
            DFF features trimmed to ``[start_date, end_date]``.
        """
        # Step 2: Compute net positioning N_i^k(t)
        net_positions = self._compute_net_positions(raw)
        if net_positions.empty:
            return pd.DataFrame()

        # Step 3: Compute daily flows J_i^k(t) = Delta N_i^k(t)
        flows = self._compute_flows(net_positions)

        # Step 4: Helmholtz decomposition
        decomposition = self._helmholtz_decompose(flows)

        # Step 5: Derive features
        features = self._derive_features(decomposition)

        # Trim to requested date range
        features = features.loc[start_date:end_date]

        logger.info(
            "DivergenceFlowBuilder: %d features, %d rows (%s to %s)",
            len(features.columns),
            len(features),
            features.index.min() if not features.empty else "N/A",
            features.index.max() if not features.empty else "N/A",
        )

        return features

    # ------------------------------------------------------------------
    # Step 1: Load raw participant OI
    # ------------------------------------------------------------------

    def _load_participant_oi(
        self,
        store: "MarketDataStore",
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame | None:
        """Load raw participant OI from DuckDB with lookback buffer.

        Loads extra days before ``start_date`` to allow rolling window
        computations (z-score, momentum, EMA) to warm up.

        Buffer = zscore_window + momentum_lag + 5 extra days (weekends/holidays).
        """
        buffer_trading_days = (
            self.config.zscore_window + self.config.momentum_lag + 5
        )
        # Convert to calendar days (conservative: ~1.5x for weekends/holidays)
        buffer_calendar_days = int(buffer_trading_days * 1.8)
        buffered_start = (
            datetime.strptime(start_date, "%Y-%m-%d")
            - timedelta(days=buffer_calendar_days)
        ).strftime("%Y-%m-%d")

        try:
            raw = store.sql(
                """
                SELECT * FROM nse_participant_oi
                WHERE date BETWEEN ? AND ?
                ORDER BY date
                """,
                [buffered_start, end_date],
            )
        except Exception:
            logger.exception("nse_participant_oi query failed")
            return None

        if raw is None or raw.empty:
            return None

        # Ensure date is datetime
        raw["date"] = pd.to_datetime(raw["date"])

        # Coerce all numeric columns
        numeric_cols = [
            c for c in raw.columns if c not in ("Client Type", "date")
        ]
        for col in numeric_cols:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

        return raw

    # ------------------------------------------------------------------
    # Step 2: Compute net positions
    # ------------------------------------------------------------------

    def _compute_net_positions(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Compute N_i^k(t) = Long_i^k - Short_i^k for all participants x instruments.

        Parameters
        ----------
        raw : pd.DataFrame
            Raw participant OI from DuckDB with columns ``date``,
            ``"Client Type"``, and OI long/short columns.

        Returns
        -------
        pd.DataFrame
            MultiIndex columns ``(participant, instrument_class)``,
            indexed by date. Values are net OI (long - short).
        """
        records: list[dict[tuple[str, str], float]] = []
        dates: list[pd.Timestamp] = []

        for dt, grp in raw.groupby("date"):
            row: dict[tuple[str, str], float] = {}
            for _, r in grp.iterrows():
                participant = str(r["Client Type"]).strip().upper()
                if participant == "TOTAL" or participant not in _PARTICIPANTS:
                    continue

                for ic, (long_col, short_col) in _OI_COLUMNS.items():
                    long_val = r.get(long_col, 0)
                    short_val = r.get(short_col, 0)
                    # Handle NaN/None
                    if pd.isna(long_val):
                        long_val = 0.0
                    if pd.isna(short_val):
                        short_val = 0.0
                    row[(participant, ic)] = float(long_val) - float(short_val)

            if row:
                records.append(row)
                dates.append(pd.Timestamp(dt))

        if not records:
            return pd.DataFrame()

        net_positions = pd.DataFrame(records, index=pd.DatetimeIndex(dates, name="date"))
        net_positions.columns = pd.MultiIndex.from_tuples(
            net_positions.columns, names=["participant", "instrument_class"]
        )
        net_positions = net_positions.sort_index()

        return net_positions

    # ------------------------------------------------------------------
    # Step 3: Compute flows
    # ------------------------------------------------------------------

    def _compute_flows(self, net_positions: pd.DataFrame) -> pd.DataFrame:
        """Compute daily flows J_i^k(t) = N_i^k(t) - N_i^k(t-1).

        Also computes per-participant aggregate flow:
            F_i(t) = Sigma_k w_k * J_i^k(t)

        Parameters
        ----------
        net_positions : pd.DataFrame
            Output of ``_compute_net_positions``.

        Returns
        -------
        pd.DataFrame
            Contains both raw flows (MultiIndex ``(participant, instrument_class)``)
            and aggregate flows (columns ``agg_<participant>``).
            Also stores ``_net_positions`` reference for downstream use.
        """
        # Daily change: J_i^k(t) = N_i^k(t) - N_i^k(t-1)
        raw_flows = net_positions.diff()
        # First row is NaN (no previous day); fill with 0 for flows
        raw_flows = raw_flows.fillna(0.0)

        # Compute aggregate flow per participant: F_i(t) = Sigma_k w_k * J_i^k(t)
        agg_flows: dict[str, pd.Series] = {}
        for participant in _PARTICIPANTS:
            weighted_sum = pd.Series(0.0, index=raw_flows.index)
            for ic in _INSTRUMENT_CLASSES:
                if (participant, ic) in raw_flows.columns:
                    w = self._weights[ic]
                    weighted_sum = weighted_sum + w * raw_flows[(participant, ic)]
            agg_flows[f"agg_{participant}"] = weighted_sum

        # Combine into a single DataFrame:
        # - raw flows with MultiIndex columns for the decomposition
        # - aggregate flows as flat columns (stored separately)
        # We attach aggregate flows and the raw flows to a result object
        result = raw_flows.copy()
        # Store aggregate flows as an attribute for use in decomposition
        result.attrs["agg_flows"] = pd.DataFrame(agg_flows, index=raw_flows.index)

        return result

    # ------------------------------------------------------------------
    # Step 4: Helmholtz decomposition
    # ------------------------------------------------------------------

    def _helmholtz_decompose(self, flows: pd.DataFrame) -> pd.DataFrame:
        """Compute Helmholtz decomposition of participant flows.

        For each instrument class k:
            d^k(t) = [J_FII^k + J_PRO^k] - [J_CLIENT^k + J_DII^k]

        Total divergence (delta-weighted):
            D(t) = Sigma_k w_k * d^k(t)

        Rotation (instrument preference of informed money):
            R(t) = w_IF * d^IF - w_SF * d^SF - (w_IOC * d^IOC + w_IOP * d^IOP)
            where IF = fut_idx, SF = fut_stk, IOC = opt_idx_call, IOP = opt_idx_put

        Energy (total activity):
            E(t) = Sigma_i Sigma_k |J_i^k(t)|

        Normalized:
            d_hat = D / max(E, floor)
            r_hat = R / max(E, floor)

        Returns
        -------
        pd.DataFrame
            Columns: ``d_<ic>`` for each instrument class, ``D``, ``R``, ``E``,
            ``d_hat``, ``r_hat``.
        """
        floor = self.config.energy_floor
        result = pd.DataFrame(index=flows.index)

        # Per-instrument divergence: d^k = (informed flows) - (uninformed flows)
        for ic in _INSTRUMENT_CLASSES:
            informed_flow = pd.Series(0.0, index=flows.index)
            uninformed_flow = pd.Series(0.0, index=flows.index)

            for p in _INFORMED:
                if (p, ic) in flows.columns:
                    informed_flow = informed_flow + flows[(p, ic)]

            for p in _UNINFORMED:
                if (p, ic) in flows.columns:
                    uninformed_flow = uninformed_flow + flows[(p, ic)]

            result[f"d_{ic}"] = informed_flow - uninformed_flow

        # Total divergence: D(t) = Sigma_k w_k * d^k(t)
        D = pd.Series(0.0, index=flows.index)
        for ic in _INSTRUMENT_CLASSES:
            D = D + self._weights[ic] * result[f"d_{ic}"]
        result["D"] = D

        # Rotation: measures whether informed money prefers index futures vs
        # stock derivatives and futures vs options.
        # R(t) = w_IF * d^IF - w_SF * d^SF - (w_IOC * d^IOC + w_IOP * d^IOP)
        R = (
            self._weights["fut_idx"] * result["d_fut_idx"]
            - self._weights["fut_stk"] * result["d_fut_stk"]
            - (
                self._weights["opt_idx_call"] * result["d_opt_idx_call"]
                + self._weights["opt_idx_put"] * result["d_opt_idx_put"]
            )
        )
        result["R"] = R

        # Energy: E(t) = Sigma_i Sigma_k |J_i^k(t)|
        E = pd.Series(0.0, index=flows.index)
        for p in _PARTICIPANTS:
            for ic in _INSTRUMENT_CLASSES:
                if (p, ic) in flows.columns:
                    E = E + flows[(p, ic)].abs()
        result["E"] = E

        # Normalized quantities
        E_clamped = E.clip(lower=floor)
        result["d_hat"] = D / E_clamped
        result["r_hat"] = R / E_clamped

        return result

    # ------------------------------------------------------------------
    # Step 5: Derive features
    # ------------------------------------------------------------------

    def _derive_features(self, decomposition: pd.DataFrame) -> pd.DataFrame:
        """Derive all DFF features from the Helmholtz decomposition.

        Core features (7):
            - ``dff_d_hat``: normalized divergence
            - ``dff_r_hat``: normalized rotation
            - ``dff_z_d``: z-scored divergence
            - ``dff_z_r``: z-scored rotation
            - ``dff_interaction``: Z_d x Z_r
            - ``dff_energy``: total system energy (log-scaled)
            - ``dff_composite``: alpha * Z_d + beta * Z_r + gamma * Z_d * Z_r

        Auxiliary features (5):
            - ``dff_d_hat_5d``: 5-day EMA of d_hat
            - ``dff_r_hat_5d``: 5-day EMA of r_hat
            - ``dff_energy_z``: z-scored log energy
            - ``dff_regime``: regime indicator (0-4 based on sign quadrant)
            - ``dff_momentum``: d_hat(t) - d_hat(t - momentum_lag)

        Parameters
        ----------
        decomposition : pd.DataFrame
            Output of ``_helmholtz_decompose``.

        Returns
        -------
        pd.DataFrame
            12 features, all prefixed with ``"dff_"``.
        """
        cfg = self.config
        features = pd.DataFrame(index=decomposition.index)

        d_hat = decomposition["d_hat"]
        r_hat = decomposition["r_hat"]
        E = decomposition["E"]

        # --- Core features ---

        # 1. Normalized divergence
        features["dff_d_hat"] = d_hat

        # 2. Normalized rotation
        features["dff_r_hat"] = r_hat

        # 3. Z-scored divergence
        z_d = self._rolling_zscore_clipped(d_hat, cfg.zscore_window)
        features["dff_z_d"] = z_d

        # 4. Z-scored rotation
        z_r = self._rolling_zscore_clipped(r_hat, cfg.zscore_window)
        features["dff_z_r"] = z_r

        # 5. Interaction term
        features["dff_interaction"] = z_d * z_r

        # 6. Log-scaled energy
        log_energy = np.log1p(E)
        features["dff_energy"] = log_energy

        # 7. Composite signal
        features["dff_composite"] = (
            cfg.alpha * z_d + cfg.beta * z_r + cfg.gamma * z_d * z_r
        )

        # --- Auxiliary features ---

        # 8. 5-day EMA of d_hat
        features["dff_d_hat_5d"] = d_hat.ewm(
            span=cfg.ema_span, min_periods=cfg.ema_span
        ).mean()

        # 9. 5-day EMA of r_hat
        features["dff_r_hat_5d"] = r_hat.ewm(
            span=cfg.ema_span, min_periods=cfg.ema_span
        ).mean()

        # 10. Z-scored log energy
        features["dff_energy_z"] = self._rolling_zscore_clipped(
            log_energy, cfg.zscore_window
        )

        # 11. Regime indicator (0-4 based on sign quadrant of d_hat, r_hat)
        #   0: both near zero (abs < 0.1)
        #   1: d_hat > 0, r_hat > 0 (informed bullish, prefer index futures)
        #   2: d_hat > 0, r_hat < 0 (informed bullish, prefer options/stocks)
        #   3: d_hat < 0, r_hat > 0 (informed bearish, prefer index futures)
        #   4: d_hat < 0, r_hat < 0 (informed bearish, prefer options/stocks)
        regime = pd.Series(0, index=decomposition.index, dtype=np.int8)
        near_zero = (d_hat.abs() < 0.1) & (r_hat.abs() < 0.1)
        regime = regime.where(
            near_zero,
            other=(
                1 * ((d_hat > 0) & (r_hat > 0)).astype(np.int8)
                + 2 * ((d_hat > 0) & (r_hat <= 0)).astype(np.int8)
                + 3 * ((d_hat <= 0) & (r_hat > 0)).astype(np.int8)
                + 4 * ((d_hat <= 0) & (r_hat <= 0)).astype(np.int8)
            ),
        )
        # For rows where near_zero is True, regime stays 0; otherwise the
        # mutually exclusive conditions assign exactly one of 1-4.
        features["dff_regime"] = regime

        # 12. Momentum: d_hat(t) - d_hat(t - lag)
        features["dff_momentum"] = d_hat - d_hat.shift(cfg.momentum_lag)

        return features

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_zscore_clipped(
        series: pd.Series,
        window: int,
        min_periods: int = 10,
        clip_bound: float = 4.0,
    ) -> pd.Series:
        """Compute rolling z-score with NaN-safe std and clip to [-clip_bound, clip_bound].

        Parameters
        ----------
        series : pd.Series
            Input time series.
        window : int
            Rolling window size.
        min_periods : int
            Minimum non-NaN observations required.
        clip_bound : float
            Symmetric clip boundary for the z-score.

        Returns
        -------
        pd.Series
            Clipped rolling z-score.
        """
        rolling_mean = series.rolling(window, min_periods=min_periods).mean()
        rolling_std = series.rolling(window, min_periods=min_periods).std(ddof=1)
        # Replace zero std with NaN to avoid division by zero
        rolling_std = rolling_std.replace(0.0, np.nan)
        z = (series - rolling_mean) / rolling_std
        return z.clip(lower=-clip_bound, upper=clip_bound)
