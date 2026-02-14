# Data Availability Report

## Overview
This document summarizes the availability of continuous futures data from Zerodha Kite for the `QuantKubera` project. The temporal depth of data is critical for training deep learning models like the Temporal Fusion Transformer (TMT).

## Data Requirements (TMT Model)
*   **Minimum Technical Requirement:** > 252 trading days (~1 year).
    *   *Reason:* The model uses a 252-day lookback window for `norm_annual_return` features.
*   **Recommended:** > 10 years.
    *   *Reason:* Deep learning models generally require extensive history covering various market regimes (bull, bear, sideways) to generalize well.

## Zerodha Kite Data Availability
*Date Probed: February 13, 2026*

| Index | Symbol (Continuous) | Earliest Available Date | Approx. Duration | Suitability |
| :--- | :--- | :--- | :--- | :--- |
| **NIFTY 50** | `NIFTY26FEBFUT` | **Jan 2011** | ~14 Years | ✅ **Excellent** |
| **BANK NIFTY** | `BANKNIFTY26FEBFUT` | **Jan 2012** | ~13 Years | ✅ **Excellent** |
| **FIN NIFTY** | `FINNIFTY26FEBFUT` | **Jan 2022** | ~3 Years | ⚠️ **Limited** |
| **MIDCP NIFTY** | `MIDCPNIFTY26FEBFUT` | **Jan 2023** | ~2 Years | ⚠️ **Limited** |

## Conclusion
*   **Primary Datasets:** `NIFTY` and `BANKNIFTY` provide sufficient historical depth (13+ years) for robust model training.
*   **Secondary Datasets:** `FINNIFTY` and `MIDCPNIFTY` can be used for transfer learning or shorter-term validation but may suffer from "insufficient data" issues if long lookback windows (>2-3 years) are strictly required for some advanced features.

## Action Items
*   Configure the training pipeline to prioritize `NIFTY` and `BANKNIFTY`.
*   Ensure the data fetcher handles `continuous=True` to seamlessly stitch contract rollovers.
