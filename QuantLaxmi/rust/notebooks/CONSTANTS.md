# Arb Analysis Constants (Frozen)

## Fee Assumptions
| Parameter | Value | Source |
|-----------|-------|--------|
| Taker fee | 0.001 (0.1%) | Binance spot default tier |
| 3-leg penalty | 0.003 (0.3%) | 3 × taker fee |

## Latency Windows to Test
| Window | Use Case |
|--------|----------|
| 20ms | Optimistic (co-located) |
| 50ms | Realistic (cloud) |
| 100ms | Conservative |

## Quote Age Cutoff
**200ms hard filter** - Discard any observation with quote age > 200ms

## Price/Qty Exponents (Binance SBE)
| Field | Exponent | Meaning |
|-------|----------|---------|
| Price | -2 | Divide mantissa by 100 |
| Qty | -8 | Divide mantissa by 100,000,000 |

## Profile-1 Symbols
### USDT Pairs
- BTCUSDT
- ETHUSDT
- BNBUSDT
- SOLUSDT
- XRPUSDT

### Cross Pairs (for triangles)
- ETHBTC
- BNBBTC
- SOLBTC

## Triangle Definitions
| Triangle | Leg A | Leg B | Leg C |
|----------|-------|-------|-------|
| ETH-BTC | BTCUSDT | ETHBTC | ETHUSDT |
| BNB-BTC | BTCUSDT | BNBBTC | BNBUSDT |
| SOL-BTC | BTCUSDT | SOLBTC | SOLUSDT |

## Residual Formulas

### Clockwise (sell C, buy A, buy B)
```
ε_cw = log(bid_C) - log(ask_A) - log(ask_B)
```

### Counter-clockwise (buy C, sell A, sell B)
```
ε_ccw = log(bid_A) + log(bid_B) - log(ask_C)
```

### Fee-adjusted
```
ε_net = ε_gross - 3f
```

### Latency-adjusted
```
slip_penalty = 3 × σ × sqrt(T_exec)
ε_eff = ε_net - slip_penalty
```

---
*Do not modify these constants during analysis.*
