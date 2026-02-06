# QuantLaxmi Python SDK

Python SDK for building trading strategies that integrate with the QuantLaxmi platform.

## Historical Note

This SDK was originally named `kubera_sdk` and was renamed to `quantlaxmi_sdk` as part of the Phase 4.8 Kubera decommission (January 2026).

## Usage

```python
from quantlaxmi_sdk import QuantLaxmiStrategy, MarketEvent, BarPayload

class MyStrategy(QuantLaxmiStrategy):
    def __init__(self):
        super().__init__("MyStrategy")

    def on_tick(self, event: MarketEvent):
        # Handle tick
        pass

    def on_bar(self, symbol: str, bar: BarPayload):
        # Handle bar
        pass
```

## Backwards Compatibility

For migration purposes, `KuberaStrategy` is available as a deprecated alias for `QuantLaxmiStrategy`:

```python
# Deprecated - use QuantLaxmiStrategy instead
from quantlaxmi_sdk import KuberaStrategy
```

## API

### Classes

- `QuantLaxmiStrategy` - Base class for trading strategies
- `MarketEvent` - Market tick event
- `MarketPayload` - Market data payload
- `BarPayload` - OHLCV bar data
- `OrderStatus` - Order execution status
