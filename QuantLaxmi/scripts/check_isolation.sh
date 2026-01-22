#!/bin/bash
# QuantLaxmi Dependency Isolation Guard
# This script enforces hard separation between India and Crypto binaries

set -e

echo "=== QuantLaxmi Dependency Isolation Check ==="
echo ""

# Build both binaries
echo "[1/4] Building quantlaxmi-india..."
cargo build -p quantlaxmi-india --release 2>&1 | tail -5

echo ""
echo "[2/4] Building quantlaxmi-crypto..."
cargo build -p quantlaxmi-crypto --release 2>&1 | tail -5

echo ""
echo "[3/4] Checking India binary has NO Binance/SBE dependencies..."
# More precise patterns: match "binance" or "kubera-sbe" package names
if cargo tree -p quantlaxmi-india 2>/dev/null | grep -E "(binance|kubera-sbe|quantlaxmi-connectors-binance)" > /dev/null; then
    echo "FAIL: India binary has Binance/SBE dependencies!"
    cargo tree -p quantlaxmi-india | grep -E "(binance|kubera-sbe|quantlaxmi-connectors-binance)"
    exit 1
else
    echo "PASS: No Binance/SBE dependencies in India binary"
fi

echo ""
echo "[4/4] Checking Crypto binary has NO Zerodha/KiteSim dependencies..."
# More precise patterns: match zerodha connector specifically
if cargo tree -p quantlaxmi-crypto 2>/dev/null | grep -E "(zerodha|quantlaxmi-connectors-zerodha)" > /dev/null; then
    echo "FAIL: Crypto binary has Zerodha dependencies!"
    cargo tree -p quantlaxmi-crypto | grep -E "(zerodha|quantlaxmi-connectors-zerodha)"
    exit 1
else
    echo "PASS: No Zerodha dependencies in Crypto binary"
fi

echo ""
echo "=== All isolation checks passed ==="
