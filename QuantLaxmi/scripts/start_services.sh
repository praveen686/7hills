#!/usr/bin/env bash
# QuantLaxmi service launcher — kills stale processes, starts fresh backend + frontend
set -euo pipefail

BACKEND_PORT=8000
FRONTEND_PORT=3000
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QLAXMI_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$QLAXMI_DIR/.." && pwd)"
VENV="$QLAXMI_DIR/env/bin/activate"
BACKEND_DIR="$QLAXMI_DIR"
FRONTEND_DIR="$QLAXMI_DIR/ui"
BACKEND_LOG="/tmp/quantlaxmi_backend.log"
FRONTEND_LOG="/tmp/quantlaxmi_frontend.log"

# ── helpers ──────────────────────────────────────────────────────
kill_port() {
    local port=$1
    local pids
    pids=$(lsof -t -i :"$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "⚡ Port $port in use (PIDs: $pids) — killing..."
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 1
        # verify
        if lsof -i :"$port" >/dev/null 2>&1; then
            echo "✗ Failed to free port $port"
            exit 1
        fi
        echo "✓ Port $port freed"
    else
        echo "✓ Port $port already free"
    fi
}

wait_for_port() {
    local port=$1
    local name=$2
    local max_wait=30
    local i=0
    while [ $i -lt $max_wait ]; do
        if curl -s "http://localhost:$port" >/dev/null 2>&1; then
            echo "✓ $name is up on port $port"
            return 0
        fi
        sleep 1
        i=$((i + 1))
    done
    echo "✗ $name failed to start on port $port (waited ${max_wait}s)"
    echo "  Check log: $3"
    return 1
}

# ── cleanup ──────────────────────────────────────────────────────
echo "── Cleaning up ports ──"
kill_port $BACKEND_PORT
kill_port $FRONTEND_PORT
# Kill ALL stray next-server / next-dev processes (prevents stacking)
stray=$(pgrep -f 'next-server|next dev' 2>/dev/null || true)
if [ -n "$stray" ]; then
    echo "  Killing stray Next.js processes: $stray"
    echo "$stray" | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# ── start backend ────────────────────────────────────────────────
echo ""
echo "── Starting backend (uvicorn on :$BACKEND_PORT) ──"
source "$VENV"
cd "$BACKEND_DIR"
nohup uvicorn engine.api.app:app \
    --host 0.0.0.0 \
    --port $BACKEND_PORT \
    --reload \
    > "$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!
echo "  PID: $BACKEND_PID | Log: $BACKEND_LOG"

# ── start frontend ───────────────────────────────────────────────
echo ""
echo "── Starting frontend (next dev on :$FRONTEND_PORT) ──"
cd "$FRONTEND_DIR"
nohup npx next dev --port $FRONTEND_PORT \
    > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
echo "  PID: $FRONTEND_PID | Log: $FRONTEND_LOG"

# ── health checks ────────────────────────────────────────────────
echo ""
echo "── Waiting for services ──"
wait_for_port $BACKEND_PORT "Backend"  "$BACKEND_LOG"
wait_for_port $FRONTEND_PORT "Frontend" "$FRONTEND_LOG"

# ── summary ──────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════"
echo "  Backend:  http://localhost:$BACKEND_PORT"
echo "  Frontend: http://localhost:$FRONTEND_PORT"
echo "  Logs:     $BACKEND_LOG / $FRONTEND_LOG"
echo "═══════════════════════════════════════════"
