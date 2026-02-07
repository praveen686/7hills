"""QuantLaxmi Paper Trading Dashboard — aiohttp server."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from aiohttp import web

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# The 3 state files we watch for changes
STATE_FILES = [
    "iv_paper_state.json",
    "india_news_state.json",
    "micro_paper_state.json",
]

# ---------------------------------------------------------------------------
# Staleness helper
# ---------------------------------------------------------------------------

def _staleness_file(filename: str) -> str:
    """Return 'running' / 'stale' / 'stopped' based on file modification time."""
    path = DATA_DIR / filename
    if not path.exists():
        return "stopped"
    try:
        mtime = path.stat().st_mtime
        age = datetime.now(timezone.utc).timestamp() - mtime
    except Exception:
        return "stopped"
    if age < 900:       # <15 min
        return "running"
    if age < 3600:      # 15-60 min
        return "stale"
    return "stopped"


def _load_json(filename: str) -> dict | None:
    path = DATA_DIR / filename
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _win_rate(trades: list) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.get("pnl_pct", 0) > 0)
    return round(wins / len(trades) * 100, 1)


# ---------------------------------------------------------------------------
# Strategy readers — each returns a uniform dict
# ---------------------------------------------------------------------------

def read_iv_mean_reversion() -> dict:
    d = _load_json("iv_paper_state.json")
    if not d:
        return _empty("IV Mean-Reversion", "India")

    closed = d.get("closed_trades", [])

    # Build latest spot lookup from iv_histories
    latest_spot: dict[str, float] = {}
    for sym, hist in d.get("iv_histories", {}).items():
        if hist:
            latest_spot[sym] = hist[-1].get("spot", 0)

    positions_raw = d.get("positions", {})
    positions = []
    for sym, pos in positions_raw.items():
        if pos is not None:
            entry = pos.get("entry_spot", 0)
            current = latest_spot.get(sym, entry)
            pnl = ((current - entry) / entry * 100) if entry else 0.0
            positions.append({
                "symbol": sym,
                "entry_spot": round(entry, 2),
                "current_spot": round(current, 2),
                "pnl_pct": round(pnl, 2),
                "iv_pctile": round(pos.get("iv_pctile", 0) * 100, 1),
                "hold_days": pos.get("hold_days", 0),
            })

    # Reconstruct equity curve from closed trades
    equity_curve = []
    eq = 1.0
    for t in closed:
        eq *= (1 + t.get("pnl_pct", 0))
        equity_curve.append([t.get("exit_date", ""), round(eq, 6)])

    # Per-index IV percentiles from last history entry
    extra = {}
    for sym, hist in d.get("iv_histories", {}).items():
        if hist:
            last = hist[-1]
            extra[sym] = {
                "atm_iv": round(last.get("atm_iv", 0) * 100, 2),
                "spot": round(last.get("spot", 0), 2),
            }

    return {
        "name": "IV Mean-Reversion",
        "market": "India",
        "status": _staleness_file("iv_paper_state.json"),
        "equity": round(d.get("equity", 1.0), 6),
        "return_pct": round((d.get("equity", 1.0) - 1) * 100, 2),
        "n_open": len(positions),
        "n_closed": len(closed),
        "win_rate": _win_rate(closed),
        "positions": positions,
        "recent_trades": [
            {
                "symbol": t["symbol"],
                "entry": t.get("entry_date", ""),
                "exit": t.get("exit_date", ""),
                "pnl_pct": round(t.get("pnl_pct", 0) * 100, 2),
                "reason": t.get("exit_reason", ""),
            }
            for t in closed[-10:]
        ],
        "equity_curve": equity_curve,
        "extra": extra,
    }


def read_india_news() -> dict:
    d = _load_json("india_news_state.json")
    if not d:
        return _empty("India News Sentiment", "India")

    active = d.get("active_trades", [])
    # active_trades can be a list or dict
    if isinstance(active, dict):
        active = list(active.values())
    closed = d.get("closed_trades", [])

    positions = []
    for t in active:
        entry = t.get("entry_price", 0)
        current = t.get("current_price")
        direction = t.get("direction", "long")
        if current and entry:
            raw = (current - entry) / entry if direction == "long" else (entry - current) / entry
            pnl = round(raw * 100, 2)
        else:
            pnl = None
        pos = {
            "symbol": t.get("symbol", ""),
            "direction": direction,
            "entry_price": round(entry, 2),
        }
        if current:
            pos["current_price"] = round(current, 2)
            pos["pnl_pct"] = pnl
        pos["event_type"] = t.get("event_type", "")
        pos["hold_days"] = t.get("hold_days", 0)
        positions.append(pos)

    # Equity curve from closed trades (pnl_pct is fractional here)
    equity_curve = []
    eq = 1.0
    for t in closed:
        eq *= (1 + t.get("pnl_pct", 0))
        equity_curve.append([t.get("exit_date", ""), round(eq, 6)])

    # Event type breakdown
    event_counts: dict[str, int] = {}
    for t in active:
        et = t.get("event_type", "unknown")
        event_counts[et] = event_counts.get(et, 0) + 1

    return {
        "name": "India News Sentiment",
        "market": "India",
        "status": _staleness_file("india_news_state.json"),
        "equity": round(d.get("equity", 1.0), 6),
        "return_pct": round((d.get("equity", 1.0) - 1) * 100, 2),
        "n_open": len(positions),
        "n_closed": len(closed),
        "win_rate": _win_rate(closed),
        "positions": positions,
        "recent_trades": [
            {
                "symbol": t["symbol"],
                "entry": t.get("entry_date", ""),
                "exit": t.get("exit_date", ""),
                "pnl_pct": round(t.get("pnl_pct", 0) * 100, 2),
                "reason": t.get("exit_reason", ""),
            }
            for t in closed[-10:]
        ],
        "equity_curve": equity_curve,
        "extra": {"event_types": event_counts},
    }


def read_microstructure() -> dict:
    d = _load_json("micro_paper_state.json")
    if not d:
        return _empty("Intraday Microstructure", "India")

    closed = d.get("closed_trades", [])
    positions_raw = d.get("positions", {})
    positions = [
        {
            "symbol": k,
            **{kk: vv for kk, vv in v.items() if kk != "symbol"},
        }
        for k, v in positions_raw.items()
    ] if isinstance(positions_raw, dict) else []

    # Compute equity from closed trades
    eq = 1.0
    equity_curve = []
    for t in closed:
        pnl = t.get("pnl_pct", 0)
        # micro trades store pnl_pct as percent (like 0.75 = 0.75%)
        eq *= (1 + pnl / 100)
        equity_curve.append([t.get("exit_time", "")[:10], round(eq, 6)])

    # Latest analytics per symbol
    analytics = d.get("analytics_log", [])
    latest_by_sym: dict[str, dict] = {}
    for entry in analytics:
        sym = entry.get("symbol", "")
        latest_by_sym[sym] = {
            "gex_regime": entry.get("gex_regime", ""),
            "combined_score": round(entry.get("combined_score", 0), 3),
            "signal": entry.get("signal_direction", "flat"),
            "pcr": round(entry.get("pcr_oi", 0), 2),
        }

    return {
        "name": "Intraday Microstructure",
        "market": "India",
        "status": _staleness_file("micro_paper_state.json"),
        "equity": round(eq, 6),
        "return_pct": round((eq - 1) * 100, 2),
        "n_open": len(positions),
        "n_closed": len(closed),
        "win_rate": _win_rate(closed),
        "positions": positions,
        "recent_trades": [
            {
                "symbol": t.get("symbol", ""),
                "entry": t.get("entry_time", "")[:16],
                "exit": t.get("exit_time", "")[:16],
                "pnl_pct": round(t.get("pnl_pct", 0), 2),
                "reason": t.get("exit_reason", ""),
            }
            for t in closed[-10:]
        ],
        "equity_curve": equity_curve,
        "extra": {"analytics": latest_by_sym},
    }


def _empty(name: str, market: str) -> dict:
    return {
        "name": name,
        "market": market,
        "status": "stopped",
        "equity": 1.0,
        "return_pct": 0.0,
        "n_open": 0,
        "n_closed": 0,
        "win_rate": 0.0,
        "positions": [],
        "recent_trades": [],
        "equity_curve": [],
        "extra": {},
    }



# ---------------------------------------------------------------------------
# Shared payload builder
# ---------------------------------------------------------------------------

def _build_payload() -> dict:
    strategies = [
        read_iv_mean_reversion(),
        read_india_news(),
        read_microstructure(),
    ]
    total_eq = 1.0
    for s in strategies:
        total_eq *= s["equity"]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_equity": round(total_eq, 6),
        "total_return_pct": round((total_eq - 1) * 100, 2),
        "strategies": strategies,
    }


# ---------------------------------------------------------------------------
# SSE file watcher + client registry
# ---------------------------------------------------------------------------

def _get_mtimes() -> dict[str, float]:
    """Return {filename: mtime} for all state files. Missing files get 0."""
    result = {}
    for fn in STATE_FILES:
        path = DATA_DIR / fn
        try:
            result[fn] = path.stat().st_mtime
        except OSError:
            result[fn] = 0.0
    return result


async def _file_watcher(app: web.Application) -> None:
    """Poll state file mtimes every 2s; push to SSE clients on change."""
    last_mtimes = _get_mtimes()
    while True:
        await asyncio.sleep(2)
        current = _get_mtimes()
        if current != last_mtimes:
            last_mtimes = current
            payload_json = json.dumps(_build_payload())
            dead: list[asyncio.Queue] = []
            for q in app["sse_clients"]:
                try:
                    q.put_nowait(payload_json)
                except asyncio.QueueFull:
                    dead.append(q)
            for q in dead:
                app["sse_clients"].discard(q)


async def start_file_watcher(app: web.Application) -> None:
    app["sse_clients"] = set()
    app["file_watcher"] = asyncio.create_task(_file_watcher(app))


async def stop_file_watcher(app: web.Application) -> None:
    task = app.get("file_watcher")
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

async def handle_api_state(request: web.Request) -> web.Response:
    return web.json_response(_build_payload())


async def handle_sse(request: web.Request) -> web.StreamResponse:
    resp = web.StreamResponse()
    resp.content_type = "text/event-stream"
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    await resp.prepare(request)

    q: asyncio.Queue[str] = asyncio.Queue(maxsize=8)
    request.app["sse_clients"].add(q)

    try:
        # Send current state immediately on connect
        initial = json.dumps(_build_payload())
        await resp.write(f"data: {initial}\n\n".encode())

        while True:
            payload_json = await q.get()
            await resp.write(f"data: {payload_json}\n\n".encode())
    except (ConnectionResetError, ConnectionAbortedError, asyncio.CancelledError):
        pass
    finally:
        request.app["sse_clients"].discard(q)

    return resp


async def handle_index(request: web.Request) -> web.Response:
    return web.Response(text=HTML_PAGE, content_type="text/html")


# ---------------------------------------------------------------------------
# HTML frontend
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QuantLaxmi Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0f0f1a;
    --card: #1a1a2e;
    --card-border: #2a2a3e;
    --text: #e0e0e0;
    --text-dim: #8888aa;
    --green: #00e676;
    --red: #ff5252;
    --yellow: #ffd740;
    --blue: #448aff;
    --mono: 'Fira Code', 'SF Mono', 'Cascadia Code', monospace;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--mono);
    font-size: 13px;
    padding: 16px;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    margin-bottom: 16px;
  }
  header h1 { font-size: 18px; font-weight: 600; display: flex; align-items: center; gap: 10px; }
  .live-badge {
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    transition: all 0.3s ease;
  }
  .live-badge.connected { background: #00e67630; color: var(--green); }
  .live-badge.reconnecting { background: #ffd74030; color: var(--yellow); }
  .header-right { text-align: right; }
  .header-right .total-eq { font-size: 22px; font-weight: 700; }
  .header-right .updated { color: var(--text-dim); font-size: 11px; }
  .grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
  }
  @media (max-width: 1000px) { .grid { grid-template-columns: 1fr; } }
  .card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .card-title {
    font-size: 15px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .status-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    display: inline-block;
  }
  .status-running { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .status-stale { background: var(--yellow); box-shadow: 0 0 6px var(--yellow); }
  .status-stopped { background: var(--red); box-shadow: 0 0 4px var(--red); }
  .market-tag {
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 4px;
    background: #2a2a4e;
    color: var(--blue);
    font-weight: 600;
    text-transform: uppercase;
  }
  .metrics {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
  }
  .metric {
    background: #12122a;
    padding: 8px;
    border-radius: 6px;
    text-align: center;
  }
  .metric-label { color: var(--text-dim); font-size: 10px; text-transform: uppercase; }
  .metric-value { font-size: 16px; font-weight: 700; margin-top: 2px; }
  .pnl-pos { color: var(--green); }
  .pnl-neg { color: var(--red); }
  .pnl-zero { color: var(--text-dim); }
  .chart-container { height: 120px; }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
  }
  th {
    text-align: left;
    color: var(--text-dim);
    font-weight: 500;
    padding: 4px 6px;
    border-bottom: 1px solid var(--card-border);
    text-transform: uppercase;
    font-size: 10px;
  }
  td {
    padding: 4px 6px;
    border-bottom: 1px solid #1e1e30;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 120px;
  }
  .section-label {
    font-size: 11px;
    color: var(--text-dim);
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.5px;
  }
  .extra-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .extra-chip {
    background: #12122a;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 11px;
  }
</style>
</head>
<body>
<header>
  <h1>QuantLaxmi Paper Trading <span class="live-badge reconnecting" id="live-badge">CONNECTING</span></h1>
  <div class="header-right">
    <div class="total-eq" id="total-eq">-</div>
    <div class="updated" id="updated">Loading...</div>
  </div>
</header>
<div class="grid" id="grid"></div>

<script>
const charts = {};

function pnlClass(v) { return v > 0.001 ? 'pnl-pos' : v < -0.001 ? 'pnl-neg' : 'pnl-zero'; }
function pnlSign(v) { return v > 0 ? '+' + v.toFixed(2) : v.toFixed(2); }

function renderCard(s, idx) {
  const id = 'card-' + idx;
  let html = `<div class="card" id="${id}">`;

  // Header
  html += `<div class="card-header">
    <div class="card-title">
      <span class="status-dot status-${s.status}"></span>
      ${s.name}
    </div>
    <span class="market-tag">${s.market}</span>
  </div>`;

  // Metrics
  const avgPnl = s.n_closed > 0
    ? s.recent_trades.reduce((a, t) => a + t.pnl_pct, 0) / s.recent_trades.length
    : 0;
  html += `<div class="metrics">
    <div class="metric">
      <div class="metric-label">Return</div>
      <div class="metric-value ${pnlClass(s.return_pct)}">${pnlSign(s.return_pct)}%</div>
    </div>
    <div class="metric">
      <div class="metric-label">Open / Closed</div>
      <div class="metric-value">${s.n_open} / ${s.n_closed}</div>
    </div>
    <div class="metric">
      <div class="metric-label">Win Rate</div>
      <div class="metric-value">${s.win_rate.toFixed(1)}%</div>
    </div>
    <div class="metric">
      <div class="metric-label">Avg PnL</div>
      <div class="metric-value ${pnlClass(avgPnl)}">${pnlSign(avgPnl)}%</div>
    </div>
  </div>`;

  // Equity chart
  if (s.equity_curve.length > 1) {
    html += `<div class="chart-container"><canvas id="chart-${idx}"></canvas></div>`;
  }

  // Active positions
  if (s.positions.length > 0) {
    html += `<div class="section-label">Positions</div>`;
    const keys = Object.keys(s.positions[0]);
    html += `<table><tr>${keys.map(k => `<th>${k}</th>`).join('')}</tr>`;
    for (const p of s.positions) {
      html += `<tr>${keys.map(k => {
        const v = p[k];
        if (typeof v === 'number' && k.includes('pnl')) return `<td class="${pnlClass(v)}">${pnlSign(v)}</td>`;
        return `<td>${v}</td>`;
      }).join('')}</tr>`;
    }
    html += `</table>`;
  }

  // Recent trades
  if (s.recent_trades.length > 0) {
    html += `<div class="section-label">Recent Trades</div>`;
    html += `<table><tr><th>Symbol</th><th>Entry</th><th>Exit</th><th>PnL%</th><th>Reason</th></tr>`;
    for (const t of s.recent_trades.slice().reverse()) {
      html += `<tr>
        <td>${t.symbol}</td>
        <td>${t.entry}</td>
        <td>${t.exit}</td>
        <td class="${pnlClass(t.pnl_pct)}">${pnlSign(t.pnl_pct)}</td>
        <td>${t.reason}</td>
      </tr>`;
    }
    html += `</table>`;
  }

  // Extra info
  if (Object.keys(s.extra).length > 0) {
    html += `<div class="section-label">Details</div><div class="extra-grid">`;
    for (const [k, v] of Object.entries(s.extra)) {
      if (typeof v === 'object' && v !== null) {
        for (const [kk, vv] of Object.entries(v)) {
          if (typeof vv === 'object') {
            const parts = Object.entries(vv).map(([a,b]) => `${a}:${typeof b==='number'?b.toFixed?b.toFixed(1):b:b}`).join(' ');
            html += `<span class="extra-chip">${kk} ${parts}</span>`;
          } else {
            html += `<span class="extra-chip">${kk}: ${typeof vv==='number'?vv.toFixed?vv.toFixed(1):vv:vv}</span>`;
          }
        }
      } else {
        html += `<span class="extra-chip">${k}: ${v}</span>`;
      }
    }
    html += `</div>`;
  }

  html += `</div>`;
  return html;
}

function drawChart(idx, curve) {
  const canvas = document.getElementById('chart-' + idx);
  if (!canvas) return;
  if (charts[idx]) { charts[idx].destroy(); }

  const labels = curve.map(p => p[0]);
  const data = curve.map(p => p[1]);
  const color = data[data.length-1] >= 1.0 ? '#00e676' : '#ff5252';

  charts[idx] = new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        data,
        borderColor: color,
        backgroundColor: color + '18',
        fill: true,
        pointRadius: 0,
        borderWidth: 1.5,
        tension: 0.3,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { display: false },
        y: {
          display: true,
          grid: { color: '#1e1e30' },
          ticks: { color: '#8888aa', font: { size: 10 }, callback: v => (v*100-100).toFixed(1)+'%' },
        }
      },
      interaction: { intersect: false, mode: 'index' },
    }
  });
}

function updateUI(data) {
  // Header
  const teq = data.total_return_pct;
  document.getElementById('total-eq').className = 'total-eq ' + pnlClass(teq);
  document.getElementById('total-eq').textContent = pnlSign(teq) + '% portfolio';
  const ts = new Date(data.timestamp);
  document.getElementById('updated').textContent = 'Updated ' + ts.toLocaleTimeString();

  // Cards
  const grid = document.getElementById('grid');
  grid.innerHTML = data.strategies.map((s, i) => renderCard(s, i)).join('');

  // Charts
  for (let i = 0; i < data.strategies.length; i++) {
    const s = data.strategies[i];
    if (s.equity_curve.length > 1) {
      drawChart(i, s.equity_curve);
    }
  }
}

// SSE real-time connection
const badge = document.getElementById('live-badge');
const es = new EventSource('/api/stream');
es.onopen = () => {
  badge.textContent = 'LIVE';
  badge.className = 'live-badge connected';
};
es.onmessage = (e) => {
  const data = JSON.parse(e.data);
  updateUI(data);
  badge.textContent = 'LIVE';
  badge.className = 'live-badge connected';
};
es.onerror = () => {
  badge.textContent = 'RECONNECTING';
  badge.className = 'live-badge reconnecting';
  document.getElementById('updated').textContent = 'Reconnecting...';
};
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# App entry
# ---------------------------------------------------------------------------

async def main(port: int = 8080):
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/state", handle_api_state)
    app.router.add_get("/api/stream", handle_sse)

    app.on_startup.append(start_file_watcher)
    app.on_cleanup.append(stop_file_watcher)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    print(f"Dashboard running at http://127.0.0.1:{port}")
    print("Access via SSH tunnel: ssh -L {0}:localhost:{0} ubuntu@<ec2-ip>".format(port))

    # Run forever
    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        await runner.cleanup()
