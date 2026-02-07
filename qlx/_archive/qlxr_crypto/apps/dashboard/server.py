"""QuantLaxmi Crypto Dashboard — aiohttp server."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from aiohttp import web

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

STATE_FILES = [
    "crypto_flow_state.json",
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


def _win_rate_pct(trades: list) -> float:
    """Win rate where pnl_pct is already in percent."""
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.get("pnl_pct", 0) > 0)
    return round(wins / len(trades) * 100, 1)


# ---------------------------------------------------------------------------
# Strategy readers
# ---------------------------------------------------------------------------


def read_crypto_flow() -> dict:
    d = _load_json("crypto_flow_state.json")
    if not d:
        return _empty("CLRS — Carry Harvester", "Crypto")

    # Aggregate all positions across 4 signal types
    positions = []
    for pool_key in ("carry_positions", "residual_positions",
                     "cascade_positions", "reversion_positions"):
        for k, v in d.get(pool_key, {}).items():
            acc_pnl = v.get("accumulated_pnl", 0)
            acc_cost = v.get("accumulated_cost", 0)
            net = acc_pnl - acc_cost
            # Extract funding % from reason string (e.g. "Funding +80.5%, VPIN 0.31 < 0.99")
            reason = v.get("reason", "")
            funding_str = ""
            if "Funding" in reason:
                funding_str = reason.split(",")[0].replace("Funding ", "")
            positions.append({
                "symbol": v.get("symbol", k),
                "type": v.get("signal_type", "").replace("carry_", "").replace("_", ""),
                "dir": v.get("direction", ""),
                "funding": funding_str,
                "net_bps": round(net * 10000, 1),
                "settle": v.get("n_settlements", 0),
                "weight": f"{v.get('notional_weight', 0):.0%}",
            })

    equity_history = d.get("equity_history", [])
    equity_curve = [[e[0][:16], round(e[1], 6)] for e in equity_history]

    trade_log = d.get("trade_log", [])
    wins = d.get("total_wins", 0)
    exits = d.get("total_exits", 0)
    wr = round(wins / exits * 100, 1) if exits > 0 else 0.0

    recent = [
        {
            "symbol": t.get("symbol", ""),
            "action": t.get("action", ""),
            "time": t.get("time", "")[:16],
            "pnl_pct": round(t.get("pnl", 0) * 100, 2),
            "reason": t.get("reason", "")[:40],
        }
        for t in trade_log[-15:]
    ]

    n_carry = len(d.get("carry_positions", {}))
    total_funding = d.get("total_funding_earned", 0)
    total_costs = d.get("total_costs_paid", 0)

    # Compute started_at duration
    started = d.get("started_at", "")
    duration = ""
    if started:
        try:
            start_dt = datetime.fromisoformat(started)
            now = datetime.now(timezone.utc)
            delta = now - start_dt
            hours = delta.total_seconds() / 3600
            if hours < 1:
                duration = f"{delta.total_seconds()/60:.0f}m"
            elif hours < 24:
                duration = f"{hours:.1f}h"
            else:
                duration = f"{delta.days}d {hours % 24:.0f}h"
        except Exception:
            pass

    return {
        "name": "CLRS — Carry Harvester",
        "market": "Crypto",
        "status": _staleness_file("crypto_flow_state.json"),
        "equity": round(d.get("equity", 1.0), 6),
        "return_pct": round((d.get("equity", 1.0) - 1) * 100, 4),
        "n_open": len(positions),
        "n_closed": exits,
        "win_rate": wr,
        "positions": positions,
        "recent_trades": recent,
        "equity_curve": equity_curve,
        "extra": {
            "positions": n_carry,
            "funding_bps": round(total_funding * 10000, 2),
            "costs_bps": round(total_costs * 10000, 2),
            "net_bps": round((total_funding - total_costs) * 10000, 2),
            "entries": d.get("total_entries", 0),
            "exits": exits,
            "running": duration,
        },
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
    clrs = read_crypto_flow()
    strategies = [clrs]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_equity": clrs["equity"],
        "total_return_pct": clrs["return_pct"],
        "strategies": strategies,
    }


# ---------------------------------------------------------------------------
# SSE file watcher + client registry
# ---------------------------------------------------------------------------

def _get_mtimes() -> dict[str, float]:
    result = {}
    for fn in STATE_FILES:
        path = DATA_DIR / fn
        try:
            result[fn] = path.stat().st_mtime
        except OSError:
            result[fn] = 0.0
    return result


async def _file_watcher(app: web.Application) -> None:
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
<title>CLRS — Crypto Carry Harvester</title>
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
    grid-template-columns: 1fr;
    gap: 16px;
    max-width: 900px;
    margin: 0 auto;
  }
  .card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .card-header { display: flex; justify-content: space-between; align-items: center; }
  .card-title { font-size: 15px; font-weight: 600; display: flex; align-items: center; gap: 8px; }
  .status-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  .status-running { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .status-stale { background: var(--yellow); box-shadow: 0 0 6px var(--yellow); }
  .status-stopped { background: var(--red); box-shadow: 0 0 4px var(--red); }
  .market-tag { font-size: 10px; padding: 2px 8px; border-radius: 4px; background: #2a2a4e; color: var(--blue); font-weight: 600; text-transform: uppercase; }
  .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
  .metric { background: #12122a; padding: 8px; border-radius: 6px; text-align: center; }
  .metric-label { color: var(--text-dim); font-size: 10px; text-transform: uppercase; }
  .metric-value { font-size: 16px; font-weight: 700; margin-top: 2px; }
  .pnl-pos { color: var(--green); }
  .pnl-neg { color: var(--red); }
  .pnl-zero { color: var(--text-dim); }
  .chart-container { height: 120px; }
  table { width: 100%; border-collapse: collapse; font-size: 11px; }
  th { text-align: left; color: var(--text-dim); font-weight: 500; padding: 4px 6px; border-bottom: 1px solid var(--card-border); text-transform: uppercase; font-size: 10px; }
  td { padding: 4px 6px; border-bottom: 1px solid #1e1e30; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 120px; }
  .section-label { font-size: 11px; color: var(--text-dim); text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }
  .extra-grid { display: flex; flex-wrap: wrap; gap: 6px; }
  .extra-chip { background: #12122a; padding: 3px 8px; border-radius: 4px; font-size: 11px; }
</style>
</head>
<body>
<header>
  <h1>CLRS Carry Harvester <span class="live-badge reconnecting" id="live-badge">CONNECTING</span></h1>
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
  html += `<div class="card-header"><div class="card-title"><span class="status-dot status-${s.status}"></span>${s.name}</div><span class="market-tag">${s.market}</span></div>`;
  const avgPnl = s.n_closed > 0 ? s.recent_trades.reduce((a, t) => a + t.pnl_pct, 0) / s.recent_trades.length : 0;
  html += `<div class="metrics">
    <div class="metric"><div class="metric-label">Return</div><div class="metric-value ${pnlClass(s.return_pct)}">${pnlSign(s.return_pct)}%</div></div>
    <div class="metric"><div class="metric-label">Open / Closed</div><div class="metric-value">${s.n_open} / ${s.n_closed}</div></div>
    <div class="metric"><div class="metric-label">Win Rate</div><div class="metric-value">${s.win_rate.toFixed(1)}%</div></div>
    <div class="metric"><div class="metric-label">Avg PnL</div><div class="metric-value ${pnlClass(avgPnl)}">${pnlSign(avgPnl)}%</div></div>
  </div>`;
  if (s.equity_curve.length > 1) { html += `<div class="chart-container"><canvas id="chart-${idx}"></canvas></div>`; }
  if (s.positions.length > 0) {
    html += `<div class="section-label">Positions</div>`;
    const keys = Object.keys(s.positions[0]);
    html += `<table><tr>${keys.map(k => `<th>${k}</th>`).join('')}</tr>`;
    for (const p of s.positions) { html += `<tr>${keys.map(k => { const v = p[k]; if (typeof v === 'number' && k.includes('pnl')) return `<td class="${pnlClass(v)}">${pnlSign(v)}</td>`; return `<td>${v}</td>`; }).join('')}</tr>`; }
    html += `</table>`;
  }
  if (s.recent_trades.length > 0) {
    html += `<div class="section-label">Recent Trades</div>`;
    html += `<table><tr><th>Symbol</th><th>Entry</th><th>Exit</th><th>PnL%</th><th>Reason</th></tr>`;
    for (const t of s.recent_trades.slice().reverse()) { html += `<tr><td>${t.symbol}</td><td>${t.entry}</td><td>${t.exit}</td><td class="${pnlClass(t.pnl_pct)}">${pnlSign(t.pnl_pct)}</td><td>${t.reason}</td></tr>`; }
    html += `</table>`;
  }
  if (Object.keys(s.extra).length > 0) {
    html += `<div class="section-label">Details</div><div class="extra-grid">`;
    for (const [k, v] of Object.entries(s.extra)) {
      if (typeof v === 'object' && v !== null) { for (const [kk, vv] of Object.entries(v)) { html += `<span class="extra-chip">${kk}: ${typeof vv==='number'?vv.toFixed?vv.toFixed(1):vv:vv}</span>`; } }
      else { html += `<span class="extra-chip">${k}: ${v}</span>`; }
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
    data: { labels, datasets: [{ data, borderColor: color, backgroundColor: color + '18', fill: true, pointRadius: 0, borderWidth: 1.5, tension: 0.3 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: { display: false }, y: { display: true, grid: { color: '#1e1e30' }, ticks: { color: '#8888aa', font: { size: 10 }, callback: v => (v*100-100).toFixed(1)+'%' } } },
      interaction: { intersect: false, mode: 'index' },
    }
  });
}

function updateUI(data) {
  const teq = data.total_return_pct;
  document.getElementById('total-eq').className = 'total-eq ' + pnlClass(teq);
  document.getElementById('total-eq').textContent = pnlSign(teq) + '% portfolio';
  document.getElementById('updated').textContent = 'Updated ' + new Date(data.timestamp).toLocaleTimeString();
  document.getElementById('grid').innerHTML = data.strategies.map((s, i) => renderCard(s, i)).join('');
  for (let i = 0; i < data.strategies.length; i++) { if (data.strategies[i].equity_curve.length > 1) drawChart(i, data.strategies[i].equity_curve); }
}

const badge = document.getElementById('live-badge');
const es = new EventSource('/api/stream');
es.onopen = () => { badge.textContent = 'LIVE'; badge.className = 'live-badge connected'; };
es.onmessage = (e) => { updateUI(JSON.parse(e.data)); badge.textContent = 'LIVE'; badge.className = 'live-badge connected'; };
es.onerror = () => { badge.textContent = 'RECONNECTING'; badge.className = 'live-badge reconnecting'; };
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

    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        await runner.cleanup()
