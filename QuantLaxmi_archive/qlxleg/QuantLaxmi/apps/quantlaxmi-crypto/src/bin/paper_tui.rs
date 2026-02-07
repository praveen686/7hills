//! Crypto Paper Trading TUI (HTTP Client)
//!
//! View-only + interactive TUI for crypto paper trading.
//! Connects to the paper trading HTTP server (does NOT spawn engine).
//!
//! Controls:
//!   q     - Quit
//!   b     - Market buy (configurable qty)
//!   s     - Market sell (configurable qty)
//!   +/-   - Adjust order quantity
//!   r     - Refresh immediately

use std::io::{self, stdout};
use std::time::Duration;

use anyhow::Result;
use chrono::{DateTime, FixedOffset, Utc};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Paragraph},
};
use serde::{Deserialize, Serialize};

// =============================================================================
// RAII TERMINAL GUARD
// =============================================================================

struct TerminalGuard;

impl TerminalGuard {
    fn enter() -> Result<Self> {
        enable_raw_mode()?;
        execute!(io::stdout(), EnterAlternateScreen)?;
        Ok(Self)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
    }
}

// =============================================================================
// API RESPONSE TYPES
// =============================================================================

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
struct TopResponse {
    symbol: String,
    best_bid: Option<f64>,
    best_ask: Option<f64>,
    spread: Option<f64>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
struct PositionResponse {
    symbol: String,
    position: f64, // Server uses 'position' not 'qty'
    realized_pnl: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OrderRequest {
    symbol: String,
    side: String,
    qty: f64,
    order_type: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
struct OrderResponse {
    order_id: String,
    status: String,
    #[serde(default)]
    filled_qty: Option<f64>,
    #[serde(default)]
    filled_price: Option<f64>,
    #[serde(default)]
    message: Option<String>,
}

// =============================================================================
// TUI STATE
// =============================================================================

#[derive(Debug, Clone, Default)]
struct CryptoPaperState {
    ts: Option<DateTime<Utc>>,
    symbol: String,
    best_bid: Option<f64>,
    best_ask: Option<f64>,
    spread: Option<f64>,
    position_qty: f64,
    avg_price: f64,
    realized_pnl: f64,
    unrealized_pnl: f64,
    last_order_status: Option<String>,
    error: Option<String>,
}

struct App {
    base_url: String,
    symbol: String,
    order_qty: f64,
    state: CryptoPaperState,
    should_quit: bool,
    client: reqwest::Client,
}

impl App {
    fn new(base_url: String, symbol: String, order_qty: f64) -> Self {
        // Create client with timeout to prevent hanging
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            base_url,
            symbol: symbol.clone(),
            order_qty,
            state: CryptoPaperState {
                symbol,
                ..Default::default()
            },
            should_quit: false,
            client,
        }
    }

    async fn fetch_top(&mut self) {
        let url = format!("{}/top?symbol={}", self.base_url, self.symbol);
        match self.client.get(&url).send().await {
            Ok(resp) => {
                if resp.status().is_success() {
                    if let Ok(top) = resp.json::<TopResponse>().await {
                        self.state.best_bid = top.best_bid;
                        self.state.best_ask = top.best_ask;
                        self.state.spread = top.spread;
                        self.state.ts = Some(Utc::now());
                        self.state.error = None;
                    } else {
                        self.state.error = Some("Failed to parse response".to_string());
                    }
                } else {
                    self.state.error = Some(format!("HTTP {}", resp.status()));
                }
            }
            Err(e) => {
                // Distinguish timeout from other errors
                let err_str = e.to_string();
                if err_str.contains("timed out") || err_str.contains("timeout") {
                    self.state.error = Some("Request timeout - server busy?".to_string());
                } else {
                    self.state.error = Some(format!("Connection: {}", err_str));
                }
            }
        }
    }

    async fn fetch_position(&mut self) {
        let url = format!("{}/position?symbol={}", self.base_url, self.symbol);
        match self.client.get(&url).send().await {
            Ok(resp) => {
                if resp.status().is_success()
                    && let Ok(pos) = resp.json::<PositionResponse>().await
                {
                    self.state.position_qty = pos.position;
                    self.state.realized_pnl = pos.realized_pnl;
                    // Server doesn't provide avg_price or unrealized_pnl
                    // Those would need to be computed client-side or added to the API
                }
                // Position endpoint might 404 if no position - that's ok
            }
            Err(_) => {
                // Ignore position fetch errors (might not have position)
            }
        }
    }

    async fn place_order(&mut self, side: &str) {
        let url = format!("{}/order", self.base_url);
        let order = OrderRequest {
            symbol: self.symbol.clone(),
            side: side.to_string(),
            qty: self.order_qty,
            order_type: "MARKET".to_string(),
        };

        match self.client.post(&url).json(&order).send().await {
            Ok(resp) => {
                if resp.status().is_success() {
                    if let Ok(result) = resp.json::<OrderResponse>().await {
                        self.state.last_order_status = Some(format!(
                            "{} {} @ {:?}",
                            result.status,
                            result.filled_qty.unwrap_or(0.0),
                            result.filled_price
                        ));
                    }
                } else {
                    self.state.last_order_status =
                        Some(format!("Order failed: HTTP {}", resp.status()));
                }
            }
            Err(e) => {
                self.state.last_order_status = Some(format!("Order error: {}", e));
            }
        }
    }

    async fn refresh(&mut self) {
        self.fetch_top().await;
        self.fetch_position().await;
    }
}

// =============================================================================
// MAIN
// =============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut base_url = "http://localhost:8080".to_string();
    let mut symbol = "BTCUSDT".to_string();
    let mut order_qty = 0.001_f64;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--url" | "-u" => {
                if i + 1 < args.len() {
                    base_url = args[i + 1].clone();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--symbol" | "-s" => {
                if i + 1 < args.len() {
                    symbol = args[i + 1].clone();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--qty" | "-q" => {
                if i + 1 < args.len() {
                    order_qty = args[i + 1].parse().unwrap_or(0.001);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--help" | "-h" => {
                println!("Crypto Paper Trading TUI (HTTP Client)");
                println!();
                println!("USAGE:");
                println!("    crypto-paper-tui [OPTIONS]");
                println!();
                println!("OPTIONS:");
                println!(
                    "    -u, --url <URL>       Paper trading server URL (default: http://localhost:8080)"
                );
                println!("    -s, --symbol <SYM>    Trading symbol (default: BTCUSDT)");
                println!("    -q, --qty <QTY>       Order quantity (default: 0.001)");
                println!("    -h, --help            Show help");
                println!();
                println!("CONTROLS:");
                println!("    q           Quit");
                println!("    b           Market BUY");
                println!("    s           Market SELL");
                println!("    +/=         Increase order qty");
                println!("    -           Decrease order qty");
                println!("    r           Refresh immediately");
                return Ok(());
            }
            _ => {
                i += 1;
            }
        }
    }

    let mut app = App::new(base_url, symbol, order_qty);
    run_tui(&mut app).await
}

async fn run_tui(app: &mut App) -> Result<()> {
    // RAII guard ensures terminal is restored even on panic/error
    let _guard = TerminalGuard::enter()?;

    let backend = CrosstermBackend::new(stdout());
    let mut terminal = Terminal::new(backend)?;

    // Initial fetch
    app.refresh().await;

    let mut last_refresh = std::time::Instant::now();
    let refresh_interval = Duration::from_millis(200); // Fast updates

    loop {
        // Always refresh if interval elapsed
        if last_refresh.elapsed() >= refresh_interval {
            app.refresh().await;
            last_refresh = std::time::Instant::now();
        }

        // Draw
        terminal.draw(|frame| ui(frame, app))?;

        // Handle input with short poll (keeps UI responsive)
        if event::poll(Duration::from_millis(50))?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
        {
            match key.code {
                KeyCode::Char('q') => app.should_quit = true,
                KeyCode::Char('b') => {
                    app.place_order("BUY").await;
                    app.refresh().await;
                }
                KeyCode::Char('s') => {
                    app.place_order("SELL").await;
                    app.refresh().await;
                }
                KeyCode::Char('+') | KeyCode::Char('=') => {
                    app.order_qty *= 2.0;
                }
                KeyCode::Char('-') => {
                    app.order_qty = (app.order_qty / 2.0).max(0.0001);
                }
                KeyCode::Char('r') => {
                    app.refresh().await;
                    last_refresh = std::time::Instant::now();
                }
                _ => {}
            }
        }

        if app.should_quit {
            break;
        }
    }

    Ok(())
}

// =============================================================================
// UI RENDERING
// =============================================================================

fn ui(frame: &mut Frame, app: &App) {
    let area = frame.area();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Length(7), // Market panel
            Constraint::Length(8), // Position panel
            Constraint::Min(3),    // Order panel
            Constraint::Length(3), // Footer
        ])
        .split(area);

    // Header
    render_header(frame, chunks[0], app);

    // Market panel
    render_market_panel(frame, chunks[1], app);

    // Position panel
    render_position_panel(frame, chunks[2], app);

    // Order panel
    render_order_panel(frame, chunks[3], app);

    // Footer
    render_footer(frame, chunks[4], app);
}

fn render_header(frame: &mut Frame, area: Rect, app: &App) {
    let ts_str = app
        .state
        .ts
        .map(|t| {
            let utc_offset = FixedOffset::east_opt(0).unwrap();
            t.with_timezone(&utc_offset)
                .format("%H:%M:%S UTC")
                .to_string()
        })
        .unwrap_or_else(|| "---".to_string());

    let status = if app.state.error.is_some() {
        Span::styled("DISCONNECTED", Style::default().fg(Color::Red))
    } else {
        Span::styled("CONNECTED", Style::default().fg(Color::Green))
    };

    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            "QuantLaxmi Crypto - Paper Trading",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  |  "),
        status,
        Span::raw("  |  "),
        Span::styled(ts_str, Style::default().fg(Color::Yellow)),
    ]))
    .alignment(Alignment::Center)
    .block(Block::default().borders(Borders::ALL));
    frame.render_widget(header, area);
}

fn render_market_panel(frame: &mut Frame, area: Rect, app: &App) {
    let bid_str = app
        .state
        .best_bid
        .map(|b| format!("{:.2}", b))
        .unwrap_or_else(|| "---".to_string());
    let ask_str = app
        .state
        .best_ask
        .map(|a| format!("{:.2}", a))
        .unwrap_or_else(|| "---".to_string());
    let spread_str = app
        .state
        .spread
        .map(|s| format!("{:.2}", s))
        .unwrap_or_else(|| "---".to_string());

    let mid_price = match (app.state.best_bid, app.state.best_ask) {
        (Some(b), Some(a)) => Some((b + a) / 2.0),
        _ => None,
    };
    let mid_str = mid_price
        .map(|m| format!("{:.2}", m))
        .unwrap_or_else(|| "---".to_string());

    let text = vec![
        Line::from(vec![
            Span::raw("Symbol: "),
            Span::styled(
                &app.state.symbol,
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::raw("Bid:    "),
            Span::styled(bid_str, Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::raw("Ask:    "),
            Span::styled(ask_str, Style::default().fg(Color::Red)),
        ]),
        Line::from(vec![
            Span::raw("Mid:    "),
            Span::styled(mid_str, Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::raw("Spread: "),
            Span::styled(spread_str, Style::default().fg(Color::Gray)),
        ]),
    ];

    let block = Paragraph::new(text).block(Block::default().title("Market").borders(Borders::ALL));
    frame.render_widget(block, area);
}

fn render_position_panel(frame: &mut Frame, area: Rect, app: &App) {
    let total_pnl = app.state.realized_pnl + app.state.unrealized_pnl;
    let pnl_color = if total_pnl >= 0.0 {
        Color::Green
    } else {
        Color::Red
    };

    let pos_color = if app.state.position_qty > 0.0 {
        Color::Green
    } else if app.state.position_qty < 0.0 {
        Color::Red
    } else {
        Color::Gray
    };

    let pos_str = if app.state.position_qty.abs() < 0.00001 {
        "FLAT".to_string()
    } else if app.state.position_qty > 0.0 {
        format!("LONG {:.6}", app.state.position_qty)
    } else {
        format!("SHORT {:.6}", app.state.position_qty.abs())
    };

    let text = vec![
        Line::from(vec![
            Span::raw("Position:     "),
            Span::styled(pos_str, Style::default().fg(pos_color)),
        ]),
        Line::from(vec![
            Span::raw("Avg Price:    "),
            Span::styled(
                format!("{:.2}", app.state.avg_price),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::raw("Unrealized:   "),
            Span::styled(
                format!("{:+.4}", app.state.unrealized_pnl),
                Style::default().fg(if app.state.unrealized_pnl >= 0.0 {
                    Color::Green
                } else {
                    Color::Red
                }),
            ),
        ]),
        Line::from(vec![
            Span::raw("Realized:     "),
            Span::styled(
                format!("{:+.4}", app.state.realized_pnl),
                Style::default().fg(if app.state.realized_pnl >= 0.0 {
                    Color::Green
                } else {
                    Color::Red
                }),
            ),
        ]),
        Line::from(vec![
            Span::raw("Total PnL:    "),
            Span::styled(
                format!("{:+.4}", total_pnl),
                Style::default().fg(pnl_color).add_modifier(Modifier::BOLD),
            ),
        ]),
    ];

    let block =
        Paragraph::new(text).block(Block::default().title("Position").borders(Borders::ALL));
    frame.render_widget(block, area);
}

fn render_order_panel(frame: &mut Frame, area: Rect, app: &App) {
    let last_status = app
        .state
        .last_order_status
        .as_deref()
        .unwrap_or("No orders yet");

    let error_line = if let Some(err) = &app.state.error {
        Line::from(vec![Span::styled(
            format!("Error: {}", err),
            Style::default().fg(Color::Red),
        )])
    } else {
        Line::from("")
    };

    let text = vec![
        Line::from(vec![
            Span::raw("Order Qty:    "),
            Span::styled(
                format!("{:.6}", app.order_qty),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  (+/- to adjust)"),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Last Order:   "),
            Span::styled(last_status, Style::default().fg(Color::Gray)),
        ]),
        error_line,
    ];

    let block = Paragraph::new(text).block(Block::default().title("Orders").borders(Borders::ALL));
    frame.render_widget(block, area);
}

fn render_footer(frame: &mut Frame, area: Rect, _app: &App) {
    let footer_text = Line::from(vec![
        Span::styled("b", Style::default().fg(Color::Green)),
        Span::raw(" BUY  "),
        Span::styled("s", Style::default().fg(Color::Red)),
        Span::raw(" SELL  "),
        Span::styled("+/-", Style::default().fg(Color::Yellow)),
        Span::raw(" Qty  "),
        Span::styled("r", Style::default().fg(Color::Cyan)),
        Span::raw(" Refresh  "),
        Span::styled("q", Style::default().fg(Color::Magenta)),
        Span::raw(" Quit"),
    ]);

    let footer = Paragraph::new(footer_text)
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL));
    frame.render_widget(footer, area);
}
