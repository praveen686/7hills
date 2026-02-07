//! Full ratatui dashboard for basis mean-reversion.
//!
//! 4-pane layout:
//! - Top-left:     Z-Score Monitor (all tracked symbols)
//! - Top-right:    Active Trades with basis P&L
//! - Bottom-left:  Trade Statistics (win rate, avg hold, avg P&L)
//! - Bottom-right: Portfolio summary
//!
//! Usage: cargo run -p basis-harvester --bin basis-tui [INITIAL_CAPITAL] [TOP_N]

use std::io;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};
use tokio::sync::watch;
use basis_harvester::feed::BasisFeed;
use basis_harvester::fill_model::BinanceFillModel;
use basis_harvester::scanner::{self, BasisOpportunity};
use basis_harvester::strategy::{BasisMeanRevStrategy, StrategyConfig};
use quantlaxmi_paper::{EngineConfig, PaperEngine, PaperState};

// ---------------------------------------------------------------------------
// Terminal RAII guard
// ---------------------------------------------------------------------------

struct TerminalGuard;

impl TerminalGuard {
    fn enter() -> anyhow::Result<Self> {
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let initial_capital: f64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000.0);

    let top_n: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);

    eprintln!("Fetching 24h volumes...");
    let volumes = scanner::fetch_24h_volumes().await.unwrap_or_default();

    eprintln!("Scanning basis...");
    let entries = scanner::fetch_premium_index().await?;
    let all_opps = scanner::rank_basis_opportunities(&entries, &volumes);

    eprintln!("Fetching spot exchange info...");
    let spot_symbols = scanner::fetch_spot_symbols().await?;
    let opps = scanner::filter_spot_available(&all_opps, &spot_symbols);

    let symbols: Vec<String> = opps.iter().take(top_n).map(|o| o.symbol.clone()).collect();

    if symbols.is_empty() {
        eprintln!("No basis pairs found.");
        return Ok(());
    }

    // Watch channel for TUI state
    let (state_tx, state_rx) = watch::channel(PaperState::default());

    // Symbol rotation + pinned symbols
    let (symbol_tx, symbol_rx) = watch::channel(symbols.clone());
    let (pinned_tx, pinned_rx) = watch::channel(Vec::<String>::new());

    let feed = BasisFeed::connect(symbols)
        .await?
        .with_symbol_updates(symbol_rx)
        .with_pinned_symbols(pinned_rx);

    let config = StrategyConfig::default();
    let strategy =
        BasisMeanRevStrategy::new(config, initial_capital).with_pinned_channel(pinned_tx);
    let fill_model = BinanceFillModel::default();

    let engine_config = EngineConfig::new(initial_capital).with_state_channel(state_tx);
    let mut engine = PaperEngine::with_config(feed, strategy, fill_model, engine_config);

    // Shared scanner data for TUI display
    let scanner_data: Arc<Mutex<Vec<BasisOpportunity>>> =
        Arc::new(Mutex::new(opps.into_iter().take(20).collect()));

    // Spawn engine in background
    tokio::spawn(async move {
        if let Err(e) = engine.run().await {
            eprintln!("[TUI] Engine error: {}", e);
        }
    });

    // Background scanner refresh + symbol rotation
    let scanner_data_bg = Arc::clone(&scanner_data);
    tokio::spawn(async move {
        let mut cached_spot = spot_symbols;
        let mut cached_volumes = volumes;
        let mut spot_refresh = Instant::now();
        let mut vol_refresh = Instant::now();

        loop {
            tokio::time::sleep(Duration::from_secs(5)).await;

            if spot_refresh.elapsed() >= Duration::from_secs(300) {
                if let Ok(new_spot) = scanner::fetch_spot_symbols().await {
                    cached_spot = new_spot;
                }
                spot_refresh = Instant::now();
            }

            if vol_refresh.elapsed() >= Duration::from_secs(60) {
                if let Ok(v) = scanner::fetch_24h_volumes().await {
                    cached_volumes = v;
                }
                vol_refresh = Instant::now();
            }

            if let Ok(entries) = scanner::fetch_premium_index().await {
                let all = scanner::rank_basis_opportunities(&entries, &cached_volumes);
                let filtered = scanner::filter_spot_available(&all, &cached_spot);

                let top20: Vec<BasisOpportunity> = filtered.iter().take(20).cloned().collect();
                if let Ok(mut data) = scanner_data_bg.lock() {
                    *data = top20;
                }

                let new_symbols: Vec<String> =
                    filtered.iter().take(top_n).map(|o| o.symbol.clone()).collect();
                let current = symbol_tx.borrow().clone();
                let current_set: std::collections::HashSet<&String> = current.iter().collect();
                let has_new = new_symbols.iter().any(|s| !current_set.contains(s));
                if has_new {
                    let _ = symbol_tx.send(new_symbols);
                }
            }
        }
    });

    // Run TUI
    let _guard = TerminalGuard::enter()?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;

    run_tui(&mut terminal, state_rx, scanner_data, initial_capital).await
}

// ---------------------------------------------------------------------------
// TUI loop
// ---------------------------------------------------------------------------

async fn run_tui<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    mut rx: watch::Receiver<PaperState>,
    scanner_data: Arc<Mutex<Vec<BasisOpportunity>>>,
    initial_capital: f64,
) -> anyhow::Result<()> {
    let mut last_draw = Instant::now();
    let draw_every = Duration::from_millis(33);

    loop {
        if event::poll(Duration::from_millis(10))?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
        {
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                _ => {}
            }
        }

        let state = (*rx.borrow()).clone();

        if state.is_finished {
            let scanner_top = scanner_data.lock().unwrap().clone();
            terminal.draw(|f| render(f, &state, &scanner_top, initial_capital))?;
            tokio::time::sleep(Duration::from_secs(2)).await;
            return Ok(());
        }

        if last_draw.elapsed() >= draw_every {
            let scanner_top = scanner_data.lock().unwrap().clone();
            terminal.draw(|f| render(f, &state, &scanner_top, initial_capital))?;
            last_draw = Instant::now();
        }

        let _ = tokio::time::timeout(Duration::from_millis(16), rx.changed()).await;
    }
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

fn render(
    f: &mut ratatui::Frame<'_>,
    state: &PaperState,
    scanner_top: &[BasisOpportunity],
    initial_capital: f64,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(3),
        ])
        .split(f.area());

    render_header(f, chunks[0], state);

    let main_cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[1]);

    let left_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(main_cols[0]);

    let right_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(main_cols[1]);

    render_z_scores(f, left_rows[0], scanner_top);
    render_trades(f, right_rows[0], state);
    render_statistics(f, left_rows[1], state);
    render_portfolio(f, right_rows[1], state, initial_capital);
    render_footer(f, chunks[2], state);
}

fn render_header(f: &mut ratatui::Frame<'_>, area: ratatui::layout::Rect, state: &PaperState) {
    let ts = state
        .ts
        .map(|t| t.format("%H:%M:%S UTC").to_string())
        .unwrap_or_else(|| "connecting...".into());

    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            "Basis Harvester",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  |  "),
        Span::styled(ts, Style::default().fg(Color::Yellow)),
        Span::raw("  |  Spot-Perp Basis Mean-Reversion"),
    ]))
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(header, area);
}

fn render_z_scores(
    f: &mut ratatui::Frame<'_>,
    area: ratatui::layout::Rect,
    opps: &[BasisOpportunity],
) {
    let mut lines = vec![Line::from(vec![Span::styled(
        format!(
            "{:<14} {:>8} {:>8} {:>8}",
            "SYMBOL", "BASIS", "|BASIS|", "FUND(bp)"
        ),
        Style::default().fg(Color::Gray),
    )])];

    for opp in opps.iter().take(12) {
        let basis_color = if opp.basis_bps > 0.0 {
            Color::Green
        } else {
            Color::Red
        };
        lines.push(Line::from(vec![
            Span::styled(
                format!("{:<14}", opp.symbol),
                Style::default().fg(Color::White),
            ),
            Span::styled(
                format!(" {:>8.1}", opp.basis_bps),
                Style::default().fg(basis_color),
            ),
            Span::styled(
                format!(" {:>8.1}", opp.abs_basis_bps),
                Style::default().fg(Color::Yellow),
            ),
            Span::styled(
                format!(" {:>8.2}", opp.funding_rate_bps),
                Style::default().fg(Color::Cyan),
            ),
        ]));
    }

    let widget = Paragraph::new(lines)
        .block(
            Block::default()
                .title(" Z-Score Monitor — Top Basis (live) ")
                .borders(Borders::ALL),
        )
        .wrap(Wrap { trim: true });
    f.render_widget(widget, area);
}

fn render_trades(f: &mut ratatui::Frame<'_>, area: ratatui::layout::Rect, state: &PaperState) {
    let mut lines = Vec::new();

    if let Some(ref view) = state.strategy_view {
        if view.positions.is_empty() {
            lines.push(Line::from(Span::styled(
                "No active trades",
                Style::default().fg(Color::Gray),
            )));
        } else {
            lines.push(Line::from(Span::styled(
                format!("{:<12} {:>6} {:>10} {:>10}", "SYMBOL", "QTY", "AVG", "P&L"),
                Style::default().fg(Color::Gray),
            )));
            for pos in &view.positions {
                let pnl_color = if pos.unrealized_pnl >= 0.0 {
                    Color::Green
                } else {
                    Color::Red
                };
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("{:<12}", pos.symbol),
                        Style::default().fg(Color::White),
                    ),
                    Span::styled(
                        format!(" {:>6}", pos.qty),
                        Style::default().fg(if pos.qty > 0 {
                            Color::Green
                        } else {
                            Color::Red
                        }),
                    ),
                    Span::styled(
                        format!(" {:>10.4}", pos.avg_price),
                        Style::default().fg(Color::Yellow),
                    ),
                    Span::styled(
                        format!(" {:>10.2}", pos.unrealized_pnl),
                        Style::default().fg(pnl_color),
                    ),
                ]));
            }
        }
    } else {
        lines.push(Line::from(Span::styled(
            "Waiting for data...",
            Style::default().fg(Color::Gray),
        )));
    }

    let widget = Paragraph::new(lines)
        .block(
            Block::default()
                .title(" Active Trades ")
                .borders(Borders::ALL),
        )
        .wrap(Wrap { trim: true });
    f.render_widget(widget, area);
}

fn render_statistics(f: &mut ratatui::Frame<'_>, area: ratatui::layout::Rect, state: &PaperState) {
    let lines = vec![
        Line::from(vec![
            Span::styled("Trades:   ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", state.fills / 2), // each trade = 2 fills (spot + perp legs)
                Style::default().fg(Color::Cyan),
            ),
        ]),
        Line::from(vec![
            Span::styled("Realized: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("${:.4}", state.realized_pnl),
                Style::default()
                    .fg(if state.realized_pnl >= 0.0 {
                        Color::Green
                    } else {
                        Color::Red
                    })
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("Fees:     ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("${:.4}", state.fees_paid),
                Style::default().fg(Color::Red),
            ),
        ]),
        Line::from(vec![
            Span::styled("Unreal:   ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("${:.4}", state.unrealized_pnl),
                Style::default()
                    .fg(if state.unrealized_pnl >= 0.0 {
                        Color::Green
                    } else {
                        Color::Red
                    }),
            ),
        ]),
        Line::from(vec![
            Span::styled("Net P&L:  ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("${:.4}", state.total_pnl),
                Style::default()
                    .fg(if state.total_pnl >= 0.0 {
                        Color::Green
                    } else {
                        Color::Red
                    })
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
    ];

    let widget = Paragraph::new(lines)
        .block(
            Block::default()
                .title(" Trade Statistics ")
                .borders(Borders::ALL),
        )
        .wrap(Wrap { trim: true });
    f.render_widget(widget, area);
}

fn render_portfolio(
    f: &mut ratatui::Frame<'_>,
    area: ratatui::layout::Rect,
    state: &PaperState,
    initial_capital: f64,
) {
    let return_pct = if initial_capital > 0.0 {
        state.total_pnl / initial_capital * 100.0
    } else {
        0.0
    };

    let lines = vec![
        Line::from(vec![
            Span::styled("Equity:    ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("${:.2}", state.equity),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("Return:    ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:+.2}%", return_pct),
                Style::default().fg(if return_pct >= 0.0 {
                    Color::Green
                } else {
                    Color::Red
                }),
            ),
        ]),
        Line::from(vec![
            Span::styled("Positions: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", state.open_positions),
                Style::default().fg(Color::Cyan),
            ),
        ]),
    ];

    let widget = Paragraph::new(lines)
        .block(
            Block::default()
                .title(" Portfolio ")
                .borders(Borders::ALL),
        )
        .wrap(Wrap { trim: true });
    f.render_widget(widget, area);
}

fn render_footer(f: &mut ratatui::Frame<'_>, area: ratatui::layout::Rect, state: &PaperState) {
    let decision = state.last_decision.as_deref().unwrap_or("---");

    let footer = Paragraph::new(Line::from(vec![
        Span::styled(
            format!("Fills: {} | Rejects: {}", state.fills, state.rejections),
            Style::default().fg(Color::Gray),
        ),
        Span::raw("  |  "),
        Span::styled(
            format!("Last: {}", &decision[..decision.len().min(60)]),
            Style::default().fg(Color::DarkGray),
        ),
        Span::raw("  |  "),
        Span::styled("q: quit", Style::default().fg(Color::DarkGray)),
    ]))
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(footer, area);
}
