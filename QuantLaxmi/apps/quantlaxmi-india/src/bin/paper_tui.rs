//! View-only TUI for India Paper Trading
//!
//! Renders PaperState emitted by the PaperEngine loop.
//! Does NOT talk to Zerodha directly or execute trades.

use std::io::{self, stdout};
use std::time::Duration;

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};

// =============================================================================
// RAII TERMINAL GUARD (prevents broken terminal on panic/error)
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
use quantlaxmi_paper::PaperState;
use quantlaxmi_runner_india::paper::{IndiaPaperConfig, make_state_channel, run_with_channel};
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Paragraph},
};
use tokio::sync::watch;

/// TUI application state.
struct App {
    state_rx: watch::Receiver<PaperState>,
    initial_capital: f64,
    should_quit: bool,
}

impl App {
    fn new(state_rx: watch::Receiver<PaperState>, initial_capital: f64) -> Self {
        Self {
            state_rx,
            initial_capital,
            should_quit: false,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse CLI arguments
    let args: Vec<String> = std::env::args().collect();
    let mut duration_secs = 120u64;
    let mut initial_capital = 1_000_000.0f64;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--duration-secs" | "-d" => {
                if i + 1 < args.len() {
                    duration_secs = args[i + 1].parse().unwrap_or(120);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--capital" | "-c" => {
                if i + 1 < args.len() {
                    initial_capital = args[i + 1].parse().unwrap_or(1_000_000.0);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--help" | "-h" => {
                println!("India Paper Trading TUI");
                println!();
                println!("USAGE:");
                println!("    paper-tui [OPTIONS]");
                println!();
                println!("OPTIONS:");
                println!("    -d, --duration-secs <SECS>   Duration in seconds (default: 120)");
                println!("    -c, --capital <INR>         Initial capital (default: 1000000)");
                println!("    -h, --help                  Show help");
                println!();
                println!("CONTROLS:");
                println!("    q           Quit TUI");
                return Ok(());
            }
            _ => {
                i += 1;
            }
        }
    }

    // Create state channel
    let (state_tx, state_rx) = make_state_channel();

    // Build config
    let config = IndiaPaperConfig {
        initial_capital,
        duration_secs,
        log_decisions: false, // Disable verbose logging in TUI mode
        ..Default::default()
    };

    // Spawn paper trading engine in background
    let engine_handle = tokio::spawn(async move {
        if let Err(e) = run_with_channel(&config, state_tx).await {
            eprintln!("[TUI] Engine error: {}", e);
        }
    });

    // Run TUI
    let app = App::new(state_rx, initial_capital);
    let result = run_tui(app).await;

    // Cleanup
    engine_handle.abort();

    result
}

async fn run_tui(mut app: App) -> Result<()> {
    // RAII guard ensures terminal is restored even on panic/error
    let _guard = TerminalGuard::enter()?;

    let backend = CrosstermBackend::new(stdout());
    let mut terminal = Terminal::new(backend)?;

    // Main loop
    loop {
        // Get latest state
        let state = app.state_rx.borrow().clone();

        // Check if finished
        if state.is_finished {
            app.should_quit = true;
        }

        // Draw
        terminal.draw(|frame| ui(frame, &state, app.initial_capital))?;

        // Handle input (non-blocking)
        if event::poll(Duration::from_millis(100))?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
            && key.code == KeyCode::Char('q')
        {
            app.should_quit = true;
        }

        if app.should_quit {
            break;
        }

        // Wait for state change or timeout
        tokio::select! {
            _ = app.state_rx.changed() => {}
            _ = tokio::time::sleep(Duration::from_millis(250)) => {}
        }
    }

    // Guard's Drop will restore terminal
    Ok(())
}

fn ui(frame: &mut Frame, state: &PaperState, initial_capital: f64) {
    let area = frame.area();

    // Main layout: header + content + footer
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Min(8),     // Top row (Account + Strategy)
            Constraint::Length(10), // Bottom row (Options + Positions)
            Constraint::Length(3),  // Footer
        ])
        .split(area);

    // Header with timestamp
    let ist_offset = chrono::FixedOffset::east_opt(5 * 3600 + 30 * 60).unwrap();
    let ts_str = state
        .ts
        .map(|t| {
            t.with_timezone(&ist_offset)
                .format("%H:%M:%S IST")
                .to_string()
        })
        .unwrap_or_else(|| "---".to_string());

    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            "QuantLaxmi India - Paper Trading",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  |  "),
        Span::styled(ts_str, Style::default().fg(Color::Yellow)),
    ]))
    .alignment(Alignment::Center)
    .block(Block::default().borders(Borders::ALL));
    frame.render_widget(header, chunks[0]);

    // Top row: Account + Strategy
    let top_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(chunks[1]);

    // Account panel
    render_account_panel(frame, top_chunks[0], state, initial_capital);

    // Strategy panel
    render_strategy_panel(frame, top_chunks[1], state);

    // Bottom row: Options + Positions
    let bottom_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(chunks[2]);

    // Options panel
    render_options_panel(frame, bottom_chunks[0], state);

    // Positions panel
    render_positions_panel(frame, bottom_chunks[1], state);

    // Footer
    let status = if state.is_finished {
        Span::styled("FINISHED", Style::default().fg(Color::Yellow))
    } else {
        Span::styled("RUNNING", Style::default().fg(Color::Green))
    };

    let footer_text = Line::from(vec![
        Span::raw("Status: "),
        status,
        Span::raw("  |  Fills: "),
        Span::styled(
            format!("{}", state.fills),
            Style::default().fg(Color::Green),
        ),
        Span::raw("  Rejects: "),
        Span::styled(
            format!("{}", state.rejections),
            Style::default().fg(Color::Red),
        ),
        Span::raw("  |  Press "),
        Span::styled("q", Style::default().fg(Color::Yellow)),
        Span::raw(" to quit"),
    ]);

    let footer = Paragraph::new(footer_text)
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL));
    frame.render_widget(footer, chunks[3]);
}

fn render_account_panel(frame: &mut Frame, area: Rect, state: &PaperState, initial_capital: f64) {
    let return_pct = if initial_capital > 0.0 {
        (state.total_pnl / initial_capital) * 100.0
    } else {
        0.0
    };

    let pnl_color = if state.total_pnl >= 0.0 {
        Color::Green
    } else {
        Color::Red
    };

    let text = vec![
        Line::from(vec![
            Span::raw("Capital:  "),
            Span::styled(
                format!("{:>10.0}", initial_capital),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::raw("Cash:     "),
            Span::styled(
                format!("{:>10.0}", state.cash),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::raw("Equity:   "),
            Span::styled(
                format!("{:>10.0}", state.equity),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::raw("Total PnL:"),
            Span::styled(
                format!("{:>10.0} ({:+.2}%)", state.total_pnl, return_pct),
                Style::default().fg(pnl_color).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::raw("Fees:     "),
            Span::styled(
                format!("{:>10.2}", state.fees_paid),
                Style::default().fg(Color::Red),
            ),
        ]),
    ];

    let block = Paragraph::new(text).block(Block::default().title("Account").borders(Borders::ALL));
    frame.render_widget(block, area);
}

fn render_strategy_panel(frame: &mut Frame, area: Rect, state: &PaperState) {
    let view = state.strategy_view.as_ref();

    let text = if let Some(v) = view {
        let edge_color = if v.net_edge_rupees >= 0.0 {
            Color::Green
        } else {
            Color::Red
        };

        let decision_color = match v.decision_type.as_str() {
            "Accept" => Color::Green,
            "Refuse" => Color::Red,
            _ => Color::Yellow,
        };

        vec![
            Line::from(vec![
                Span::raw("Strategy: "),
                Span::styled(&v.name, Style::default().fg(Color::Cyan)),
                Span::raw("  Underlying: "),
                Span::styled(&v.underlying, Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::raw("Spot: "),
                Span::styled(
                    v.spot
                        .map(|s| format!("{:.2}", s))
                        .unwrap_or_else(|| "---".to_string()),
                    Style::default().fg(Color::Yellow),
                ),
            ]),
            Line::from(vec![
                Span::raw("Edge:     "),
                Span::styled(
                    format!("{:>8.2}", v.edge_rupees),
                    Style::default().fg(Color::White),
                ),
                Span::raw("  Friction: "),
                Span::styled(
                    format!("{:>8.2}", v.friction_rupees),
                    Style::default().fg(Color::Red),
                ),
            ]),
            Line::from(vec![
                Span::raw("Net Edge: "),
                Span::styled(
                    format!("{:>8.2}", v.net_edge_rupees),
                    Style::default().fg(edge_color).add_modifier(Modifier::BOLD),
                ),
                Span::raw("  (Entry: "),
                Span::styled(
                    format!("{:.2}", v.entry_threshold_rupees),
                    Style::default().fg(Color::Cyan),
                ),
                Span::raw(" / Exit: "),
                Span::styled(
                    format!("{:.2}", v.exit_threshold_rupees),
                    Style::default().fg(Color::Cyan),
                ),
                Span::raw(")"),
            ]),
            Line::from(vec![
                Span::raw("Decision: "),
                Span::styled(&v.decision_type, Style::default().fg(decision_color)),
            ]),
        ]
    } else {
        vec![Line::from(Span::styled(
            "Waiting for strategy data...",
            Style::default().fg(Color::Gray),
        ))]
    };

    let block =
        Paragraph::new(text).block(Block::default().title("Strategy").borders(Borders::ALL));
    frame.render_widget(block, area);
}

fn render_options_panel(frame: &mut Frame, area: Rect, state: &PaperState) {
    let view = state.strategy_view.as_ref();

    let text = if let Some(v) = view {
        let mut lines = vec![Line::from(vec![
            Span::styled("  Symbol", Style::default().fg(Color::Cyan)),
            Span::raw("              "),
            Span::styled("Bid", Style::default().fg(Color::Green)),
            Span::raw("      "),
            Span::styled("Ask", Style::default().fg(Color::Red)),
            Span::raw("      "),
            Span::styled("Mid", Style::default().fg(Color::Yellow)),
            Span::raw("    "),
            Span::styled("Age", Style::default().fg(Color::Gray)),
        ])];

        // Front leg
        if let Some(leg) = &v.front_leg {
            lines.push(Line::from(vec![
                Span::styled("F ", Style::default().fg(Color::Magenta)),
                Span::raw(format!("{:<18}", truncate_symbol(&leg.symbol, 18))),
                Span::styled(
                    format!("{:>6.1}", leg.bid.unwrap_or(0.0)),
                    Style::default().fg(Color::Green),
                ),
                Span::raw("  "),
                Span::styled(
                    format!("{:>6.1}", leg.ask.unwrap_or(0.0)),
                    Style::default().fg(Color::Red),
                ),
                Span::raw("  "),
                Span::styled(
                    format!("{:>6.1}", leg.mid.unwrap_or(0.0)),
                    Style::default().fg(Color::Yellow),
                ),
                Span::raw("  "),
                Span::styled(
                    format!("{}ms", leg.age_ms),
                    Style::default().fg(if leg.age_ms > 5000 {
                        Color::Red
                    } else {
                        Color::Gray
                    }),
                ),
            ]));
        }

        // Back leg
        if let Some(leg) = &v.back_leg {
            lines.push(Line::from(vec![
                Span::styled("B ", Style::default().fg(Color::Blue)),
                Span::raw(format!("{:<18}", truncate_symbol(&leg.symbol, 18))),
                Span::styled(
                    format!("{:>6.1}", leg.bid.unwrap_or(0.0)),
                    Style::default().fg(Color::Green),
                ),
                Span::raw("  "),
                Span::styled(
                    format!("{:>6.1}", leg.ask.unwrap_or(0.0)),
                    Style::default().fg(Color::Red),
                ),
                Span::raw("  "),
                Span::styled(
                    format!("{:>6.1}", leg.mid.unwrap_or(0.0)),
                    Style::default().fg(Color::Yellow),
                ),
                Span::raw("  "),
                Span::styled(
                    format!("{}ms", leg.age_ms),
                    Style::default().fg(if leg.age_ms > 5000 {
                        Color::Red
                    } else {
                        Color::Gray
                    }),
                ),
            ]));
        }

        if v.front_leg.is_none() && v.back_leg.is_none() {
            lines.push(Line::from(Span::styled(
                "No options data",
                Style::default().fg(Color::Gray),
            )));
        }

        // Add expiry info
        if let Some(leg) = &v.front_leg {
            lines.push(Line::from(""));
            lines.push(Line::from(vec![
                Span::raw("Front Expiry: "),
                Span::styled(&leg.expiry, Style::default().fg(Color::Magenta)),
            ]));
        }
        if let Some(leg) = &v.back_leg {
            lines.push(Line::from(vec![
                Span::raw("Back Expiry:  "),
                Span::styled(&leg.expiry, Style::default().fg(Color::Blue)),
            ]));
        }

        lines
    } else {
        vec![Line::from(Span::styled(
            "Waiting for options data...",
            Style::default().fg(Color::Gray),
        ))]
    };

    let block = Paragraph::new(text).block(
        Block::default()
            .title("Options (F=Front, B=Back)")
            .borders(Borders::ALL),
    );
    frame.render_widget(block, area);
}

fn render_positions_panel(frame: &mut Frame, area: Rect, state: &PaperState) {
    let view = state.strategy_view.as_ref();

    let text = if state.open_positions == 0 {
        vec![
            Line::from(""),
            Line::from(Span::styled(
                "  No active trades",
                Style::default().fg(Color::Gray),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "  Strategy is evaluating...",
                Style::default().fg(Color::DarkGray),
            )),
        ]
    } else if let Some(v) = view {
        let mut lines = vec![Line::from(vec![
            Span::styled("Symbol", Style::default().fg(Color::Cyan)),
            Span::raw("        "),
            Span::styled("Qty", Style::default().fg(Color::White)),
            Span::raw("    "),
            Span::styled("PnL", Style::default().fg(Color::Yellow)),
        ])];

        for pos in &v.positions {
            let pnl_color = if pos.unrealized_pnl >= 0.0 {
                Color::Green
            } else {
                Color::Red
            };
            lines.push(Line::from(vec![
                Span::raw(format!("{:<12}", truncate_symbol(&pos.symbol, 12))),
                Span::styled(
                    format!("{:>6}", pos.qty),
                    Style::default().fg(if pos.qty > 0 {
                        Color::Green
                    } else {
                        Color::Red
                    }),
                ),
                Span::raw("  "),
                Span::styled(
                    format!("{:>8.2}", pos.unrealized_pnl),
                    Style::default().fg(pnl_color),
                ),
            ]));
        }

        if v.positions.is_empty() {
            lines.push(Line::from(Span::styled(
                "Position details pending...",
                Style::default().fg(Color::Gray),
            )));
        }

        lines
    } else {
        vec![Line::from(Span::styled(
            "Loading positions...",
            Style::default().fg(Color::Gray),
        ))]
    };

    let title = format!("Positions ({})", state.open_positions);
    let block = Paragraph::new(text).block(Block::default().title(title).borders(Borders::ALL));
    frame.render_widget(block, area);
}

/// Truncate string safely (handles UTF-8 multibyte chars).
fn truncate_symbol(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_len.saturating_sub(3)).collect();
        format!("{}...", truncated)
    }
}
