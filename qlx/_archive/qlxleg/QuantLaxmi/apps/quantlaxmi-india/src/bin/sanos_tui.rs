//! View-only TUI for SANOS Calendar Carry Paper Trading
//!
//! Renders PaperState + SanosTuiState from the SANOS-gated PaperEngine loop.
//! Layout: Header | Account + SANOS Surface | Gate Status + IV Term Structure | Options + Positions | Footer

use std::io::{self, stdout};
use std::time::Duration;

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use quantlaxmi_paper::PaperState;
use quantlaxmi_runner_india::init_observability;
use quantlaxmi_runner_india::sanos_cal_carry::{
    self, SanosCalCarryConfig, SanosTuiState, make_state_channels, run_with_channel,
};
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Paragraph},
};
use tokio::sync::watch;

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

/// TUI application state.
struct App {
    state_rx: watch::Receiver<PaperState>,
    sanos_rx: watch::Receiver<SanosTuiState>,
    initial_capital: f64,
    should_quit: bool,
}

impl App {
    fn new(
        state_rx: watch::Receiver<PaperState>,
        sanos_rx: watch::Receiver<SanosTuiState>,
        initial_capital: f64,
    ) -> Self {
        Self {
            state_rx,
            sanos_rx,
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
    let mut underlying = "NIFTY".to_string();
    let mut headless = false;
    let mut relax_e_gates = false;

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
            "--underlying" | "-u" => {
                if i + 1 < args.len() {
                    underlying = args[i + 1].clone();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--headless" => {
                headless = true;
                i += 1;
            }
            "--relax-gates" => {
                relax_e_gates = true;
                i += 1;
            }
            "--help" | "-h" => {
                println!("SANOS Calendar Carry TUI");
                println!();
                println!("USAGE:");
                println!("    sanos-tui [OPTIONS]");
                println!();
                println!("OPTIONS:");
                println!("    -d, --duration-secs <SECS>    Duration in seconds (default: 120)");
                println!("    -c, --capital <INR>           Initial capital (default: 1000000)");
                println!("    -u, --underlying <SYM>        Underlying (default: NIFTY)");
                println!("        --headless                Run without TUI (log to stderr)");
                println!(
                    "        --relax-gates             Relax E1/E2/E3 gates for pipeline testing"
                );
                println!("    -h, --help                    Show help");
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

    // Build config
    let lot_size = if underlying == "BANKNIFTY" { 15 } else { 25 };
    let config = SanosCalCarryConfig {
        underlying,
        initial_capital,
        duration_secs,
        lot_size,
        log_decisions: headless, // enable decision logging in headless mode
        relax_e_gates,
        ..Default::default()
    };

    if headless {
        let _guards = init_observability("sanos-headless");
        return sanos_cal_carry::run(&config).await;
    }

    // Create state channels
    let (paper_tx, paper_rx, sanos_tx, sanos_rx) = make_state_channels();

    // Spawn engine in background
    let engine_handle = tokio::spawn(async move {
        if let Err(e) = run_with_channel(&config, paper_tx, sanos_tx).await {
            eprintln!("[SANOS-TUI] Engine error: {}", e);
        }
    });

    // Run TUI
    let app = App::new(paper_rx, sanos_rx, initial_capital);
    let result = run_tui(app).await;

    // Cleanup
    engine_handle.abort();

    result
}

async fn run_tui(mut app: App) -> Result<()> {
    let _guard = TerminalGuard::enter()?;

    let backend = CrosstermBackend::new(stdout());
    let mut terminal = Terminal::new(backend)?;

    loop {
        let state = app.state_rx.borrow().clone();
        let sanos_state = app.sanos_rx.borrow().clone();

        if state.is_finished {
            app.should_quit = true;
        }

        terminal.draw(|frame| ui(frame, &state, &sanos_state, app.initial_capital))?;

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

        tokio::select! {
            _ = app.state_rx.changed() => {}
            _ = app.sanos_rx.changed() => {}
            _ = tokio::time::sleep(Duration::from_millis(250)) => {}
        }
    }

    Ok(())
}

fn ui(frame: &mut Frame, state: &PaperState, sanos: &SanosTuiState, initial_capital: f64) {
    let area = frame.area();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Min(6),     // Top row (Account + SANOS Surface)
            Constraint::Length(12), // Middle row (Gates + IV Term Structure)
            Constraint::Length(8),  // Bottom row (Options + Positions)
            Constraint::Length(3),  // Footer
        ])
        .split(area);

    // Header
    let ist_offset = chrono::FixedOffset::east_opt(5 * 3600 + 30 * 60).unwrap();
    let ts_str = state
        .ts
        .map(|t| {
            t.with_timezone(&ist_offset)
                .format("%H:%M:%S IST")
                .to_string()
        })
        .unwrap_or_else(|| "---".to_string());

    let calib_str = sanos
        .secs_since_calibration
        .map(|s| format!("Calib: {:.0}s ago", s))
        .unwrap_or_else(|| "Calib: warmup".to_string());

    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            "SANOS Calendar Carry",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  |  "),
        Span::styled(ts_str, Style::default().fg(Color::Yellow)),
        Span::raw("  |  "),
        Span::styled(
            calib_str,
            Style::default().fg(if sanos.warmup {
                Color::Red
            } else {
                Color::Green
            }),
        ),
    ]))
    .alignment(Alignment::Center)
    .block(Block::default().borders(Borders::ALL));
    frame.render_widget(header, chunks[0]);

    // Top row: Account + SANOS Surface
    let top_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(chunks[1]);

    render_account_panel(frame, top_chunks[0], state, initial_capital);
    render_surface_panel(frame, top_chunks[1], sanos);

    // Middle row: Gates + IV Term Structure
    let mid_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
        .split(chunks[2]);

    render_gates_panel(frame, mid_chunks[0], sanos);
    render_iv_panel(frame, mid_chunks[1], sanos);

    // Bottom row: Options + Positions
    let bot_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(chunks[3]);

    render_decision_panel(frame, bot_chunks[0], state, sanos);
    render_positions_panel(frame, bot_chunks[1], state);

    // Footer
    let status = if state.is_finished {
        Span::styled("FINISHED", Style::default().fg(Color::Yellow))
    } else {
        Span::styled("RUNNING", Style::default().fg(Color::Green))
    };

    let footer = Paragraph::new(Line::from(vec![
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
    ]))
    .alignment(Alignment::Center)
    .block(Block::default().borders(Borders::ALL));
    frame.render_widget(footer, chunks[4]);
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
    ];

    let block = Paragraph::new(text).block(Block::default().title("Account").borders(Borders::ALL));
    frame.render_widget(block, area);
}

fn render_surface_panel(frame: &mut Frame, area: Rect, sanos: &SanosTuiState) {
    let text = if sanos.surfaces.is_empty() {
        vec![Line::from(Span::styled(
            "Waiting for SANOS calibration...",
            Style::default().fg(Color::Gray),
        ))]
    } else {
        let mut lines = vec![Line::from(vec![
            Span::styled("  Expiry", Style::default().fg(Color::Cyan)),
            Span::raw("      "),
            Span::styled("LP Status", Style::default().fg(Color::White)),
            Span::raw("    "),
            Span::styled("MaxErr", Style::default().fg(Color::Yellow)),
            Span::raw("  "),
            Span::styled("Forward", Style::default().fg(Color::White)),
        ])];

        for s in &sanos.surfaces {
            let status_color = if s.lp_status.contains("Optimal") {
                Color::Green
            } else {
                Color::Red
            };

            lines.push(Line::from(vec![
                Span::raw(format!("  {:<12}", &s.expiry)),
                Span::styled(
                    format!("{:<10}", truncate(&s.lp_status, 10)),
                    Style::default().fg(status_color),
                ),
                Span::styled(
                    format!("{:>6.4}", s.max_fit_error),
                    Style::default().fg(Color::Yellow),
                ),
                Span::raw("  "),
                Span::styled(
                    format!("{:>8.1}", s.forward),
                    Style::default().fg(Color::White),
                ),
            ]));
        }

        lines
    };

    let block = Paragraph::new(text).block(
        Block::default()
            .title("SANOS Surface")
            .borders(Borders::ALL),
    );
    frame.render_widget(block, area);
}

fn render_gates_panel(frame: &mut Frame, area: Rect, sanos: &SanosTuiState) {
    let text = if sanos.gates.is_empty() {
        vec![Line::from(Span::styled(
            "Waiting for gate evaluation...",
            Style::default().fg(Color::Gray),
        ))]
    } else {
        let mut lines = Vec::new();

        // Render gates in 2 columns
        let mid = sanos.gates.len().div_ceil(2);
        for row in 0..mid {
            let mut spans = Vec::new();

            // Left column
            if let Some(g) = sanos.gates.get(row) {
                let (marker, color) = if g.passed {
                    ("PASS", Color::Green)
                } else {
                    ("FAIL", Color::Red)
                };
                spans.push(Span::styled(
                    format!("{:>4}", marker),
                    Style::default().fg(color),
                ));
                spans.push(Span::raw(" "));
                spans.push(Span::styled(
                    format!("{:<22}", &g.name),
                    Style::default().fg(Color::White),
                ));

                if let Some(v) = g.value {
                    spans.push(Span::styled(
                        format!("{:>6.2}", v),
                        Style::default().fg(Color::Yellow),
                    ));
                } else {
                    spans.push(Span::raw("      "));
                }
            }

            spans.push(Span::raw("  "));

            // Right column
            if let Some(g) = sanos.gates.get(row + mid) {
                let (marker, color) = if g.passed {
                    ("PASS", Color::Green)
                } else {
                    ("FAIL", Color::Red)
                };
                spans.push(Span::styled(
                    format!("{:>4}", marker),
                    Style::default().fg(color),
                ));
                spans.push(Span::raw(" "));
                spans.push(Span::styled(
                    format!("{:<22}", &g.name),
                    Style::default().fg(Color::White),
                ));

                if let Some(v) = g.value {
                    spans.push(Span::styled(
                        format!("{:>6.2}", v),
                        Style::default().fg(Color::Yellow),
                    ));
                }
            }

            lines.push(Line::from(spans));
        }

        // Add margin gate status
        lines.push(Line::from(""));
        let all_pass = sanos.gates.iter().all(|g| g.passed);
        let summary_color = if all_pass { Color::Green } else { Color::Red };
        let pass_count = sanos.gates.iter().filter(|g| g.passed).count();
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(
                format!("Gates: {}/{} passed", pass_count, sanos.gates.len()),
                Style::default()
                    .fg(summary_color)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));

        lines
    };

    let block = Paragraph::new(text).block(
        Block::default()
            .title("Gate Status (12 gates)")
            .borders(Borders::ALL),
    );
    frame.render_widget(block, area);
}

fn render_iv_panel(frame: &mut Frame, area: Rect, sanos: &SanosTuiState) {
    let text = if let Some(f) = &sanos.features {
        let mut lines = vec![];

        // IV term structure
        lines.push(Line::from(vec![
            Span::raw("IV T1:  "),
            Span::styled(
                format!("{:.3}%", f.iv1 * 100.0),
                Style::default().fg(Color::Cyan),
            ),
        ]));
        if let Some(iv2) = f.iv2 {
            lines.push(Line::from(vec![
                Span::raw("IV T2:  "),
                Span::styled(
                    format!("{:.3}%", iv2 * 100.0),
                    Style::default().fg(Color::Cyan),
                ),
            ]));
        }
        if let Some(iv3) = f.iv3 {
            lines.push(Line::from(vec![
                Span::raw("IV T3:  "),
                Span::styled(
                    format!("{:.3}%", iv3 * 100.0),
                    Style::default().fg(Color::Cyan),
                ),
            ]));
        }

        lines.push(Line::from(""));

        // Calendar gaps
        if let Some(cal12) = f.cal12 {
            lines.push(Line::from(vec![
                Span::raw("Cal 1-2: "),
                Span::styled(
                    format!("{:.4}", cal12),
                    Style::default().fg(if cal12 > 0.0 {
                        Color::Green
                    } else {
                        Color::Red
                    }),
                ),
            ]));
        }
        if let Some(cal23) = f.cal23 {
            lines.push(Line::from(vec![
                Span::raw("Cal 2-3: "),
                Span::styled(
                    format!("{:.4}", cal23),
                    Style::default().fg(if cal23 > 0.0 {
                        Color::Green
                    } else {
                        Color::Red
                    }),
                ),
            ]));
        }

        // Term structure slopes
        if let Some(ts12) = f.ts12 {
            lines.push(Line::from(vec![
                Span::raw("TS 1-2:  "),
                Span::styled(format!("{:.4}", ts12), Style::default().fg(Color::Yellow)),
            ]));
        }

        // Skew
        if let Some(sk1) = f.sk1 {
            lines.push(Line::from(vec![
                Span::raw("Skew T1: "),
                Span::styled(format!("{:.4}", sk1), Style::default().fg(Color::Magenta)),
            ]));
        }

        lines
    } else {
        vec![Line::from(Span::styled(
            "Waiting for IV features...",
            Style::default().fg(Color::Gray),
        ))]
    };

    let block = Paragraph::new(text).block(
        Block::default()
            .title("IV Term Structure")
            .borders(Borders::ALL),
    );
    frame.render_widget(block, area);
}

fn render_decision_panel(frame: &mut Frame, area: Rect, state: &PaperState, sanos: &SanosTuiState) {
    let view = state.strategy_view.as_ref();
    // Available text width = panel width minus border (2) and padding
    let max_text = area.width.saturating_sub(3) as usize;

    let text = if let Some(v) = view {
        let decision_color = match v.decision_type.as_str() {
            "Accept" => Color::Green,
            "Refuse" => Color::Red,
            _ => Color::Yellow,
        };

        let mut lines = vec![
            Line::from(vec![
                Span::raw("Decision: "),
                Span::styled(&v.decision_type, Style::default().fg(decision_color)),
            ]),
            Line::from(vec![
                Span::raw("Reason: "),
                Span::styled(&v.decision_reason, Style::default().fg(Color::White)),
            ]),
        ];

        if let Some(spot) = v.spot {
            lines.push(Line::from(vec![
                Span::raw("Spot: "),
                Span::styled(format!("{:.2}", spot), Style::default().fg(Color::Yellow)),
            ]));
        }

        // Show SANOS-specific decision
        if let Some(d) = &sanos.last_decision {
            lines.push(Line::from(vec![
                Span::raw("SANOS: "),
                Span::styled(
                    truncate(d, max_text.saturating_sub(7)),
                    Style::default().fg(Color::Gray),
                ),
            ]));
        }

        lines
    } else {
        vec![Line::from(Span::styled(
            "Waiting for strategy data...",
            Style::default().fg(Color::Gray),
        ))]
    };

    let block = Paragraph::new(text).block(
        Block::default()
            .title("Strategy Decision")
            .borders(Borders::ALL),
    );
    frame.render_widget(block, area);
}

fn render_positions_panel(frame: &mut Frame, area: Rect, state: &PaperState) {
    let text = if state.open_positions == 0 {
        vec![
            Line::from(""),
            Line::from(Span::styled(
                "  No active trades",
                Style::default().fg(Color::Gray),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "  Evaluating...",
                Style::default().fg(Color::DarkGray),
            )),
        ]
    } else if let Some(v) = &state.strategy_view {
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
                Span::raw(format!("{:<12}", truncate(&pos.symbol, 12))),
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

        lines
    } else {
        vec![Line::from(Span::styled(
            "Loading...",
            Style::default().fg(Color::Gray),
        ))]
    };

    let title = format!("Positions ({})", state.open_positions);
    let block = Paragraph::new(text).block(Block::default().title(title).borders(Borders::ALL));
    frame.render_widget(block, area);
}

/// Truncate string safely (UTF-8 aware).
fn truncate(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_len.saturating_sub(3)).collect();
        format!("{}...", truncated)
    }
}
