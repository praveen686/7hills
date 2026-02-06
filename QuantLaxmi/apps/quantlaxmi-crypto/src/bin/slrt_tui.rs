use std::io;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode},
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

use quantlaxmi_runner_crypto::paper::{
    engine::{PaperEngine, SharedState},
    state::{R3Eligibility, UiSnapshot},
    telemetry::TelemetryBus,
};
// Note: SniperStats is accessed via snap.sniper_stats - no explicit import needed

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ----- runner wiring (in-process for now) -----
    let shared = SharedState::new();
    let (bus, rx) = TelemetryBus::new();

    let engine = PaperEngine::new("BTCUSDT", shared.clone(), bus.clone()).with_log_dir("runs");
    tokio::spawn(async move {
        engine.run().await;
    });

    // ----- TUI -----
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let res = run_tui(&mut terminal, rx).await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    res
}

async fn run_tui<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    mut rx: watch::Receiver<UiSnapshot>,
) -> anyhow::Result<()> {
    let mut last_draw = Instant::now();
    let draw_every = Duration::from_millis(33); // ~30 FPS for smooth updates

    loop {
        // Poll keyboard without blocking UI refresh.
        if event::poll(Duration::from_millis(10))?
            && let Event::Key(k) = event::read()?
        {
            match k.code {
                KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                _ => {}
            }
        }

        // Pull latest snapshot if changed.
        let snap = (*rx.borrow()).clone();

        if last_draw.elapsed() >= draw_every {
            terminal.draw(|f| render(f, &snap))?;
            last_draw = Instant::now();
        }

        // If there are updates, wait for them briefly to keep CPU low.
        let _ = tokio::time::timeout(Duration::from_millis(16), rx.changed()).await;
    }
}

fn render(f: &mut ratatui::Frame<'_>, snap: &UiSnapshot) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // header
            Constraint::Length(8), // market data + trades + position
            Constraint::Length(9), // SLRT metrics (with FTI diagnostics)
            Constraint::Min(5),    // refusal + intent
        ])
        .split(f.area());

    // Header
    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            "QuantLaxmi Crypto — SLRT Paper Trading (Real-Time)",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("   "),
        Span::raw("press "),
        Span::styled("q", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(" to quit"),
    ]))
    .block(Block::default().borders(Borders::ALL).title("Status"));
    f.render_widget(header, chunks[0]);

    // Market Data + Trades + Position (split horizontally)
    let market_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(40),
            Constraint::Percentage(30),
            Constraint::Percentage(30),
        ])
        .split(chunks[1]);

    render_market_data(f, market_chunks[0], snap);
    render_trades(f, market_chunks[1], snap);
    render_position(f, market_chunks[2], snap);

    // SLRT Metrics
    render_slrt_metrics(f, chunks[2], snap);

    // Bottom: Refusal reasons + Proposed + Accepted (this tick) + Last Accepted (historical)
    let lower_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(30), // Refusal
            Constraint::Percentage(23), // Proposed
            Constraint::Percentage(23), // Accepted this tick
            Constraint::Percentage(24), // Historical
        ])
        .split(chunks[3]);

    render_refusal_reasons(f, lower_chunks[0], snap);
    render_proposed_this_tick(f, lower_chunks[1], snap);
    render_accepted_this_tick(f, lower_chunks[2], snap);
    render_last_accepted_historical(f, lower_chunks[3], snap);
}

fn render_market_data(f: &mut ratatui::Frame<'_>, area: ratatui::layout::Rect, snap: &UiSnapshot) {
    let md = &snap.market_data;

    let mid_str = md
        .mid_price
        .map(|p| format!("${:.2}", p))
        .unwrap_or("—".into());
    let bid_str = md
        .best_bid
        .map(|p| format!("${:.2}", p))
        .unwrap_or("—".into());
    let ask_str = md
        .best_ask
        .map(|p| format!("${:.2}", p))
        .unwrap_or("—".into());
    let spread_str = md
        .spread_bps
        .map(|s| format!("{:.2} bps", s))
        .unwrap_or("—".into());
    let imb_str = md
        .imbalance
        .map(|i| format!("{:+.2}", i))
        .unwrap_or("—".into());

    let imb_color = match md.imbalance {
        Some(i) if i > 0.1 => Color::Green,
        Some(i) if i < -0.1 => Color::Red,
        _ => Color::White,
    };

    let lines = vec![
        Line::from(vec![
            Span::styled(
                format!("{:<12}", md.symbol),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("Mid: ", Style::default().fg(Color::Gray)),
            Span::styled(
                mid_str,
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled("Spread: ", Style::default().fg(Color::Gray)),
            Span::styled(spread_str, Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("Bid: ", Style::default().fg(Color::Green)),
            Span::styled(
                format!("{:<14}", bid_str),
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("Ask: ", Style::default().fg(Color::Red)),
            Span::styled(
                ask_str,
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("BidQty: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!(
                    "{:<10}",
                    md.bid_qty
                        .map(|q| format!("{:.4}", q))
                        .unwrap_or("—".into())
                ),
                Style::default().fg(Color::Green),
            ),
            Span::styled("AskQty: ", Style::default().fg(Color::Gray)),
            Span::styled(
                md.ask_qty
                    .map(|q| format!("{:.4}", q))
                    .unwrap_or("—".into()),
                Style::default().fg(Color::Red),
            ),
        ]),
        Line::from(vec![
            Span::styled("Imbalance: ", Style::default().fg(Color::Gray)),
            Span::styled(
                imb_str,
                Style::default().fg(imb_color).add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled("Tick: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", md.tick_count),
                Style::default().fg(Color::Cyan),
            ),
        ]),
    ];

    let block =
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title("Order Book"));
    f.render_widget(block, area);
}

fn render_trades(f: &mut ratatui::Frame<'_>, area: ratatui::layout::Rect, snap: &UiSnapshot) {
    let md = &snap.market_data;

    // Calculate how many trades we can show (area height - 2 for borders)
    let max_lines = (area.height as usize).saturating_sub(2);

    // Build trade tape lines (most recent first)
    let mut lines: Vec<Line> = md
        .trade_tape
        .recent()
        .take(max_lines)
        .map(|t| {
            let color = if t.is_buy { Color::Green } else { Color::Red };
            let side = if t.is_buy { "B" } else { "S" };
            Line::from(vec![
                Span::styled(
                    side,
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                ),
                Span::raw(" "),
                Span::styled(format!("{:.2}", t.price), Style::default().fg(color)),
                Span::raw(" "),
                Span::styled(format!("{:.4}", t.qty), Style::default().fg(Color::White)),
            ])
        })
        .collect();

    // If no trades yet, show placeholder
    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "waiting...",
            Style::default().fg(Color::Gray),
        )));
    }

    let title = format!("Tape ({:.0}/s)", md.trades_per_sec);
    let block = Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(title));
    f.render_widget(block, area);
}

fn render_position(f: &mut ratatui::Frame<'_>, area: ratatui::layout::Rect, snap: &UiSnapshot) {
    let pos = &snap.position;

    // Position size and direction
    let (pos_str, pos_color) = if pos.size.abs() < 1e-9 {
        ("FLAT".to_string(), Color::Gray)
    } else if pos.size > 0.0 {
        (format!("+{:.6}", pos.size), Color::Green)
    } else {
        (format!("{:.6}", pos.size), Color::Red)
    };

    // PnL colors
    let total_pnl = pos.total_pnl();
    let total_color = if total_pnl >= 0.0 {
        Color::Green
    } else {
        Color::Red
    };

    // Entry price
    let entry_str = if pos.size.abs() > 1e-9 {
        format!("${:.2}", pos.avg_entry_price)
    } else {
        "—".to_string()
    };

    let lines = vec![
        Line::from(vec![
            Span::styled("Pos: ", Style::default().fg(Color::Gray)),
            Span::styled(
                pos_str,
                Style::default().fg(pos_color).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" @ "),
            Span::styled(entry_str, Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::styled("Gross: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("${:+.2}", pos.realized_pnl + pos.unrealized_pnl),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("Fees:  ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("-${:.2}", pos.total_fees),
                Style::default().fg(Color::Red),
            ),
            Span::styled(
                format!(" ({:.1}%)", pos.fee_rate * 100.0),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(vec![
            Span::styled("Net:   ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("${:+.2}", total_pnl),
                Style::default()
                    .fg(total_color)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("Fills: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", pos.fill_count),
                Style::default().fg(Color::Cyan),
            ),
        ]),
    ];

    let block =
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title("Position/PnL"));
    f.render_widget(block, area);
}

fn render_slrt_metrics(f: &mut ratatui::Frame<'_>, area: ratatui::layout::Rect, snap: &UiSnapshot) {
    let elig_style = match snap.last_decision.eligibility {
        R3Eligibility::Eligible => Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD),
        R3Eligibility::Refused => Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        R3Eligibility::Unknown => Style::default().fg(Color::Gray),
    };

    let metrics = &snap.last_decision.metrics;
    let sniper = &snap.sniper_stats;

    // Sniper rate limit status
    let cooldown_str = match sniper.secs_since_last {
        Some(s) if !sniper.cooldown_ok => format!("{}s/{}", s, sniper.cooldown_secs),
        Some(s) => format!("{}s OK", s),
        None => "—".to_string(),
    };
    let cooldown_color = if sniper.cooldown_ok {
        Color::Green
    } else {
        Color::Yellow
    };

    let rate_color = if sniper.entries_last_hour < sniper.max_per_hour {
        Color::Green
    } else {
        Color::Red
    };

    // Persistence colors (0.3 threshold for R3)
    let fti_persist_val = metrics.fti_persist.unwrap_or(0.0);
    let tox_persist_val = metrics.toxicity_persist.unwrap_or(0.0);
    let fti_persist_color = if fti_persist_val >= 0.3 {
        Color::Green
    } else {
        Color::White
    };
    let tox_persist_color = if tox_persist_val >= 0.3 {
        Color::Green
    } else {
        Color::White
    };

    // R3 cause - only meaningful when regime is R3
    let is_r3_regime = sniper.regime.as_str() == "R3";
    let r3_cause_display = if is_r3_regime {
        sniper.r3_cause.clone()
    } else {
        "N/A".to_string()
    };
    let r3_cause_color = if !is_r3_regime {
        Color::DarkGray
    } else {
        match sniper.r3_cause.as_str() {
            "FTI" => Color::Green,
            "TOX" => Color::Yellow,
            "BOTH" => Color::Cyan,
            _ => Color::DarkGray,
        }
    };

    // Regime color
    let regime_color = match sniper.regime.as_str() {
        "R3" => Color::Green,
        "R2" => Color::Yellow,
        "R1" => Color::White,
        _ => Color::DarkGray,
    };

    let lines = vec![
        Line::from(vec![
            Span::styled("Trade: ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{}", snap.last_decision.eligibility), elig_style),
            Span::raw("  "),
            Span::styled("Regime: ", Style::default().fg(Color::Gray)),
            Span::styled(
                &sniper.regime,
                Style::default()
                    .fg(regime_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled("R3 cause: ", Style::default().fg(Color::Gray)),
            Span::styled(&r3_cause_display, Style::default().fg(r3_cause_color)),
            Span::raw("  "),
            Span::styled(
                format_time(snap.last_decision.decided_at_unix_ms),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(vec![
            Span::styled("confidence  ", Style::default().fg(Color::Yellow)),
            Span::styled(
                fmt_metric(metrics.confidence),
                Style::default().fg(Color::White),
            ),
            Span::raw("    "),
            Span::styled("d_perp      ", Style::default().fg(Color::Yellow)),
            Span::styled(
                fmt_metric(metrics.d_perp),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("fragility   ", Style::default().fg(Color::Yellow)),
            Span::styled(
                fmt_metric(metrics.fragility),
                Style::default().fg(Color::White),
            ),
            Span::raw("    "),
            Span::styled("toxicity    ", Style::default().fg(Color::Yellow)),
            Span::styled(
                fmt_toxicity(metrics.toxicity),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("fti_persist ", Style::default().fg(Color::Yellow)),
            Span::styled(
                format!("{:.2}", fti_persist_val),
                Style::default().fg(fti_persist_color),
            ),
            Span::raw("  "),
            Span::styled("tox_persist ", Style::default().fg(Color::Yellow)),
            Span::styled(
                format!("{:.2}", tox_persist_val),
                Style::default().fg(tox_persist_color),
            ),
        ]),
        // FTI diagnostics line
        {
            let fti_level = metrics.fti_level.unwrap_or(0.0);
            let fti_thresh = metrics.fti_thresh.unwrap_or(1.0);
            let fti_elevated = metrics.fti_elevated.unwrap_or(false);
            let fti_calibrated = metrics.fti_calibrated.unwrap_or(false);

            let elev_str = if fti_elevated { "Y" } else { "N" };
            let elev_color = if fti_elevated {
                Color::Green
            } else {
                Color::Red
            };
            let cal_str = if fti_calibrated { "cal" } else { "def" };
            let cal_color = if fti_calibrated {
                Color::Cyan
            } else {
                Color::Yellow
            };

            Line::from(vec![
                Span::styled("fti_level   ", Style::default().fg(Color::Yellow)),
                Span::styled(
                    format!("{:.2}", fti_level),
                    Style::default().fg(Color::White),
                ),
                Span::raw("  "),
                Span::styled("thresh ", Style::default().fg(Color::Gray)),
                Span::styled(format!("{:.2}", fti_thresh), Style::default().fg(cal_color)),
                Span::styled(
                    format!(" ({})", cal_str),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::raw("  "),
                Span::styled("elev ", Style::default().fg(Color::Gray)),
                Span::styled(
                    elev_str,
                    Style::default().fg(elev_color).add_modifier(Modifier::BOLD),
                ),
            ])
        },
        Line::from(vec![
            Span::styled(
                "Sniper: ",
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("{}/{}", sniper.entries_last_hour, sniper.max_per_hour),
                Style::default().fg(rate_color),
            ),
            Span::styled("/hr  ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}/{}", sniper.session_entries, sniper.max_per_session),
                Style::default().fg(Color::Cyan),
            ),
            Span::styled("/ses  ", Style::default().fg(Color::Gray)),
            Span::styled("CD: ", Style::default().fg(Color::Gray)),
            Span::styled(cooldown_str, Style::default().fg(cooldown_color)),
        ]),
    ];

    let block =
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title("SLRT Metrics"));
    f.render_widget(block, area);
}

fn render_refusal_reasons(
    f: &mut ratatui::Frame<'_>,
    area: ratatui::layout::Rect,
    snap: &UiSnapshot,
) {
    let refusal_text = if snap.last_decision.refusal_reasons.is_empty() {
        vec![Line::from(Span::styled(
            "— none —",
            Style::default().fg(Color::Gray),
        ))]
    } else {
        snap.last_decision
            .refusal_reasons
            .iter()
            .map(|r| {
                Line::from(vec![
                    Span::styled(
                        r.code,
                        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                    ),
                    Span::raw("  "),
                    Span::styled(r.detail.clone(), Style::default().fg(Color::White)),
                ])
            })
            .collect()
    };

    let block = Paragraph::new(refusal_text)
        .wrap(Wrap { trim: false })
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Refusal Reasons"),
        );
    f.render_widget(block, area);
}

fn render_proposed_this_tick(
    f: &mut ratatui::Frame<'_>,
    area: ratatui::layout::Rect,
    snap: &UiSnapshot,
) {
    let intent_lines = match &snap.proposed_this_tick {
        None => vec![Line::from(Span::styled(
            "— none —",
            Style::default().fg(Color::Gray),
        ))],
        Some(i) => {
            let side_color = match i.side {
                quantlaxmi_runner_crypto::paper::intent::Side::Buy => Color::Green,
                quantlaxmi_runner_crypto::paper::intent::Side::Sell => Color::Red,
            };
            vec![
                Line::from(vec![
                    Span::styled(
                        format!("{}", i.side),
                        Style::default().fg(side_color).add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(" "),
                    Span::styled(format!("{:.4}", i.qty), Style::default().fg(Color::White)),
                ]),
                Line::from(vec![Span::styled(
                    i.intent_id.clone(),
                    Style::default().fg(Color::DarkGray),
                )]),
            ]
        }
    };

    let block = Paragraph::new(intent_lines)
        .wrap(Wrap { trim: true })
        .block(Block::default().borders(Borders::ALL).title("Proposed"));
    f.render_widget(block, area);
}

fn render_accepted_this_tick(
    f: &mut ratatui::Frame<'_>,
    area: ratatui::layout::Rect,
    snap: &UiSnapshot,
) {
    let intent_lines = match &snap.accepted_this_tick {
        None => vec![Line::from(Span::styled(
            "— none —",
            Style::default().fg(Color::Gray),
        ))],
        Some(i) => {
            let side_color = match i.side {
                quantlaxmi_runner_crypto::paper::intent::Side::Buy => Color::Green,
                quantlaxmi_runner_crypto::paper::intent::Side::Sell => Color::Red,
            };
            vec![
                Line::from(vec![
                    Span::styled(
                        format!("{}", i.side),
                        Style::default().fg(side_color).add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(" "),
                    Span::styled(format!("{:.4}", i.qty), Style::default().fg(Color::White)),
                ]),
                Line::from(vec![Span::styled(
                    i.intent_id.clone(),
                    Style::default().fg(Color::DarkGray),
                )]),
            ]
        }
    };

    // Green title when accepted this tick, otherwise gray
    let title_style = if snap.accepted_this_tick.is_some() {
        Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let block = Paragraph::new(intent_lines)
        .wrap(Wrap { trim: true })
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("This Tick")
                .title_style(title_style),
        );
    f.render_widget(block, area);
}

fn render_last_accepted_historical(
    f: &mut ratatui::Frame<'_>,
    area: ratatui::layout::Rect,
    snap: &UiSnapshot,
) {
    let intent_lines = match &snap.last_accepted_historical {
        None => vec![Line::from(Span::styled(
            "— none —",
            Style::default().fg(Color::Gray),
        ))],
        Some(i) => {
            let side_color = match i.side {
                quantlaxmi_runner_crypto::paper::intent::Side::Buy => Color::Green,
                quantlaxmi_runner_crypto::paper::intent::Side::Sell => Color::Red,
            };
            vec![
                Line::from(vec![
                    Span::styled(format!("{}", i.side), Style::default().fg(side_color)),
                    Span::raw(" "),
                    Span::styled(format!("{:.4}", i.qty), Style::default().fg(Color::White)),
                ]),
                Line::from(vec![Span::styled(
                    i.intent_id.clone(),
                    Style::default().fg(Color::DarkGray),
                )]),
            ]
        }
    };

    let block = Paragraph::new(intent_lines)
        .wrap(Wrap { trim: true })
        .block(Block::default().borders(Borders::ALL).title("Last Trade"));
    f.render_widget(block, area);
}

fn fmt_metric(v: Option<f64>) -> String {
    match v {
        None => "—".to_string(),
        Some(x) => format!("{:.4}", x),
    }
}

fn fmt_toxicity(v: Option<f64>) -> String {
    match v {
        None => "—".to_string(),
        Some(x) if x >= 1.0 => format!("{:.2}*", x), // Clamped indicator
        Some(x) => format!("{:.4}", x),
    }
}

fn format_time(unix_ms: u64) -> String {
    if unix_ms == 0 {
        return "—".to_string();
    }
    let secs = (unix_ms / 1000) % 86400;
    let hours = secs / 3600;
    let mins = (secs % 3600) / 60;
    let secs = secs % 60;
    let millis = unix_ms % 1000;
    format!("{:02}:{:02}:{:02}.{:03}", hours, mins, secs, millis)
}
