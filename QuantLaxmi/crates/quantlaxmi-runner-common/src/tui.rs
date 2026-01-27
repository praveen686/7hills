//! # TUI Framework
//!
//! Shared terminal UI components for QuantLaxmi runners.
//!
//! ## Phase 17A Extensions
//! - `render_session_panel()` - Session state, duration, active kill-switches
//! - `render_kill_switch_panel()` - Hierarchical view of active switches
//! - `render_override_panel()` - Recent manual overrides with operator audit

use crate::control_view::{
    ExecutionControlView, format_kill_switch_scope, format_session_state, format_transition_reason,
};
use crossterm::{
    event::{self, Event as CEvent, KeyCode},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use quantlaxmi_gates::{ManualOverrideEvent, SessionState};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
};
use std::io::Stdout;
use std::time::Duration;

pub type TuiTerminal = Terminal<CrosstermBackend<Stdout>>;

/// Initialize the TUI terminal
pub fn init_terminal() -> anyhow::Result<TuiTerminal> {
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    Terminal::new(backend).map_err(|e| anyhow::anyhow!("Failed to create terminal: {}", e))
}

/// Cleanup the TUI terminal
pub fn cleanup_terminal(terminal: &mut TuiTerminal) -> anyhow::Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

/// Print headless mode banner
pub fn print_headless_banner(service_name: &str, initial_capital: f64) {
    println!("═══════════════════════════════════════════════════════════════");
    println!("        {} - Headless Mode", service_name);
    println!("═══════════════════════════════════════════════════════════════");
    println!("[HEADLESS] Initial Capital: ${:.2}", initial_capital);
    println!("[HEADLESS] Press Ctrl+C to stop");
    println!("═══════════════════════════════════════════════════════════════");
}

/// Check for quit key press (non-blocking)
pub fn check_quit_key() -> bool {
    if event::poll(Duration::from_millis(50)).unwrap_or(false)
        && let Ok(CEvent::Key(key)) = event::read()
    {
        return key.code == KeyCode::Char('q') || key.code == KeyCode::Esc;
    }
    false
}

/// Render a basic status panel
pub fn render_status_panel(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    equity: f64,
    pnl: f64,
    mode: &str,
) {
    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::Cyan));

    let pnl_color = if pnl >= 0.0 { Color::Green } else { Color::Red };

    let text = format!("Mode: {}\nEquity: ${:.2}\nP&L: ${:.2}", mode, equity, pnl);

    let paragraph = Paragraph::new(text)
        .block(block)
        .style(Style::default().fg(pnl_color));

    frame.render_widget(paragraph, area);
}

/// Render order log panel
pub fn render_order_log(frame: &mut Frame, area: Rect, logs: &[String]) {
    let block = Block::default()
        .title("Order Log")
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::Yellow));

    let items: Vec<ListItem> = logs
        .iter()
        .rev()
        .take(10)
        .map(|log| ListItem::new(log.as_str()))
        .collect();

    let list = List::new(items)
        .block(block)
        .style(Style::default().fg(Color::White));

    frame.render_widget(list, area);
}

// =============================================================================
// Phase 17A: Execution Control Plane Panels
// =============================================================================

/// Map session state to TUI color.
fn session_state_color(state: SessionState) -> Color {
    match state {
        SessionState::Active => Color::Green,
        SessionState::Halted => Color::Red,
        SessionState::ReduceOnly => Color::Yellow,
        SessionState::Draining => Color::Blue,
        SessionState::Terminated => Color::DarkGray,
    }
}

/// Format duration from nanoseconds to human-readable string.
fn format_duration_ns(duration_ns: i64) -> String {
    let secs = duration_ns / 1_000_000_000;
    let mins = secs / 60;
    let hours = mins / 60;

    if hours > 0 {
        format!("{}h {}m", hours, mins % 60)
    } else if mins > 0 {
        format!("{}m {}s", mins, secs % 60)
    } else {
        format!("{}s", secs)
    }
}

/// Render session state panel (Phase 17A).
///
/// Shows:
/// - Current session state with color coding
/// - Time in current state
/// - Active kill-switch count
/// - Last transition reason (if any)
pub fn render_session_panel(frame: &mut Frame, area: Rect, view: &ExecutionControlView) {
    let state = view.session.state;
    let (state_text, _) = format_session_state(state);
    let state_color = session_state_color(state);

    // Calculate time in state
    let now_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
    let duration_ns = now_ns.saturating_sub(view.session.state_since_ts_ns);
    let duration_str = format_duration_ns(duration_ns);

    let kill_switch_count = view.active_kill_switches.len();

    let block = Block::default()
        .title("Session Control")
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::Cyan));

    // Build content lines
    let mut lines = vec![
        Line::from(vec![
            Span::raw("State: "),
            Span::styled(
                state_text,
                Style::default()
                    .fg(state_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!(" ({})", duration_str)),
        ]),
        Line::from(format!("Session: {}", view.session.session_id)),
        Line::from(format!("Transitions: {}", view.session.transition_count)),
        Line::from(format!("Overrides: {}", view.session.override_count)),
    ];

    // Kill-switch indicator
    if kill_switch_count > 0 {
        lines.push(Line::from(vec![Span::styled(
            format!("Kill-switches: {} ACTIVE", kill_switch_count),
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        )]));
    } else {
        lines.push(Line::from(vec![Span::styled(
            "Kill-switches: None",
            Style::default().fg(Color::Green),
        )]));
    }

    // Last transition reason
    if let Some(reason) = &view.session.last_transition_reason {
        let reason_str = format_transition_reason(reason);
        lines.push(Line::from(format!("Last: {}", reason_str)));
    }

    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, area);
}

/// Render kill-switch panel (Phase 17A).
///
/// Shows hierarchical view of active kill-switches:
/// - Global switches (highest priority)
/// - Bucket-level switches
/// - Strategy-level switches
/// - Symbol-level switches
pub fn render_kill_switch_panel(frame: &mut Frame, area: Rect, view: &ExecutionControlView) {
    let block = Block::default()
        .title("Kill-Switches")
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::Red));

    if view.active_kill_switches.is_empty() {
        let paragraph = Paragraph::new("No active kill-switches")
            .block(block)
            .style(Style::default().fg(Color::Green));
        frame.render_widget(paragraph, area);
        return;
    }

    // Group by scope type for hierarchical display
    let mut items: Vec<ListItem> = view
        .active_kill_switches
        .iter()
        .map(|scope| {
            let scope_str = format_kill_switch_scope(scope);
            ListItem::new(Line::from(vec![
                Span::styled("● ", Style::default().fg(Color::Red)),
                Span::raw(scope_str),
            ]))
        })
        .collect();

    // Add count header
    items.insert(
        0,
        ListItem::new(Line::from(vec![Span::styled(
            format!("Active: {}", view.active_kill_switches.len()),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )])),
    );

    let list = List::new(items).block(block);
    frame.render_widget(list, area);
}

/// Format override type for display.
fn format_override_type(event: &ManualOverrideEvent) -> &'static str {
    match event.override_type {
        quantlaxmi_gates::OverrideType::ForceHalt => "HALT",
        quantlaxmi_gates::OverrideType::ForceReduceOnly => "REDUCE_ONLY",
        quantlaxmi_gates::OverrideType::ClearHalt => "CLEAR_HALT",
        quantlaxmi_gates::OverrideType::RestoreFull => "RESTORE",
        quantlaxmi_gates::OverrideType::EmergencyFlatten => "FLATTEN",
        quantlaxmi_gates::OverrideType::CancelAllOrders => "CANCEL_ALL",
        quantlaxmi_gates::OverrideType::AdjustLimit { .. } => "ADJUST_LMT",
    }
}

/// Format timestamp for display (HH:MM:SS).
fn format_timestamp_ns(ts_ns: i64) -> String {
    use chrono::{TimeZone, Utc};
    let secs = ts_ns / 1_000_000_000;
    let nanos = (ts_ns % 1_000_000_000) as u32;
    if let Some(dt) = Utc.timestamp_opt(secs, nanos).single() {
        dt.format("%H:%M:%S").to_string()
    } else {
        "??:??:??".to_string()
    }
}

/// Render override panel (Phase 17A).
///
/// Shows recent manual overrides with:
/// - Timestamp
/// - Operator ID
/// - Override type
/// - Reason (truncated)
pub fn render_override_panel(frame: &mut Frame, area: Rect, view: &ExecutionControlView) {
    let block = Block::default()
        .title("Recent Overrides")
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::Magenta));

    if view.recent_overrides.is_empty() {
        let paragraph = Paragraph::new("No recent overrides")
            .block(block)
            .style(Style::default().fg(Color::DarkGray));
        frame.render_widget(paragraph, area);
        return;
    }

    // Show last 5 overrides (newest first, already sorted)
    let items: Vec<ListItem> = view
        .recent_overrides
        .iter()
        .take(5)
        .map(|event| {
            let time_str = format_timestamp_ns(event.ts_ns);
            let type_str = format_override_type(event);

            // Color based on override type
            let type_color = match event.override_type {
                quantlaxmi_gates::OverrideType::ForceHalt
                | quantlaxmi_gates::OverrideType::EmergencyFlatten => Color::Red,
                quantlaxmi_gates::OverrideType::ClearHalt
                | quantlaxmi_gates::OverrideType::RestoreFull => Color::Green,
                quantlaxmi_gates::OverrideType::ForceReduceOnly
                | quantlaxmi_gates::OverrideType::CancelAllOrders
                | quantlaxmi_gates::OverrideType::AdjustLimit { .. } => Color::Yellow,
            };

            // Truncate reason if too long
            let reason = if event.reason.len() > 20 {
                format!("{}...", &event.reason[..17])
            } else {
                event.reason.clone()
            };

            ListItem::new(Line::from(vec![
                Span::raw(format!("{} ", time_str)),
                Span::styled(format!("{:<12}", type_str), Style::default().fg(type_color)),
                Span::raw(format!(" {} - {}", event.operator_id, reason)),
            ]))
        })
        .collect();

    let list = List::new(items).block(block);
    frame.render_widget(list, area);
}

/// Render combined control view panel (Phase 17A).
///
/// Compact view showing key control plane status in single panel.
/// Useful when screen real estate is limited.
pub fn render_control_summary_panel(frame: &mut Frame, area: Rect, view: &ExecutionControlView) {
    let state = view.session.state;
    let (state_text, _) = format_session_state(state);
    let state_color = session_state_color(state);

    let block = Block::default()
        .title("Control Plane")
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::Cyan));

    let kill_switch_status = if view.active_kill_switches.is_empty() {
        Span::styled("OK", Style::default().fg(Color::Green))
    } else {
        Span::styled(
            format!("{} ACTIVE", view.active_kill_switches.len()),
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        )
    };

    let lines = vec![
        Line::from(vec![
            Span::raw("State: "),
            Span::styled(
                state_text,
                Style::default()
                    .fg(state_color)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![Span::raw("Kill-switches: "), kill_switch_status]),
        Line::from(format!(
            "Transitions: {} | Overrides: {}",
            view.session.transition_count, view.session.override_count
        )),
    ];

    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, area);
}
