//! # TUI Framework
//!
//! Shared terminal UI components for QuantLaxmi runners.

use ratatui::{
    backend::CrosstermBackend,
    widgets::{Block, Borders, List, ListItem, Paragraph, Table, Row, Cell},
    layout::{Layout, Constraint, Direction, Rect},
    Terminal,
    style::{Style, Color, Modifier},
    Frame,
};
use crossterm::{
    event::{self, Event as CEvent, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use std::io::Stdout;
use std::time::Duration;
use tracing::info;

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
    if event::poll(Duration::from_millis(50)).unwrap_or(false) {
        if let Ok(CEvent::Key(key)) = event::read() {
            return key.code == KeyCode::Char('q') || key.code == KeyCode::Esc;
        }
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

    let text = format!(
        "Mode: {}\nEquity: ${:.2}\nP&L: ${:.2}",
        mode, equity, pnl
    );

    let paragraph = Paragraph::new(text)
        .block(block)
        .style(Style::default().fg(Color::White));

    frame.render_widget(paragraph, area);
}

/// Render order log panel
pub fn render_order_log(frame: &mut Frame, area: Rect, logs: &[String]) {
    let block = Block::default()
        .title("Order Log")
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::Yellow));

    let items: Vec<ListItem> = logs.iter()
        .rev()
        .take(10)
        .map(|log| ListItem::new(log.as_str()))
        .collect();

    let list = List::new(items)
        .block(block)
        .style(Style::default().fg(Color::White));

    frame.render_widget(list, area);
}
