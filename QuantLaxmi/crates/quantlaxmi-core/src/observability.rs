//! # System Observability & Telemetry
//!
//! Integration with OpenTelemetry and Prometheus for production monitoring.
//!
//! ## Description
//! Provides centralized initialization for the system's observability stack:
//! - **Metrics**: Prometheus HTTP exporter for hardware and business logic metrics.
//! - **Distributed Tracing**: OpenTelemetry integration for service-level profiling.
//! - **Structured Logging**: Integration with `tracing` for hierarchical log streams.
//!
//! ## Logging Architecture
//! - **stdout**: WARN only by default (prevents log amplification in agent/transcript contexts)
//! - **file**: INFO for quantlaxmi crates, WARN for deps (daily rotation)
//! - **RUST_LOG**: Honored for file logs only; stdout always bounded to WARN
//!
//! ## References
//! - IEEE Std 1016-2009: Software Design Descriptions
//! - OpenTelemetry documentation
//! - Prometheus Monitoring Guide

use std::{fs, net::SocketAddr, path::Path};

use metrics_exporter_prometheus::PrometheusBuilder;
use opentelemetry::global;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::{Config, Sampler};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{
    fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer,
};

/// Guards that must be held for the lifetime of the process.
/// Dropping this will cause buffered logs to be lost.
pub struct TracingGuards {
    _file_guard: WorkerGuard,
}

/// Initializes the Prometheus metrics exporter on the specified socket.
///
/// # Parameters
/// * `addr` - The network address to bind the HTTP metrics endpoint to.
pub fn init_metrics(addr: SocketAddr) {
    let builder = PrometheusBuilder::new().with_http_listener(addr);

    match builder.install() {
        Ok(_) => {
            tracing::info!("Prometheus metrics exporter started on {}", addr);
        }
        Err(e) => {
            tracing::warn!(
                "Failed to start Prometheus metrics exporter on {}: {} (continuing without metrics)",
                addr,
                e
            );
        }
    }
}

/// Ensures the logs directory exists.
fn ensure_logs_dir() {
    let dir = Path::new("logs");
    if !dir.exists() {
        // Best effort: if this fails, we still want stdout logs to work.
        let _ = fs::create_dir_all(dir);
    }
}

/// Initializes tracing with bounded stdout + rotated file logs + OpenTelemetry.
///
/// # Logging Policy
/// - **stdout**: Always WARN only (hard-coded, ignores RUST_LOG)
///   - Prevents log amplification when run under agents/transcripts
///   - Compact format, no timestamps (human-friendly)
/// - **file**: INFO for quantlaxmi, WARN for deps (honors RUST_LOG override)
///   - Daily rotation to `logs/{service_name}.log`
///   - Non-blocking to minimize capture latency impact
///   - Full metadata (timestamps, thread IDs, targets)
/// - **OpenTelemetry**: Preserved for distributed tracing
///
/// # Returns
/// `TracingGuards` - Must be held for the lifetime of the process or logs may be lost.
///
/// # Parameters
/// * `service_name` - Identifier for the current executing binary/context.
pub fn init_tracing(service_name: &str) -> TracingGuards {
    ensure_logs_dir();

    // --- File Appender (non-blocking, daily rotation) ---
    let file_appender =
        tracing_appender::rolling::daily("logs", format!("{}.log", service_name));
    let (file_writer, file_guard) = tracing_appender::non_blocking(file_appender);

    // --- Filter Definitions ---
    // stdout: ALWAYS WARN only (hard guarantee, ignores RUST_LOG)
    // This prevents log amplification catastrophe in agent/transcript contexts.
    let stdout_filter = EnvFilter::new("warn");

    // file: Default to INFO for our crates, WARN for noisy deps.
    // Honors RUST_LOG if set (for debugging), otherwise safe default.
    let default_file_filter = "quantlaxmi=info,warn";
    let file_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(default_file_filter));

    // --- stdout Layer: Bounded, Human-Friendly ---
    let stdout_layer = fmt::layer()
        .with_target(false)
        .with_level(true)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_ansi(true)
        .compact()
        .with_filter(stdout_filter);

    // --- File Layer: Detailed, Non-Blocking ---
    let file_layer = fmt::layer()
        .with_writer(file_writer)
        .with_ansi(false)
        .with_target(true)
        .with_level(true)
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_filter(file_filter);

    // --- OpenTelemetry Layer (preserved from original) ---
    global::set_text_map_propagator(TraceContextPropagator::new());

    let provider = opentelemetry_sdk::trace::TracerProvider::builder()
        .with_config(Config::default().with_sampler(Sampler::AlwaysOn))
        .build();

    global::set_tracer_provider(provider.clone());

    let tracer =
        opentelemetry::trace::TracerProvider::tracer(&provider, service_name.to_string());
    let telemetry = tracing_opentelemetry::layer().with_tracer(tracer);

    // --- Compose and Initialize ---
    tracing_subscriber::registry()
        .with(stdout_layer)
        .with(file_layer)
        .with(telemetry)
        .init();

    tracing::info!(
        "Observability initialized for service: {} (stdout=WARN, file=logs/{}.log)",
        service_name,
        service_name
    );

    TracingGuards {
        _file_guard: file_guard,
    }
}
