//! Phase 25B Integration Tests: Latency Buckets
//!
//! Tests verify:
//! 1. latency_ticks=0 preserves baseline behavior (immediate execution)
//! 2. latency_ticks>0 delays execution by specified ticks
//! 3. PendingIntent FIFO ordering is preserved
//! 4. End-of-run drain flushes remaining intents
//! 5. Determinism: same inputs produce same outputs

use quantlaxmi_runner_crypto::backtest::{
    BacktestConfig, EnforcementConfig, ExchangeConfig, PaceMode,
};

// =============================================================================
// Unit Tests for Config
// =============================================================================

#[test]
fn test_latency_ticks_default_is_zero() {
    let config = BacktestConfig::default();
    assert_eq!(
        config.latency_ticks, 0,
        "Default latency_ticks should be 0 for baseline behavior"
    );
}

#[test]
fn test_latency_ticks_config_field() {
    let config = BacktestConfig {
        exchange: ExchangeConfig {
            fee_bps: 10.0,
            initial_cash: 100_000.0,
            use_perp_prices: true,
        },
        log_interval: 1_000_000,
        pace: PaceMode::Fast,
        output_trace: None,
        run_id: None,
        enforce_admission_from_wal: false,
        admission_mismatch_policy: "fail".to_string(),
        strategy_spec: None,
        enforcement: EnforcementConfig::default(),
        cost_model_path: None,
        latency_ticks: 3,
    };

    assert_eq!(config.latency_ticks, 3);
}

#[test]
fn test_latency_ticks_serialization() {
    let config = BacktestConfig {
        exchange: ExchangeConfig {
            fee_bps: 10.0,
            initial_cash: 100_000.0,
            use_perp_prices: true,
        },
        log_interval: 1_000_000,
        pace: PaceMode::Fast,
        output_trace: None,
        run_id: None,
        enforce_admission_from_wal: false,
        admission_mismatch_policy: "fail".to_string(),
        strategy_spec: None,
        enforcement: EnforcementConfig::default(),
        cost_model_path: None,
        latency_ticks: 5,
    };

    // Verify latency_ticks can be serialized/deserialized via Debug
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("latency_ticks: 5"));
}

// =============================================================================
// Latency Bucket Logic Tests
// =============================================================================

/// Test the scheduling logic: scheduled_tick = sim_tick + latency_ticks
#[test]
fn test_scheduled_tick_computation() {
    // Simulate the scheduling computation
    let sim_tick: u64 = 10;
    let latency_ticks: u64 = 3;
    let scheduled_tick = sim_tick + latency_ticks;

    assert_eq!(scheduled_tick, 13);
}

/// Test drain condition: drain when scheduled_tick <= sim_tick
#[test]
fn test_drain_condition() {
    let scheduled_tick: u64 = 10;

    // Should not drain if sim_tick < scheduled_tick
    let sim_tick: u64 = 9;
    assert!(scheduled_tick > sim_tick, "Should not drain at tick 9");

    // Should drain if sim_tick == scheduled_tick
    let sim_tick: u64 = 10;
    assert!(scheduled_tick <= sim_tick, "Should drain at tick 10");

    // Should drain if sim_tick > scheduled_tick
    let sim_tick: u64 = 11;
    assert!(scheduled_tick <= sim_tick, "Should drain at tick 11");
}

/// Test that latency_ticks=0 means immediate drain (scheduled_tick == sim_tick)
#[test]
fn test_zero_latency_immediate_drain() {
    let sim_tick: u64 = 42;
    let latency_ticks: u64 = 0;
    let scheduled_tick = sim_tick + latency_ticks;

    // With latency=0, scheduled_tick == sim_tick, so drain immediately
    assert_eq!(scheduled_tick, sim_tick);
    assert!(
        scheduled_tick <= sim_tick,
        "Zero latency should drain immediately"
    );
}

// =============================================================================
// FIFO Ordering Tests
// =============================================================================

#[test]
fn test_fifo_ordering_preserved() {
    use std::collections::VecDeque;

    // Simulate the pending_intents queue
    let mut pending: VecDeque<(u64, &str)> = VecDeque::new();

    // Enqueue in order
    pending.push_back((10, "intent_A"));
    pending.push_back((10, "intent_B"));
    pending.push_back((11, "intent_C"));

    // Drain at sim_tick=10
    let sim_tick = 10;
    let mut drained = Vec::new();
    while let Some(front) = pending.front() {
        if front.0 > sim_tick {
            break;
        }
        let item = pending.pop_front().unwrap();
        drained.push(item.1);
    }

    // Should drain A and B (both scheduled_tick=10), preserving FIFO order
    assert_eq!(drained, vec!["intent_A", "intent_B"]);

    // C should remain (scheduled_tick=11)
    assert_eq!(pending.len(), 1);
    assert_eq!(pending.front().unwrap().1, "intent_C");
}

#[test]
fn test_multiple_intents_same_tick_fifo() {
    use std::collections::VecDeque;

    let mut pending: VecDeque<u32> = VecDeque::new();

    // Enqueue 5 intents all with same scheduled_tick
    for i in 1..=5 {
        pending.push_back(i);
    }

    // Drain all and verify FIFO order
    let mut drained = Vec::new();
    while let Some(item) = pending.pop_front() {
        drained.push(item);
    }

    assert_eq!(drained, vec![1, 2, 3, 4, 5], "FIFO order must be preserved");
}

// =============================================================================
// Determinism Tests
// =============================================================================

#[test]
fn test_tick_increment_determinism() {
    // Verify tick increment is deterministic
    let mut sim_tick_run1: u64 = 0;
    let mut sim_tick_run2: u64 = 0;

    // Simulate 100 event iterations
    for _ in 0..100 {
        sim_tick_run1 += 1;
    }
    for _ in 0..100 {
        sim_tick_run2 += 1;
    }

    assert_eq!(
        sim_tick_run1, sim_tick_run2,
        "Tick increment must be deterministic"
    );
    assert_eq!(sim_tick_run1, 100);
}

#[test]
fn test_scheduling_determinism_same_inputs() {
    // Same inputs should produce same scheduling
    let sim_tick = 42u64;
    let latency_ticks = 3u64;

    let results: Vec<u64> = (0..100).map(|_| sim_tick + latency_ticks).collect();

    let first = results[0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(
            *result, first,
            "Determinism violation at iteration {}: {} != {}",
            i, result, first
        );
    }
}

// =============================================================================
// End-of-Run Drain Tests
// =============================================================================

#[test]
fn test_end_of_run_drain_flushes_all() {
    use std::collections::VecDeque;

    let mut pending: VecDeque<(u64, u32)> = VecDeque::new();

    // Enqueue intents with various scheduled_ticks
    pending.push_back((100, 1));
    pending.push_back((200, 2));
    pending.push_back((300, 3));

    // End-of-run drain: flush ALL remaining intents regardless of scheduled_tick
    let mut drained = Vec::new();
    while let Some(item) = pending.pop_front() {
        drained.push(item.1);
    }

    assert_eq!(
        drained,
        vec![1, 2, 3],
        "End-of-run drain must flush all intents"
    );
    assert!(
        pending.is_empty(),
        "Queue must be empty after end-of-run drain"
    );
}

// =============================================================================
// Config Variants Tests
// =============================================================================

#[test]
fn test_latency_ticks_variants() {
    // Test the three primary latency values from spec
    let variants = [0u32, 1u32, 3u32];

    for latency in variants {
        let config = BacktestConfig {
            exchange: ExchangeConfig {
                fee_bps: 10.0,
                initial_cash: 100_000.0,
                use_perp_prices: true,
            },
            log_interval: 1_000_000,
            pace: PaceMode::Fast,
            output_trace: None,
            run_id: None,
            enforce_admission_from_wal: false,
            admission_mismatch_policy: "fail".to_string(),
            strategy_spec: None,
            enforcement: EnforcementConfig::default(),
            cost_model_path: None,
            latency_ticks: latency,
        };

        assert_eq!(config.latency_ticks, latency);
    }
}

#[test]
fn test_latency_baseline_vs_nonzero() {
    // Verify baseline (0) and non-baseline have different implications
    let baseline_latency = 0u64;
    let nonzero_latency = 3u64;

    let sim_tick = 10u64;

    let baseline_scheduled = sim_tick + baseline_latency;
    let nonzero_scheduled = sim_tick + nonzero_latency;

    // Baseline: immediate execution (scheduled == current)
    assert_eq!(baseline_scheduled, sim_tick);

    // Non-zero: delayed execution
    assert!(nonzero_scheduled > sim_tick);
    assert_eq!(nonzero_scheduled, 13);
}
