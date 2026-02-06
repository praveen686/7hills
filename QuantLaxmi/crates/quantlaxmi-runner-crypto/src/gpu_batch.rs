//! GPU Batch Scoring Integration
//!
//! Emits FeatureBatch artifacts and invokes the GPU scorer subprocess.
//! Designed for zero-touch integration with existing tournament runners.

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use ndarray_npy::{read_npy, write_npy};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::process::Command;

use crate::backtest::BacktestMetricsV1;

/// Feature row for GPU batch scoring.
#[derive(Clone, Debug)]
pub struct FeatureRow {
    pub refuse: bool,
    pub x: Vec<f32>,             // length F
    pub config_hash_hex: String, // stable id for diagnostics
}

/// Score row returned from GPU scorer.
#[derive(Clone, Debug)]
pub struct ScoreRow {
    pub refuse: bool,
    pub score: f32,
    pub config_hash_hex: String,
}

/// Feature batch metadata (hashable, auditable).
#[derive(Serialize)]
struct FeatureBatchMeta<'a> {
    schema: &'a str,
    r: usize,
    f: usize,
    x_sha256: String,
    scorer_cmd: &'a str,
    note: &'a str,
}

/// Extract feature vector from BacktestMetricsV1.
///
/// Returns a fixed-size Vec of f32 values.
/// No defaults, no padding. NaN/Inf should trigger refusal upstream.
pub fn metrics_to_feature_vec(m: &BacktestMetricsV1) -> Vec<f32> {
    vec![
        m.total_trades as f32,
        m.winning_trades as f32,
        m.losing_trades as f32,
        m.win_rate as f32,
        m.gross_profit as f32,
        m.gross_loss as f32,
        m.net_pnl as f32,
        m.profit_factor as f32,
        m.expectancy as f32,
        m.avg_win as f32,
        m.avg_loss as f32,
        m.avg_win_loss_ratio as f32,
        m.largest_win as f32,
        m.largest_loss as f32,
        m.max_drawdown as f32,
        m.max_drawdown_pct as f32,
        m.sharpe_ratio as f32,
        m.sortino_ratio as f32,
        m.avg_trade_duration_secs as f32,
    ]
}

/// Check if any metric is NaN or Inf (should trigger refusal).
pub fn has_invalid_metrics(m: &BacktestMetricsV1) -> bool {
    let vals = [
        m.win_rate,
        m.gross_profit,
        m.gross_loss,
        m.net_pnl,
        m.profit_factor,
        m.expectancy,
        m.avg_win,
        m.avg_loss,
        m.avg_win_loss_ratio,
        m.largest_win,
        m.largest_loss,
        m.max_drawdown,
        m.max_drawdown_pct,
        m.sharpe_ratio,
        m.sortino_ratio,
        m.avg_trade_duration_secs,
    ];
    vals.iter().any(|v| !v.is_finite())
}

/// Write X.npy + refuse.npy + meta.json into `feature_dir`.
pub fn emit_feature_batch(feature_dir: &Path, rows: &[FeatureRow]) -> Result<()> {
    std::fs::create_dir_all(feature_dir)
        .with_context(|| format!("create_dir_all {}", feature_dir.display()))?;

    let r = rows.len();
    anyhow::ensure!(r > 0, "emit_feature_batch called with empty rows");

    let f = rows[0].x.len();
    anyhow::ensure!(f > 0, "feature vector length F must be > 0");

    // Ensure fixed F (no silent padding)
    for (i, row) in rows.iter().enumerate() {
        anyhow::ensure!(
            row.x.len() == f,
            "row {} has F={} but expected F={}",
            i,
            row.x.len(),
            f
        );
    }

    let mut x = Array2::<f32>::zeros((r, f));
    let mut refuse = Array1::<u8>::zeros(r);

    for (i, row) in rows.iter().enumerate() {
        refuse[i] = if row.refuse { 1 } else { 0 };
        for j in 0..f {
            x[(i, j)] = row.x[j];
        }
    }

    // Hash the raw matrix bytes (canonical C-order bytes)
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut hasher = Sha256::new();
    hasher.update(&x_bytes);
    let x_sha256 = hex::encode(hasher.finalize());

    write_npy(feature_dir.join("X.npy"), &x).with_context(|| "write X.npy")?;
    write_npy(feature_dir.join("refuse.npy"), &refuse).with_context(|| "write refuse.npy")?;

    // Keep the command string stable for provenance
    let scorer_cmd = "source $HOME/venvs/pt/bin/activate && python $HOME/bin/gpu_scorer.py";

    let meta = FeatureBatchMeta {
        schema: "feature_batch_v0",
        r,
        f,
        x_sha256,
        scorer_cmd,
        note: "QuantLaxmi tournament GPU scoring batch",
    };

    let meta_json = serde_json::to_vec_pretty(&meta)?;
    std::fs::write(feature_dir.join("meta.json"), meta_json).with_context(|| "write meta.json")?;

    Ok(())
}

/// Minimal shell escaping for paths (spaces unlikely, but do it anyway)
fn shell_escape(p: &Path) -> String {
    // Wrap in single quotes, escape embedded single quotes: ' -> '\'' (bash)
    let s = p.display().to_string();
    let escaped = s.replace('\'', r#"'\''"#);
    format!("'{}'", escaped)
}

/// Invoke the GPU scorer to produce scores.npy into `score_dir`.
pub fn run_gpu_scorer(feature_dir: &Path, score_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(score_dir)
        .with_context(|| format!("create_dir_all {}", score_dir.display()))?;

    // Use bash -lc so `source` works.
    let cmd = format!(
        "source $HOME/venvs/pt/bin/activate && python $HOME/bin/gpu_scorer.py {} {}",
        shell_escape(feature_dir),
        shell_escape(score_dir),
    );

    let output = Command::new("bash")
        .arg("-lc")
        .arg(&cmd)
        .output()
        .with_context(|| "failed to spawn gpu scorer")?;

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "gpu_scorer failed: status={:?}\n--- stdout ---\n{}\n--- stderr ---\n{}",
            output.status.code(),
            stdout,
            stderr
        );
    }
    Ok(())
}

/// Read scores.npy and return f32 scores aligned with input rows.
pub fn read_scores(score_dir: &Path) -> Result<Vec<f32>> {
    let scores: Array1<f32> =
        read_npy(score_dir.join("scores.npy")).with_context(|| "read scores.npy")?;
    Ok(scores.to_vec())
}

/// End-to-end helper: emits batch, scores it, and returns ScoreRows aligned with input rows.
pub fn score_rows_with_gpu(work_dir: &Path, rows: &[FeatureRow]) -> Result<Vec<ScoreRow>> {
    let feature_dir = work_dir.join("feature_batch");
    let score_dir = work_dir.join("score_batch");

    emit_feature_batch(&feature_dir, rows)?;
    run_gpu_scorer(&feature_dir, &score_dir)?;
    let scores = read_scores(&score_dir)?;

    anyhow::ensure!(
        scores.len() == rows.len(),
        "scores length {} != rows length {}",
        scores.len(),
        rows.len()
    );

    let out = rows
        .iter()
        .zip(scores)
        .map(|(r, s)| ScoreRow {
            refuse: r.refuse,
            score: s,
            config_hash_hex: r.config_hash_hex.clone(),
        })
        .collect();

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_vec_length() {
        // Ensure we have exactly 19 features as documented
        let m = BacktestMetricsV1 {
            schema_version: "v1".into(),
            total_trades: 10,
            winning_trades: 6,
            losing_trades: 4,
            win_rate: 0.6,
            gross_profit: 100.0,
            gross_loss: 50.0,
            net_pnl: 50.0,
            profit_factor: 2.0,
            expectancy: 5.0,
            avg_win: 16.67,
            avg_loss: 12.5,
            avg_win_loss_ratio: 1.33,
            largest_win: 30.0,
            largest_loss: 20.0,
            max_drawdown: 15.0,
            max_drawdown_pct: 1.5,
            sharpe_ratio: 1.5,
            sortino_ratio: 2.0,
            avg_trade_duration_secs: 3600.0,
            total_fees: 5.0,
        };
        let features = metrics_to_feature_vec(&m);
        assert_eq!(features.len(), 19);
    }

    #[test]
    fn test_invalid_metrics_detection() {
        let mut m = BacktestMetricsV1 {
            schema_version: "v1".into(),
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            win_rate: f64::NAN,
            gross_profit: 0.0,
            gross_loss: 0.0,
            net_pnl: 0.0,
            profit_factor: f64::INFINITY,
            expectancy: 0.0,
            avg_win: 0.0,
            avg_loss: 0.0,
            avg_win_loss_ratio: 0.0,
            largest_win: 0.0,
            largest_loss: 0.0,
            max_drawdown: 0.0,
            max_drawdown_pct: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            avg_trade_duration_secs: 0.0,
            total_fees: 0.0,
        };
        assert!(has_invalid_metrics(&m));

        m.win_rate = 0.0;
        m.profit_factor = 0.0;
        assert!(!has_invalid_metrics(&m));
    }
}
