//! Two-Pass Tournament Mode
//!
//! Pass 1: Wide scan across stratified segment sample, all configs
//! Pass 2: Deep validation on all segments, top-K configs only
//!
//! Key features:
//! - Deterministic stratified segment selection (time-diverse)
//! - Robust per-config aggregation (median + p20 + refuse_rate)
//! - Configurable promotion criteria

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;
use tracing::info;

use crate::tournament::{GridRunResult, ParamValue};

/// Two-pass tournament configuration.
#[derive(Debug, Clone)]
pub struct TwoPassConfig {
    /// Fraction of segments to use in Pass 1 (0.0-1.0)
    pub pass1_segment_fraction: f64,
    /// Number of bins for stratified sampling
    pub pass1_bins: usize,
    /// Number of top configs to promote to Pass 2
    pub select_top_k: usize,
    /// Refuse rate threshold - configs above this are excluded
    pub refuse_threshold: f64,
    /// Optional max configs per parameter family
    pub max_per_family: Option<usize>,
}

impl Default for TwoPassConfig {
    fn default() -> Self {
        Self {
            pass1_segment_fraction: 0.30,
            pass1_bins: 10,
            select_top_k: 50,
            refuse_threshold: 0.50,
            max_per_family: None,
        }
    }
}

/// Segment selection result for Pass 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pass1SegmentSelection {
    pub schema_version: String,
    pub total_segments: usize,
    pub selected_count: usize,
    pub target_fraction: f64,
    pub bins_used: usize,
    pub seed_hex: String,
    pub selected_indices: Vec<usize>,
    pub selected_names: Vec<String>,
}

/// Per-config aggregation across Pass 1 segments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigAggregation {
    pub param_hash: String,
    pub params: BTreeMap<String, ParamValue>,
    pub segment_count: usize,
    pub valid_count: usize,
    pub refused_count: usize,
    pub refuse_rate: f64,
    pub median_score: f64,
    pub p20_score: f64,
    pub mean_score: f64,
    pub promo_score: f64,
    pub promoted: bool,
}

/// Selected config for Pass 2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectedConfig {
    pub rank: usize,
    pub param_hash: String,
    pub params: BTreeMap<String, ParamValue>,
    pub promo_score: f64,
    pub median_score: f64,
    pub p20_score: f64,
    pub refuse_rate: f64,
    pub pass1_segments: usize,
}

/// Deterministic stratified segment selection.
///
/// Algorithm:
/// 1. Sort segments by name (assumed to encode time)
/// 2. Partition into `bins` equal-sized bins
/// 3. From each bin, select k segments using deterministic hash
///
/// This ensures time diversity while maintaining reproducibility.
pub fn select_pass1_segments(
    all_segments: &[String],
    config: &TwoPassConfig,
    seed_material: &str,
) -> Pass1SegmentSelection {
    let n = all_segments.len();
    let target_count = ((n as f64) * config.pass1_segment_fraction).ceil() as usize;
    let target_count = target_count.max(1).min(n);

    // Compute deterministic seed from material
    let mut hasher = Sha256::new();
    hasher.update(seed_material.as_bytes());
    let seed_hash = hasher.finalize();
    let seed_hex = hex::encode(&seed_hash[..8]);

    // Sort segments (they should already be sorted, but ensure it)
    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| all_segments[a].cmp(&all_segments[b]));

    // Partition into bins
    let bins = config.pass1_bins.min(n).max(1);
    let per_bin = target_count.div_ceil(bins);

    let mut selected_indices: Vec<usize> = Vec::new();

    for bin_idx in 0..bins {
        let bin_start = (bin_idx * n) / bins;
        let bin_end = ((bin_idx + 1) * n) / bins;
        let bin_size = bin_end - bin_start;

        if bin_size == 0 {
            continue;
        }

        // Deterministic selection within bin using seed + bin index
        let mut bin_hasher = Sha256::new();
        bin_hasher.update(seed_hash);
        bin_hasher.update((bin_idx as u64).to_le_bytes());
        let bin_hash = bin_hasher.finalize();

        // Select up to per_bin segments from this bin
        let select_from_bin = per_bin.min(bin_size);
        for sel_idx in 0..select_from_bin {
            // Use hash bytes to pick deterministically
            let hash_byte = bin_hash[(sel_idx * 4) % 32] as usize;
            let offset = (hash_byte * bin_size) / 256;
            let pick = bin_start + (offset % bin_size);

            let actual_idx = sorted_indices[pick];
            if !selected_indices.contains(&actual_idx) {
                selected_indices.push(actual_idx);
            }

            if selected_indices.len() >= target_count {
                break;
            }
        }

        if selected_indices.len() >= target_count {
            break;
        }
    }

    // Sort selected indices for consistent ordering
    selected_indices.sort();

    let selected_names: Vec<String> = selected_indices
        .iter()
        .map(|&i| all_segments[i].clone())
        .collect();

    Pass1SegmentSelection {
        schema_version: "v1".to_string(),
        total_segments: n,
        selected_count: selected_indices.len(),
        target_fraction: config.pass1_segment_fraction,
        bins_used: bins,
        seed_hex,
        selected_indices,
        selected_names,
    }
}

/// Aggregate Pass 1 results per config.
///
/// Computes robust statistics: median, p20, refuse_rate, and promotion score.
pub fn aggregate_pass1_results(
    results: &[GridRunResult],
    config: &TwoPassConfig,
) -> Vec<ConfigAggregation> {
    // Group by param_hash
    let mut by_hash: BTreeMap<String, Vec<&GridRunResult>> = BTreeMap::new();
    for r in results {
        by_hash.entry(r.param_hash.clone()).or_default().push(r);
    }

    let mut aggregations: Vec<ConfigAggregation> = Vec::new();

    for (param_hash, runs) in &by_hash {
        if runs.is_empty() {
            continue;
        }

        let params = runs[0].params.clone();
        let segment_count = runs.len();

        // Collect scores (GPU scores if available, else CPU fallback)
        let mut scores: Vec<f64> = Vec::new();
        let mut refused_count = 0usize;

        for r in runs {
            let is_refused =
                r.metrics.total_trades == 0 || r.gpu_score.map(|s| s < -1e20).unwrap_or(false);

            if is_refused {
                refused_count += 1;
            } else if let Some(gpu_score) = r.gpu_score {
                scores.push(gpu_score as f64);
            } else {
                // CPU fallback
                let cpu_score = r.manifest.return_pct / 100.0 - 2.0 * r.metrics.max_drawdown_pct;
                scores.push(cpu_score);
            }
        }

        let valid_count = scores.len();
        let refuse_rate = refused_count as f64 / segment_count as f64;

        // Compute statistics
        let (median_score, p20_score, mean_score) = if scores.is_empty() {
            (f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY)
        } else {
            scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let median = if scores.len().is_multiple_of(2) && scores.len() > 1 {
                let mid = scores.len() / 2;
                (scores[mid - 1] + scores[mid]) / 2.0
            } else {
                scores[scores.len() / 2]
            };

            // 20th percentile (lower tail)
            let p20_idx = ((scores.len() as f64) * 0.20).floor() as usize;
            let p20 = scores[p20_idx.min(scores.len() - 1)];

            let mean = scores.iter().sum::<f64>() / scores.len() as f64;

            (median, p20, mean)
        };

        // Promotion score: 0.7 * median + 0.3 * p20 - 2.0 * refuse_rate
        let promo_score = 0.7 * median_score + 0.3 * p20_score - 2.0 * refuse_rate;

        // Check if promoted (will be finalized after sorting)
        let promoted = refuse_rate <= config.refuse_threshold && valid_count > 0;

        aggregations.push(ConfigAggregation {
            param_hash: param_hash.clone(),
            params,
            segment_count,
            valid_count,
            refused_count,
            refuse_rate,
            median_score,
            p20_score,
            mean_score,
            promo_score,
            promoted,
        });
    }

    // Sort by promo_score descending
    aggregations.sort_by(|a, b| {
        b.promo_score
            .partial_cmp(&a.promo_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    aggregations
}

/// Select top-K configs for Pass 2.
pub fn select_configs_for_pass2(
    aggregations: &mut [ConfigAggregation],
    config: &TwoPassConfig,
) -> Vec<SelectedConfig> {
    // Mark top-K as promoted (respecting refuse threshold)
    let mut promoted_count = 0;
    for agg in aggregations.iter_mut() {
        if agg.refuse_rate > config.refuse_threshold || agg.valid_count == 0 {
            agg.promoted = false;
            continue;
        }

        if promoted_count < config.select_top_k {
            agg.promoted = true;
            promoted_count += 1;
        } else {
            agg.promoted = false;
        }
    }

    // Build selected configs list
    let selected: Vec<SelectedConfig> = aggregations
        .iter()
        .filter(|a| a.promoted)
        .enumerate()
        .map(|(i, a)| SelectedConfig {
            rank: i + 1,
            param_hash: a.param_hash.clone(),
            params: a.params.clone(),
            promo_score: a.promo_score,
            median_score: a.median_score,
            p20_score: a.p20_score,
            refuse_rate: a.refuse_rate,
            pass1_segments: a.segment_count,
        })
        .collect();

    selected
}

/// Write Pass 1 artifacts.
pub fn write_pass1_artifacts(
    out_dir: &Path,
    segment_selection: &Pass1SegmentSelection,
    aggregations: &[ConfigAggregation],
    selected_configs: &[SelectedConfig],
) -> Result<()> {
    use std::io::Write;

    // pass1_segments.json
    let segments_path = out_dir.join("pass1_segments.json");
    let segments_json = serde_json::to_string_pretty(segment_selection)?;
    std::fs::write(&segments_path, segments_json)?;
    info!("Wrote {}", segments_path.display());

    // pass1_config_agg.csv
    let agg_path = out_dir.join("pass1_config_agg.csv");
    let file = std::fs::File::create(&agg_path)?;
    let mut writer = std::io::BufWriter::new(file);

    writeln!(
        writer,
        "param_hash,segment_count,valid_count,refused_count,refuse_rate,median_score,p20_score,mean_score,promo_score,promoted"
    )?;

    for agg in aggregations {
        writeln!(
            writer,
            "{},{},{},{},{:.4},{:.6},{:.6},{:.6},{:.6},{}",
            agg.param_hash,
            agg.segment_count,
            agg.valid_count,
            agg.refused_count,
            agg.refuse_rate,
            agg.median_score,
            agg.p20_score,
            agg.mean_score,
            agg.promo_score,
            if agg.promoted { 1 } else { 0 }
        )?;
    }
    drop(writer);
    info!("Wrote {}", agg_path.display());

    // selected_configs.jsonl
    let selected_path = out_dir.join("selected_configs.jsonl");
    let file = std::fs::File::create(&selected_path)?;
    let mut writer = std::io::BufWriter::new(file);

    for cfg in selected_configs {
        let line = serde_json::to_string(cfg)?;
        writeln!(writer, "{}", line)?;
    }
    drop(writer);
    info!("Wrote {}", selected_path.display());

    // Summary log
    info!(
        "Pass 1 complete: {} configs evaluated, {} promoted to Pass 2",
        aggregations.len(),
        selected_configs.len()
    );

    Ok(())
}

/// Load selected configs from Pass 1 artifacts.
pub fn load_selected_configs(out_dir: &Path) -> Result<Vec<SelectedConfig>> {
    use std::io::BufRead;

    let selected_path = out_dir.join("selected_configs.jsonl");
    let file = std::fs::File::open(&selected_path)
        .with_context(|| format!("Failed to open {}", selected_path.display()))?;
    let reader = std::io::BufReader::new(file);

    let mut configs: Vec<SelectedConfig> = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let cfg: SelectedConfig = serde_json::from_str(&line)?;
        configs.push(cfg);
    }

    Ok(configs)
}

// =============================================================================
// Pass 2: Stability Report
// =============================================================================

/// Per-config stability metrics from Pass 2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigStability {
    pub param_hash: String,
    pub segment_count: usize,
    pub valid_count: usize,
    pub refused_count: usize,
    pub refuse_rate: f64,
    /// Median score across all Pass-2 segments
    pub median_score: f64,
    /// 20th percentile score (lower tail robustness)
    pub p20_score: f64,
    /// 80th percentile score (upper performance)
    pub p80_score: f64,
    /// Mean score
    pub mean_score: f64,
    /// Standard deviation of scores
    pub std_score: f64,
    /// Interquartile range (p75 - p25)
    pub iqr_score: f64,
    /// Segment win rate: % of segments where config is in top-N
    pub segment_win_rate: f64,
    /// Worst segment score
    pub worst_score: f64,
    /// Name of worst segment
    pub worst_segment: String,
    /// Best segment score
    pub best_score: f64,
    /// Name of best segment
    pub best_segment: String,
    /// Final stability score (composite)
    pub stability_score: f64,
}

/// Summary statistics across all configs in Pass 2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pass2StabilityStats {
    pub schema_version: String,
    pub total_configs: usize,
    pub total_segments: usize,
    pub total_runs: usize,
    /// Median of median_scores across configs
    pub median_of_medians: f64,
    /// Median of p20_scores across configs
    pub median_of_p20s: f64,
    /// Average segment win rate
    pub avg_segment_win_rate: f64,
    /// Configs with >50% segment win rate
    pub high_win_rate_count: usize,
    /// Average refuse rate
    pub avg_refuse_rate: f64,
    /// Score correlation between median and p20 (consistency indicator)
    pub median_p20_correlation: f64,
    /// Top 10 most stable configs (by stability_score)
    pub top_stable_configs: Vec<String>,
}

/// Compute stability metrics for Pass 2 results.
///
/// For each config, computes robust statistics across all segments.
/// Also determines segment win rate (% segments where config is in top-N).
pub fn compute_pass2_stability(
    results: &[GridRunResult],
    top_n_threshold: usize, // e.g., 10 for "top 10 in segment"
) -> Vec<ConfigStability> {
    // Group by param_hash
    let mut by_hash: BTreeMap<String, Vec<&GridRunResult>> = BTreeMap::new();
    for r in results {
        by_hash.entry(r.param_hash.clone()).or_default().push(r);
    }

    // Also compute per-segment rankings for win rate calculation
    let mut by_segment: BTreeMap<String, Vec<(&GridRunResult, f64)>> = BTreeMap::new();
    for r in results {
        let score = r
            .gpu_score
            .map(|s| s as f64)
            .unwrap_or_else(|| r.manifest.return_pct / 100.0 - 2.0 * r.metrics.max_drawdown_pct);
        by_segment
            .entry(r.segment.clone())
            .or_default()
            .push((r, score));
    }

    // Sort each segment by score descending
    for (_seg, runs) in by_segment.iter_mut() {
        runs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    }

    // For each config, count how many segments it's in top-N
    let mut config_wins: BTreeMap<String, usize> = BTreeMap::new();
    for runs in by_segment.values() {
        for (i, (r, _score)) in runs.iter().enumerate() {
            if i < top_n_threshold {
                *config_wins.entry(r.param_hash.clone()).or_insert(0) += 1;
            }
        }
    }

    let segment_count = by_segment.len();

    let mut stability_results: Vec<ConfigStability> = Vec::new();

    for (param_hash, runs) in &by_hash {
        if runs.is_empty() {
            continue;
        }

        let total = runs.len();

        // Collect scores and track worst/best
        let mut scores: Vec<f64> = Vec::new();
        let mut refused_count = 0usize;
        let mut worst_score = f64::INFINITY;
        let mut worst_segment = String::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_segment = String::new();

        for r in runs {
            let is_refused =
                r.metrics.total_trades == 0 || r.gpu_score.map(|s| s < -1e20).unwrap_or(false);

            if is_refused {
                refused_count += 1;
                continue;
            }

            let score = r.gpu_score.map(|s| s as f64).unwrap_or_else(|| {
                r.manifest.return_pct / 100.0 - 2.0 * r.metrics.max_drawdown_pct
            });

            scores.push(score);

            if score < worst_score {
                worst_score = score;
                worst_segment = r.segment.clone();
            }
            if score > best_score {
                best_score = score;
                best_segment = r.segment.clone();
            }
        }

        let valid_count = scores.len();
        let refuse_rate = refused_count as f64 / total as f64;

        // Compute statistics
        let (median, p20, p80, mean, std_dev, iqr) = if scores.is_empty() {
            (
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
                0.0,
                0.0,
            )
        } else {
            scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = scores.len();

            let median = if n.is_multiple_of(2) && n > 1 {
                (scores[n / 2 - 1] + scores[n / 2]) / 2.0
            } else {
                scores[n / 2]
            };

            let p20_idx = ((n as f64) * 0.20).floor() as usize;
            let p20 = scores[p20_idx.min(n - 1)];

            let p80_idx = ((n as f64) * 0.80).floor() as usize;
            let p80 = scores[p80_idx.min(n - 1)];

            let p25_idx = ((n as f64) * 0.25).floor() as usize;
            let p75_idx = ((n as f64) * 0.75).floor() as usize;
            let iqr = scores[p75_idx.min(n - 1)] - scores[p25_idx.min(n - 1)];

            let mean = scores.iter().sum::<f64>() / n as f64;
            let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n as f64;
            let std_dev = variance.sqrt();

            (median, p20, p80, mean, std_dev, iqr)
        };

        // Segment win rate
        let wins = config_wins.get(param_hash).copied().unwrap_or(0);
        let segment_win_rate = wins as f64 / segment_count as f64;

        // Stability score: rewards consistency, penalizes dispersion and refusal
        // stability = 0.5 * median + 0.3 * p20 + 0.2 * segment_win_rate - 0.5 * (std/median) - 1.0 * refuse_rate
        let normalized_std = if median.abs() > 1e-8 {
            std_dev / median.abs()
        } else {
            std_dev
        };
        let stability_score = 0.5 * median + 0.3 * p20 + 0.2 * segment_win_rate * 3.0
            - 0.5 * normalized_std.min(2.0)
            - 1.0 * refuse_rate;

        stability_results.push(ConfigStability {
            param_hash: param_hash.clone(),
            segment_count: total,
            valid_count,
            refused_count,
            refuse_rate,
            median_score: median,
            p20_score: p20,
            p80_score: p80,
            mean_score: mean,
            std_score: std_dev,
            iqr_score: iqr,
            segment_win_rate,
            worst_score: if worst_score.is_finite() {
                worst_score
            } else {
                0.0
            },
            worst_segment,
            best_score: if best_score.is_finite() {
                best_score
            } else {
                0.0
            },
            best_segment,
            stability_score,
        });
    }

    // Sort by stability_score descending
    stability_results.sort_by(|a, b| {
        b.stability_score
            .partial_cmp(&a.stability_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    stability_results
}

/// Compute summary statistics for Pass 2.
pub fn compute_pass2_stats(
    stability: &[ConfigStability],
    total_segments: usize,
) -> Pass2StabilityStats {
    let total_configs = stability.len();
    let total_runs: usize = stability.iter().map(|s| s.segment_count).sum();

    // Collect metrics for summary
    let medians: Vec<f64> = stability
        .iter()
        .map(|s| s.median_score)
        .filter(|&m| m.is_finite())
        .collect();
    let p20s: Vec<f64> = stability
        .iter()
        .map(|s| s.p20_score)
        .filter(|&p| p.is_finite())
        .collect();
    let win_rates: Vec<f64> = stability.iter().map(|s| s.segment_win_rate).collect();
    let refuse_rates: Vec<f64> = stability.iter().map(|s| s.refuse_rate).collect();

    let median_of_medians = if medians.is_empty() {
        0.0
    } else {
        let mut sorted = medians.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted[sorted.len() / 2]
    };

    let median_of_p20s = if p20s.is_empty() {
        0.0
    } else {
        let mut sorted = p20s.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted[sorted.len() / 2]
    };

    let avg_segment_win_rate = if win_rates.is_empty() {
        0.0
    } else {
        win_rates.iter().sum::<f64>() / win_rates.len() as f64
    };

    let high_win_rate_count = stability
        .iter()
        .filter(|s| s.segment_win_rate > 0.5)
        .count();

    let avg_refuse_rate = if refuse_rates.is_empty() {
        0.0
    } else {
        refuse_rates.iter().sum::<f64>() / refuse_rates.len() as f64
    };

    // Simple correlation between median and p20 (Pearson)
    let median_p20_correlation = if medians.len() >= 2 && medians.len() == p20s.len() {
        let n = medians.len() as f64;
        let mean_m = medians.iter().sum::<f64>() / n;
        let mean_p = p20s.iter().sum::<f64>() / n;

        let cov: f64 = medians
            .iter()
            .zip(p20s.iter())
            .map(|(m, p)| (m - mean_m) * (p - mean_p))
            .sum::<f64>()
            / n;

        let std_m = (medians.iter().map(|m| (m - mean_m).powi(2)).sum::<f64>() / n).sqrt();
        let std_p = (p20s.iter().map(|p| (p - mean_p).powi(2)).sum::<f64>() / n).sqrt();

        if std_m > 1e-8 && std_p > 1e-8 {
            cov / (std_m * std_p)
        } else {
            0.0
        }
    } else {
        0.0
    };

    let top_stable_configs: Vec<String> = stability
        .iter()
        .take(10)
        .map(|s| s.param_hash.clone())
        .collect();

    Pass2StabilityStats {
        schema_version: "v1".to_string(),
        total_configs,
        total_segments,
        total_runs,
        median_of_medians,
        median_of_p20s,
        avg_segment_win_rate,
        high_win_rate_count,
        avg_refuse_rate,
        median_p20_correlation,
        top_stable_configs,
    }
}

/// Write Pass 2 stability artifacts.
pub fn write_pass2_stability(
    out_dir: &Path,
    stability: &[ConfigStability],
    stats: &Pass2StabilityStats,
) -> Result<()> {
    use std::io::Write;

    // pass2_stability.csv
    let csv_path = out_dir.join("pass2_stability.csv");
    let file = std::fs::File::create(&csv_path)?;
    let mut writer = std::io::BufWriter::new(file);

    writeln!(
        writer,
        "param_hash,segment_count,valid_count,refused_count,refuse_rate,median_score,p20_score,p80_score,mean_score,std_score,iqr_score,segment_win_rate,worst_score,worst_segment,best_score,best_segment,stability_score"
    )?;

    for s in stability {
        writeln!(
            writer,
            "{},{},{},{},{:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.4},{:.6},{},{:.6},{},{:.6}",
            s.param_hash,
            s.segment_count,
            s.valid_count,
            s.refused_count,
            s.refuse_rate,
            s.median_score,
            s.p20_score,
            s.p80_score,
            s.mean_score,
            s.std_score,
            s.iqr_score,
            s.segment_win_rate,
            s.worst_score,
            s.worst_segment,
            s.best_score,
            s.best_segment,
            s.stability_score,
        )?;
    }
    drop(writer);
    info!("Wrote {}", csv_path.display());

    // pass2_stability_stats.json
    let stats_path = out_dir.join("pass2_stability_stats.json");
    let stats_json = serde_json::to_string_pretty(stats)?;
    std::fs::write(&stats_path, stats_json)?;
    info!("Wrote {}", stats_path.display());

    // Log summary
    info!(
        "Pass 2 stability: {} configs, median_of_medians={:.3}, avg_win_rate={:.1}%, high_win_rate={}",
        stats.total_configs,
        stats.median_of_medians,
        stats.avg_segment_win_rate * 100.0,
        stats.high_win_rate_count
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stratified_segment_selection() {
        let segments: Vec<String> = (1..=20).map(|i| format!("2024-01-{:02}", i)).collect();

        let config = TwoPassConfig {
            pass1_segment_fraction: 0.30,
            pass1_bins: 5,
            ..Default::default()
        };

        let selection = select_pass1_segments(&segments, &config, "test_seed");

        // Should select ~6 segments (30% of 20)
        assert!(selection.selected_count >= 5 && selection.selected_count <= 8);
        assert_eq!(selection.total_segments, 20);

        // Should be deterministic
        let selection2 = select_pass1_segments(&segments, &config, "test_seed");
        assert_eq!(selection.selected_indices, selection2.selected_indices);

        // Different seed should give different selection
        let selection3 = select_pass1_segments(&segments, &config, "different_seed");
        assert_ne!(selection.selected_indices, selection3.selected_indices);
    }

    #[test]
    fn test_promo_score_formula() {
        // promo_score = 0.7 * median + 0.3 * p20 - 2.0 * refuse_rate
        let median: f64 = 1.0;
        let p20: f64 = 0.5;
        let refuse_rate: f64 = 0.1;

        let expected: f64 = 0.7 * median + 0.3 * p20 - 2.0 * refuse_rate;
        // 0.7 + 0.15 - 0.2 = 0.65
        assert!((expected - 0.65).abs() < 0.001);
    }
}
