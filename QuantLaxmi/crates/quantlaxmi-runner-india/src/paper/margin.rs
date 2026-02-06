//! India F&O Margin Gate - Actual SPAN Margins via Zerodha API
//!
//! Uses Zerodha Kite Connect margin API for accurate SPAN calculations.
//! No approximations - real exchange margin requirements.
//!
//! ## API Endpoint
//!
//! POST https://api.kite.trade/margins/basket
//!
//! Returns actual SPAN + exposure margins from NSE clearing.
//!
//! ## Critical: Basket Margin
//!
//! For spread/hedge positions, use `data.final.total` NOT sum of per-order totals.
//! The API applies netting benefits at the basket level.

use anyhow::{Context, Result};
use serde::Serialize;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};
use tracing::{debug, info};

/// Order parameters for margin calculation.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct MarginOrderParams {
    pub exchange: String,         // "NFO"
    pub tradingsymbol: String,    // e.g., "NIFTY2510223400CE"
    pub transaction_type: String, // "BUY" or "SELL"
    pub variety: String,          // "regular"
    pub product: String,          // "NRML" for F&O
    pub order_type: String,       // "MARKET" or "LIMIT"
    pub quantity: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub price: Option<f64>,
}

// Manual Hash impl to handle Option<f64>
impl MarginOrderParams {
    fn cache_key(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.exchange.hash(&mut hasher);
        self.tradingsymbol.hash(&mut hasher);
        self.transaction_type.hash(&mut hasher);
        self.variety.hash(&mut hasher);
        self.product.hash(&mut hasher);
        self.order_type.hash(&mut hasher);
        self.quantity.hash(&mut hasher);
        hasher.finish()
    }
}

// NOTE: We deserialize the Zerodha /margins/basket response as serde_json::Value
// rather than typed structs. The API shape has many optional/varying fields and
// we only need `data.final.total` plus per-order diagnostics. This approach is
// robust against upstream API changes.

/// Margin requirement for a basket (uses final_margin.total).
#[derive(Debug, Clone)]
pub struct MarginRequirement {
    /// Total basket margin required (from final_margin.total)
    /// This is the ONLY number to use for accept/reject decisions.
    pub total: f64,
    /// Per-order breakdown (diagnostics only)
    pub per_order_span: f64,
    pub per_order_exposure: f64,
    pub per_order_premium: f64,
    /// Whether this came from cache
    pub from_cache: bool,
}

/// Cached basket margin with TTL.
#[derive(Debug, Clone)]
struct CachedMargin {
    requirement: MarginRequirement,
    cached_at: Instant,
}

/// Margin gate using Zerodha's SPAN API.
///
/// ## Design Notes
///
/// - Uses `final_margin.total` for basket decisions (includes netting)
/// - Caches basket results with TTL to avoid rate limits
/// - Tracks `margin_used` for open positions
/// - Tracks `available_cash` (updated from ledger each step)
/// - `margin_at_risk_cap` limits total margin exposure (NOT drawdown risk)
pub struct MarginGate {
    api_key: String,
    access_token: String,
    client: reqwest::Client,
    /// Basket cache: hash of orders -> cached margin
    basket_cache: std::collections::HashMap<u64, CachedMargin>,
    /// Cache TTL (default 5 seconds)
    cache_ttl: Duration,
    /// Current margin reserved by open positions
    margin_used: f64,
    /// Current available cash (updated from ledger before each step)
    available_cash: f64,
    /// Max margin-at-risk as fraction of available cash (e.g., 0.80 = 80%)
    /// This is NOT drawdown risk - it's a crude leverage cap.
    margin_at_risk_cap: f64,
}

impl MarginGate {
    pub fn new(api_key: String, access_token: String) -> Self {
        Self {
            api_key,
            access_token,
            client: reqwest::Client::new(),
            basket_cache: std::collections::HashMap::new(),
            cache_ttl: Duration::from_secs(5),
            margin_used: 0.0,
            available_cash: 0.0,
            margin_at_risk_cap: 0.80, // 80% of available cash max
        }
    }

    /// Set cache TTL.
    pub fn with_cache_ttl(mut self, ttl: Duration) -> Self {
        self.cache_ttl = ttl;
        self
    }

    /// Set margin-at-risk cap (NOT drawdown).
    /// This limits total margin exposure as fraction of available cash.
    pub fn with_margin_at_risk_cap(mut self, cap: f64) -> Self {
        self.margin_at_risk_cap = cap.clamp(0.1, 1.0);
        self
    }

    /// Update available cash from ledger.
    /// MUST be called before each engine step for accurate margin checks.
    pub fn set_available_cash(&mut self, cash: f64) {
        self.available_cash = cash;
    }

    /// Get current available cash.
    pub fn available_cash(&self) -> f64 {
        self.available_cash
    }

    /// Compute basket cache key from orders.
    fn basket_cache_key(orders: &[MarginOrderParams]) -> u64 {
        let mut hasher = DefaultHasher::new();
        for order in orders {
            order.cache_key().hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Fetch actual SPAN margin from Zerodha basket API.
    /// Uses /margins/basket which returns initial/final with netting benefits.
    /// Parses response as generic JSON and extracts only the fields we need.
    async fn fetch_margin(&self, orders: &[MarginOrderParams]) -> Result<MarginRequirement> {
        let url = "https://api.kite.trade/margins/basket";

        let response = self
            .client
            .post(url)
            .header("X-Kite-Version", "3")
            .header(
                "Authorization",
                format!("token {}:{}", self.api_key, self.access_token),
            )
            .json(orders)
            .send()
            .await
            .context("Failed to call Zerodha margin API")?;

        let status = response.status();
        let body = response.text().await?;

        debug!(status = %status, body_len = body.len(), "Margin API response");
        debug!(raw_body = %body, "Margin API raw response body");

        if !status.is_success() {
            anyhow::bail!("Margin API error ({}): {}", status, body);
        }

        let resp: serde_json::Value = serde_json::from_str(&body)
            .map_err(|e| anyhow::anyhow!("JSON parse error: {} | Body: {}", e, body))?;

        if resp.get("status").and_then(|s| s.as_str()) != Some("success") {
            let msg = resp
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown");
            anyhow::bail!("Margin API failed: {}", msg);
        }

        let data = resp
            .get("data")
            .ok_or_else(|| anyhow::anyhow!("No 'data' field in margin response"))?;

        // CRITICAL: Use data.final.total for basket decision (includes netting)
        let basket_total = data
            .get("final")
            .and_then(|f| f.get("total"))
            .and_then(|t| t.as_f64())
            .ok_or_else(|| anyhow::anyhow!("Missing data.final.total in margin response"))?;

        // Per-order breakdown for diagnostics only
        let orders_arr = data.get("orders").and_then(|o| o.as_array());
        let (per_order_span, per_order_exposure, per_order_premium) = if let Some(arr) = orders_arr
        {
            let span: f64 = arr
                .iter()
                .filter_map(|o| o.get("span").and_then(|v| v.as_f64()))
                .sum();
            let exposure: f64 = arr
                .iter()
                .filter_map(|o| o.get("exposure").and_then(|v| v.as_f64()))
                .sum();
            let premium: f64 = arr
                .iter()
                .filter_map(|o| o.get("option_premium").and_then(|v| v.as_f64()))
                .sum();
            (span, exposure, premium)
        } else {
            (0.0, 0.0, 0.0)
        };

        info!(
            basket_total = basket_total,
            per_order_span = per_order_span,
            per_order_exposure = per_order_exposure,
            "[MARGIN] Basket margin from API (using data.final.total)"
        );

        Ok(MarginRequirement {
            total: basket_total,
            per_order_span,
            per_order_exposure,
            per_order_premium,
            from_cache: false,
        })
    }

    /// Check margin for a basket of orders.
    ///
    /// Uses `final_margin.total` for the decision (includes spread/hedge benefits).
    /// Caches results to avoid rate limiting.
    /// Uses internal `available_cash` (call `set_available_cash` before each step).
    ///
    /// # Arguments
    /// * `orders` - Basket of orders to check
    pub async fn check_basket_entry(
        &mut self,
        orders: Vec<MarginOrderParams>,
    ) -> Result<MarginRequirement, MarginRejectReason> {
        let available_cash = self.available_cash;
        if orders.is_empty() {
            return Err(MarginRejectReason::InvalidRequest(
                "Empty order basket".to_string(),
            ));
        }

        // Check cache first
        let cache_key = Self::basket_cache_key(&orders);
        if let Some(cached) = self.basket_cache.get(&cache_key)
            && cached.cached_at.elapsed() < self.cache_ttl
        {
            debug!(
                cache_key = cache_key,
                total = cached.requirement.total,
                "[MARGIN] Using cached basket margin"
            );
            let mut req = cached.requirement.clone();
            req.from_cache = true;
            return self.validate_margin(available_cash, req);
        }

        // Fetch from API
        let requirement = self
            .fetch_margin(&orders)
            .await
            .map_err(|e| MarginRejectReason::ApiError(e.to_string()))?;

        // Cache it
        self.basket_cache.insert(
            cache_key,
            CachedMargin {
                requirement: requirement.clone(),
                cached_at: Instant::now(),
            },
        );

        self.validate_margin(available_cash, requirement)
    }

    fn validate_margin(
        &self,
        available_cash: f64,
        requirement: MarginRequirement,
    ) -> Result<MarginRequirement, MarginRejectReason> {
        // Available margin = cash - already reserved
        let available_margin = (available_cash - self.margin_used).max(0.0);

        // Check 1: Do we have enough margin?
        if requirement.total > available_margin {
            return Err(MarginRejectReason::InsufficientMargin {
                required: requirement.total,
                available: available_margin,
                already_used: self.margin_used,
            });
        }

        // Check 2: Would this exceed margin-at-risk cap?
        // This is NOT drawdown risk - it's a crude leverage limit.
        let total_margin_if_accepted = self.margin_used + requirement.total;
        let max_margin_allowed = available_cash * self.margin_at_risk_cap;

        if total_margin_if_accepted > max_margin_allowed {
            return Err(MarginRejectReason::MarginAtRiskCapExceeded {
                would_use: total_margin_if_accepted,
                cap: max_margin_allowed,
                cap_pct: self.margin_at_risk_cap * 100.0,
            });
        }

        Ok(requirement)
    }

    /// Reserve margin when a position is opened.
    /// Call this AFTER fills succeed for the entry.
    pub fn reserve_margin(&mut self, margin: f64) {
        self.margin_used += margin;
        info!(
            reserved = margin,
            total_used = self.margin_used,
            "[MARGIN] Reserved margin for position"
        );
    }

    /// Release margin when a position is closed.
    /// Call this AFTER fills succeed for the exit.
    pub fn release_margin(&mut self, margin: f64) {
        self.margin_used = (self.margin_used - margin).max(0.0);
        info!(
            released = margin,
            total_used = self.margin_used,
            "[MARGIN] Released margin from position"
        );
    }

    /// Get current margin utilization.
    pub fn margin_used(&self) -> f64 {
        self.margin_used
    }

    /// Clear cache (call on reconnect or token refresh).
    pub fn clear_cache(&mut self) {
        self.basket_cache.clear();
    }

    /// Prune expired cache entries.
    pub fn prune_cache(&mut self) {
        let ttl = self.cache_ttl;
        self.basket_cache.retain(|_, v| v.cached_at.elapsed() < ttl);
    }
}

/// Reason for margin rejection.
#[derive(Debug, Clone)]
pub enum MarginRejectReason {
    /// SPAN margin exceeds available
    InsufficientMargin {
        required: f64,
        available: f64,
        already_used: f64,
    },
    /// Would exceed margin-at-risk cap (NOT drawdown)
    MarginAtRiskCapExceeded {
        would_use: f64,
        cap: f64,
        cap_pct: f64,
    },
    /// API call failed
    ApiError(String),
    /// Invalid request
    InvalidRequest(String),
}

impl std::fmt::Display for MarginRejectReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MarginRejectReason::InsufficientMargin {
                required,
                available,
                already_used,
            } => {
                write!(
                    f,
                    "Insufficient margin: need ₹{:.2}, have ₹{:.2} available (₹{:.2} already reserved)",
                    required, available, already_used
                )
            }
            MarginRejectReason::MarginAtRiskCapExceeded {
                would_use,
                cap,
                cap_pct,
            } => {
                write!(
                    f,
                    "Margin-at-risk cap exceeded: would use ₹{:.2}, cap is ₹{:.2} ({:.0}%)",
                    would_use, cap, cap_pct
                )
            }
            MarginRejectReason::ApiError(msg) => write!(f, "Margin API error: {}", msg),
            MarginRejectReason::InvalidRequest(msg) => write!(f, "Invalid request: {}", msg),
        }
    }
}

impl std::error::Error for MarginRejectReason {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_margin_sufficient() {
        let gate = MarginGate::new("test_key".to_string(), "test_token".to_string());

        let requirement = MarginRequirement {
            total: 50_000.0,
            per_order_span: 40_000.0,
            per_order_exposure: 10_000.0,
            per_order_premium: 0.0,
            from_cache: false,
        };

        let result = gate.validate_margin(100_000.0, requirement);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_margin_insufficient() {
        let gate = MarginGate::new("test_key".to_string(), "test_token".to_string());

        let requirement = MarginRequirement {
            total: 150_000.0,
            per_order_span: 100_000.0,
            per_order_exposure: 50_000.0,
            per_order_premium: 0.0,
            from_cache: false,
        };

        let result = gate.validate_margin(100_000.0, requirement);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            MarginRejectReason::InsufficientMargin { .. }
        ));
    }

    #[test]
    fn test_validate_margin_cap_exceeded() {
        let gate = MarginGate::new("test_key".to_string(), "test_token".to_string())
            .with_margin_at_risk_cap(0.50); // 50% cap

        let requirement = MarginRequirement {
            total: 60_000.0, // 60% of 100k
            per_order_span: 50_000.0,
            per_order_exposure: 10_000.0,
            per_order_premium: 0.0,
            from_cache: false,
        };

        let result = gate.validate_margin(100_000.0, requirement);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            MarginRejectReason::MarginAtRiskCapExceeded { .. }
        ));
    }

    #[test]
    fn test_reserve_release_margin() {
        let mut gate = MarginGate::new("test_key".to_string(), "test_token".to_string());

        assert_eq!(gate.margin_used(), 0.0);

        gate.reserve_margin(50_000.0);
        assert_eq!(gate.margin_used(), 50_000.0);

        gate.reserve_margin(30_000.0);
        assert_eq!(gate.margin_used(), 80_000.0);

        gate.release_margin(50_000.0);
        assert_eq!(gate.margin_used(), 30_000.0);

        gate.release_margin(50_000.0); // Release more than used
        assert_eq!(gate.margin_used(), 0.0); // Clamped to 0
    }

    #[test]
    fn test_basket_cache_key() {
        let orders1 = vec![
            MarginOrderParams {
                exchange: "NFO".to_string(),
                tradingsymbol: "NIFTY25FEB25000CE".to_string(),
                transaction_type: "SELL".to_string(),
                variety: "regular".to_string(),
                product: "NRML".to_string(),
                order_type: "MARKET".to_string(),
                quantity: 25,
                price: None,
            },
            MarginOrderParams {
                exchange: "NFO".to_string(),
                tradingsymbol: "NIFTY25FEB25000PE".to_string(),
                transaction_type: "SELL".to_string(),
                variety: "regular".to_string(),
                product: "NRML".to_string(),
                order_type: "MARKET".to_string(),
                quantity: 25,
                price: None,
            },
        ];

        let orders2 = orders1.clone();
        let orders3 = vec![orders1[0].clone()]; // Different basket

        let key1 = MarginGate::basket_cache_key(&orders1);
        let key2 = MarginGate::basket_cache_key(&orders2);
        let key3 = MarginGate::basket_cache_key(&orders3);

        assert_eq!(key1, key2); // Same orders = same key
        assert_ne!(key1, key3); // Different orders = different key
    }
}
