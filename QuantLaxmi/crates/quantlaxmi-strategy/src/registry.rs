//! Strategy registry for discovery and instantiation.
//!
//! The registry maps strategy names to factory functions.
//! Each factory loads its concrete config type and returns a boxed strategy.

use crate::Strategy;
use crate::strategies::funding_bias::{FUNDING_BIAS_NAME, funding_bias_factory};
use crate::strategies::micro_breakout::{MICRO_BREAKOUT_NAME, micro_breakout_factory};
use crate::strategies::spread_mean_revert::{SPREAD_MEAN_REVERT_NAME, spread_mean_revert_factory};
use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

/// Factory function type for creating strategies.
///
/// Takes an optional config path and returns a boxed strategy.
/// The factory is responsible for:
/// 1. Loading the concrete config type
/// 2. Parsing/validating the config
/// 3. Computing config hash (via canonical_bytes)
/// 4. Constructing the strategy
pub type StrategyFactory = fn(config_path: Option<&Path>) -> Result<Box<dyn Strategy>>;

/// Registry for strategy discovery and instantiation.
///
/// ## Usage
/// ```ignore
/// let registry = StrategyRegistry::with_builtins();
/// let strategy = registry.create("funding_bias", Some(Path::new("config.toml")))?;
/// println!("Strategy ID: {}", strategy.strategy_id());
/// ```
pub struct StrategyRegistry {
    factories: HashMap<String, StrategyFactory>,
}

impl Default for StrategyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    /// Create a registry with built-in strategies pre-registered.
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();

        // Register built-in strategies
        registry.register(FUNDING_BIAS_NAME, funding_bias_factory);
        registry.register(MICRO_BREAKOUT_NAME, micro_breakout_factory);
        registry.register(SPREAD_MEAN_REVERT_NAME, spread_mean_revert_factory);

        registry
    }

    /// Register a strategy factory.
    ///
    /// # Arguments
    /// * `name` - Strategy name (e.g., "funding_bias")
    /// * `factory` - Factory function that creates strategy instances
    pub fn register(&mut self, name: &str, factory: StrategyFactory) {
        self.factories.insert(name.to_string(), factory);
    }

    /// Create a strategy instance by name.
    ///
    /// # Arguments
    /// * `name` - Strategy name (must be registered)
    /// * `config_path` - Optional path to config file (TOML format)
    ///
    /// # Returns
    /// Boxed strategy instance, or error if name not found or config invalid.
    pub fn create(&self, name: &str, config_path: Option<&Path>) -> Result<Box<dyn Strategy>> {
        let factory = self.factories.get(name).ok_or_else(|| {
            anyhow::anyhow!(
                "Unknown strategy: '{}'. Use --list to see available strategies.",
                name
            )
        })?;
        factory(config_path)
    }

    /// List all registered strategy names.
    pub fn list(&self) -> Vec<&str> {
        let mut names: Vec<_> = self.factories.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    /// Check if a strategy is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.factories.contains_key(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_with_builtins() {
        let registry = StrategyRegistry::with_builtins();
        assert!(registry.contains(FUNDING_BIAS_NAME));
    }

    #[test]
    fn test_registry_list() {
        let registry = StrategyRegistry::with_builtins();
        let names = registry.list();
        assert!(names.contains(&FUNDING_BIAS_NAME));
    }

    #[test]
    fn test_registry_create_with_defaults() {
        let registry = StrategyRegistry::with_builtins();
        let strategy = registry.create(FUNDING_BIAS_NAME, None).unwrap();
        assert_eq!(strategy.name(), FUNDING_BIAS_NAME);
        assert!(!strategy.config_hash().is_empty());
    }

    #[test]
    fn test_registry_unknown_strategy() {
        let registry = StrategyRegistry::with_builtins();
        let result = registry.create("nonexistent", None);
        assert!(result.is_err());
    }
}
