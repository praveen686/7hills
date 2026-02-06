//! Session directory management for paper trading.

use std::path::{Path, PathBuf};

/// Session directory for paper trading.
///
/// Creates a structured directory for storing session artifacts:
/// - `depth_wal.jsonl` - Market depth events
/// - `fills.jsonl` - Execution fills
/// - `ledger.jsonl` - Ledger/position updates
pub struct SessionDir {
    root: PathBuf,
    symbol: String,
}

impl SessionDir {
    /// Create a new paper trading session directory.
    ///
    /// Creates: `<base_dir>/paper_sessions/<symbol>_<timestamp>/`
    pub async fn new_paper(base_dir: &Path, symbol: &str) -> anyhow::Result<Self> {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();

        let root = base_dir
            .join("paper_sessions")
            .join(format!("{symbol}_{ts}"));

        tokio::fs::create_dir_all(&root).await?;

        tracing::info!("Created session directory: {:?}", root);

        Ok(Self {
            root,
            symbol: symbol.to_string(),
        })
    }

    /// Create a session directory at an explicit path.
    pub async fn at_path(path: impl AsRef<Path>, symbol: &str) -> anyhow::Result<Self> {
        let root = path.as_ref().to_path_buf();
        tokio::fs::create_dir_all(&root).await?;

        Ok(Self {
            root,
            symbol: symbol.to_string(),
        })
    }

    /// Get the root path of the session directory.
    pub fn path(&self) -> &Path {
        &self.root
    }

    /// Get the symbol for this session.
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// Get the path for the depth WAL file.
    pub fn depth_wal_path(&self) -> PathBuf {
        self.root.join("depth_wal.jsonl")
    }

    /// Get the path for the fills file.
    pub fn fills_path(&self) -> PathBuf {
        self.root.join("fills.jsonl")
    }

    /// Get the path for the ledger file.
    pub fn ledger_path(&self) -> PathBuf {
        self.root.join("ledger.jsonl")
    }

    /// Get the path for the equity curve file.
    pub fn equity_path(&self) -> PathBuf {
        self.root.join("equity.jsonl")
    }

    /// Get the path for session metadata/manifest.
    pub fn manifest_path(&self) -> PathBuf {
        self.root.join("manifest.json")
    }
}
