//! Generic JSONL (JSON Lines) writer for streaming serialization.

use std::path::Path;

use tokio::fs::OpenOptions;
use tokio::io::{AsyncWriteExt, BufWriter};

/// Generic JSONL writer for any serializable type.
pub struct JsonlWriter<T> {
    w: BufWriter<tokio::fs::File>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: serde::Serialize> JsonlWriter<T> {
    /// Open a file for appending JSONL records.
    pub async fn open_append(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await?;
        Ok(Self {
            w: BufWriter::new(f),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Open a file for writing JSONL records (truncate if exists).
    pub async fn open_new(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .await?;
        Ok(Self {
            w: BufWriter::new(f),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Write a record to the file.
    pub async fn write(&mut self, item: &T) -> anyhow::Result<()> {
        let line = serde_json::to_string(item)?;
        self.w.write_all(line.as_bytes()).await?;
        self.w.write_all(b"\n").await?;
        Ok(())
    }

    /// Flush buffered data to disk.
    pub async fn flush(&mut self) -> anyhow::Result<()> {
        self.w.flush().await?;
        Ok(())
    }
}

/// Generic JSONL reader for any deserializable type.
pub struct JsonlReader<T> {
    lines: std::io::Lines<std::io::BufReader<std::fs::File>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: serde::de::DeserializeOwned> JsonlReader<T> {
    /// Open a file for reading JSONL records.
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        use std::io::BufRead;
        let f = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(f);
        Ok(Self {
            lines: reader.lines(),
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<T: serde::de::DeserializeOwned> Iterator for JsonlReader<T> {
    type Item = anyhow::Result<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.lines.next().map(|line_result| {
            let line = line_result?;
            let item = serde_json::from_str(&line)?;
            Ok(item)
        })
    }
}
