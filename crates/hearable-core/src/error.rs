/// The crate-wide error type. Domain crates map their native errors into these variants.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("audio error: {0}")]
    Audio(String),
    #[error("asr error: {0}")]
    Asr(String),
    #[error("speaker error: {0}")]
    Speaker(String),
    #[error("store error: {0}")]
    Store(String),
    #[error("config error: {0}")]
    Config(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
