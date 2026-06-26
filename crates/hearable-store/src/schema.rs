//! Database schema. Kept as plain SQL strings; a real migration runner
//! (`rusqlite_migration`) is introduced when the schema starts to evolve in Phase 5.

pub const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS profiles (
    name       TEXT PRIMARY KEY,
    embeddings BLOB NOT NULL,
    created_ms INTEGER NOT NULL DEFAULT 0,
    updated_ms INTEGER NOT NULL DEFAULT 0
);
"#;
