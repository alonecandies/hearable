use hearable_core::{Embedding, Error, Profile, ProfileStore, Result};
use rusqlite::Connection;
use std::path::Path;

/// SQLite-backed [`ProfileStore`]. Stores voice embeddings only — never raw audio (spec §6).
pub struct SqliteProfileStore {
    conn: Connection,
}

impl SqliteProfileStore {
    /// Open an in-memory database (used by tests).
    pub fn open_in_memory() -> Result<Self> {
        Self::from_conn(Connection::open_in_memory().map_err(se)?)
    }

    /// Open (creating if needed) a database file at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_conn(Connection::open(path).map_err(se)?)
    }

    fn from_conn(conn: Connection) -> Result<Self> {
        conn.execute_batch(crate::schema::SCHEMA).map_err(se)?;
        Ok(Self { conn })
    }
}

impl ProfileStore for SqliteProfileStore {
    fn load_profiles(&self) -> Result<Vec<Profile>> {
        let mut stmt = self
            .conn
            .prepare("SELECT name, embeddings FROM profiles ORDER BY name")
            .map_err(se)?;
        let rows = stmt
            .query_map([], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?))
            })
            .map_err(se)?;
        let mut out = Vec::new();
        for row in rows {
            let (name, blob) = row.map_err(se)?;
            out.push(Profile {
                name,
                embeddings: decode(&blob)?,
            });
        }
        Ok(out)
    }

    fn upsert_profile(&self, name: &str, embeddings: &[Embedding]) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO profiles(name, embeddings) VALUES(?1, ?2)
                 ON CONFLICT(name) DO UPDATE SET embeddings = excluded.embeddings",
                rusqlite::params![name, encode(embeddings)],
            )
            .map_err(se)?;
        Ok(())
    }
}

fn se(e: rusqlite::Error) -> Error {
    Error::Store(e.to_string())
}

// BLOB layout: a sequence of [u32 dim little-endian][dim * f32 little-endian].
fn encode(embs: &[Embedding]) -> Vec<u8> {
    let mut buf = Vec::new();
    for e in embs {
        buf.extend_from_slice(&(e.0.len() as u32).to_le_bytes());
        for x in &e.0 {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    buf
}

fn decode(buf: &[u8]) -> Result<Vec<Embedding>> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < buf.len() {
        if i + 4 > buf.len() {
            return Err(Error::Store("truncated embedding header".into()));
        }
        let dim = u32::from_le_bytes(buf[i..i + 4].try_into().unwrap()) as usize;
        i += 4;
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            if i + 4 > buf.len() {
                return Err(Error::Store("truncated embedding body".into()));
            }
            v.push(f32::from_le_bytes(buf[i..i + 4].try_into().unwrap()));
            i += 4;
        }
        out.push(Embedding(v));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hearable_core::{Embedding, ProfileStore};

    #[test]
    fn upsert_then_load_roundtrips_embeddings() {
        let store = SqliteProfileStore::open_in_memory().unwrap();
        let embs = vec![
            Embedding(vec![0.1, 0.2, 0.3]),
            Embedding(vec![-1.0, 0.5, 2.0]),
        ];
        store.upsert_profile("Mom", &embs).unwrap();
        let loaded = store.load_profiles().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].name, "Mom");
        assert_eq!(loaded[0].embeddings, embs);
    }

    #[test]
    fn upsert_same_name_replaces() {
        let store = SqliteProfileStore::open_in_memory().unwrap();
        store
            .upsert_profile("Tom", &[Embedding(vec![1.0])])
            .unwrap();
        store
            .upsert_profile("Tom", &[Embedding(vec![2.0]), Embedding(vec![3.0])])
            .unwrap();
        let loaded = store.load_profiles().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].embeddings.len(), 2);
    }

    #[test]
    fn empty_store_loads_nothing() {
        let store = SqliteProfileStore::open_in_memory().unwrap();
        assert!(store.load_profiles().unwrap().is_empty());
    }
}
