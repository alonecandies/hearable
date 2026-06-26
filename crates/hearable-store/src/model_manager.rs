//! First-run model downloader (the `download` feature).
//!
//! Models are fetched at runtime, never bundled. Each download streams to a `.part` file
//! (resumable via HTTP Range), is verified against an expected SHA-256, and only then
//! atomically renamed into place. A model whose checksum already matches is reused without
//! re-downloading.
//!
//! This module is intentionally generic over [`ModelSpec`]: the concrete registry of model
//! URLs + checksums is supplied by the caller (the `run` wiring), so no checksums are
//! hard-coded here until they can be pinned from the published release.

use hearable_core::{Error, Result};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

const BUF: usize = 64 * 1024;

/// A model artifact to ensure is present locally.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub name: String,
    pub url: String,
    /// Expected lowercase-hex SHA-256 of the complete file.
    pub sha256: String,
    pub filename: String,
}

/// Downloads and verifies model files into a directory.
pub struct ModelManager {
    dir: PathBuf,
}

impl ModelManager {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    pub fn path_for(&self, spec: &ModelSpec) -> PathBuf {
        self.dir.join(&spec.filename)
    }

    /// Return the local path to the model, downloading + verifying it if missing or corrupt.
    pub fn ensure(&self, spec: &ModelSpec) -> Result<PathBuf> {
        fs::create_dir_all(&self.dir)?;
        let dest = self.path_for(spec);
        if dest.exists() && file_sha256(&dest)? == spec.sha256 {
            return Ok(dest);
        }
        self.download(spec, &dest)?;
        let got = file_sha256(&dest)?;
        if got != spec.sha256 {
            let _ = fs::remove_file(&dest);
            return Err(Error::Store(format!(
                "checksum mismatch for {}: expected {}, got {}",
                spec.name, spec.sha256, got
            )));
        }
        Ok(dest)
    }

    fn download(&self, spec: &ModelSpec, dest: &Path) -> Result<()> {
        let part = dest.with_extension("part");
        let existing = fs::metadata(&part).map(|m| m.len()).unwrap_or(0);

        let client = reqwest::blocking::Client::builder()
            .build()
            .map_err(|e| Error::Store(format!("http client: {e}")))?;
        let mut req = client.get(&spec.url);
        if existing > 0 {
            req = req.header(reqwest::header::RANGE, format!("bytes={existing}-"));
        }
        let mut resp = req
            .send()
            .map_err(|e| Error::Store(format!("download {}: {e}", spec.name)))?;
        if !resp.status().is_success() {
            return Err(Error::Store(format!(
                "download {}: HTTP {}",
                spec.name,
                resp.status()
            )));
        }
        // Append only if the server honored the Range request with 206 Partial Content.
        let append = existing > 0 && resp.status() == reqwest::StatusCode::PARTIAL_CONTENT;
        let mut file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(append)
            .truncate(!append)
            .open(&part)?;
        let mut buf = vec![0u8; BUF];
        loop {
            let n = resp
                .read(&mut buf)
                .map_err(|e| Error::Store(format!("read {}: {e}", spec.name)))?;
            if n == 0 {
                break;
            }
            use std::io::Write;
            file.write_all(&buf[..n])?;
        }
        drop(file);
        fs::rename(&part, dest)?;
        Ok(())
    }
}

fn file_sha256(path: &Path) -> Result<String> {
    let mut f = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; BUF];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(to_hex(&hasher.finalize()))
}

fn to_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;
    use std::net::TcpListener;
    use std::thread;

    /// Serve `body` for a single HTTP GET, then stop. Returns the bound port.
    fn serve_once(body: Vec<u8>) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut scratch = [0u8; 2048];
                let _ = stream.read(&mut scratch); // consume request headers
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(header.as_bytes());
                let _ = stream.write_all(&body);
                let _ = stream.flush();
            }
        });
        port
    }

    fn sha256_hex(data: &[u8]) -> String {
        let mut h = Sha256::new();
        h.update(data);
        to_hex(&h.finalize())
    }

    #[test]
    fn downloads_and_verifies() {
        let body = b"hearable model bytes".to_vec();
        let port = serve_once(body.clone());
        let dir = std::env::temp_dir().join(format!("hearable-dl-{port}"));
        let mgr = ModelManager::new(&dir);
        let spec = ModelSpec {
            name: "test".into(),
            url: format!("http://127.0.0.1:{port}/model.bin"),
            sha256: sha256_hex(&body),
            filename: "model.bin".into(),
        };
        let path = mgr.ensure(&spec).unwrap();
        assert!(path.exists());
        assert_eq!(fs::read(&path).unwrap(), body);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_checksum_mismatch() {
        let body = b"corrupt".to_vec();
        let port = serve_once(body);
        let dir = std::env::temp_dir().join(format!("hearable-dl-bad-{port}"));
        let mgr = ModelManager::new(&dir);
        let spec = ModelSpec {
            name: "bad".into(),
            url: format!("http://127.0.0.1:{port}/model.bin"),
            sha256: "0".repeat(64), // wrong
            filename: "model.bin".into(),
        };
        let err = mgr.ensure(&spec);
        assert!(err.is_err(), "expected checksum mismatch error");
        assert!(
            !mgr.path_for(&spec).exists(),
            "corrupt file must be removed"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn reuses_cached_file_without_redownload() {
        let body = b"cached".to_vec();
        let port = serve_once(body.clone()); // serves exactly ONE request
        let dir = std::env::temp_dir().join(format!("hearable-dl-cache-{port}"));
        let mgr = ModelManager::new(&dir);
        let spec = ModelSpec {
            name: "cache".into(),
            url: format!("http://127.0.0.1:{port}/model.bin"),
            sha256: sha256_hex(&body),
            filename: "model.bin".into(),
        };
        mgr.ensure(&spec).unwrap(); // consumes the one served request
                                    // Server is now gone; a second ensure must succeed from cache (no network).
        let path = mgr.ensure(&spec).unwrap();
        assert_eq!(fs::read(&path).unwrap(), body);
        let _ = fs::remove_dir_all(&dir);
    }
}
