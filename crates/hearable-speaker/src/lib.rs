//! Speaker embeddings, online clustering, and cross-session identification.
//!
//! Phase 0 ships the pure-logic [`LeaderClusterIdentifier`]. The sherpa-onnx-backed
//! [`hearable_core::EmbeddingExtractor`] implementation (ERes2NetV2) arrives in Phase 1.

pub mod identify;

pub use identify::{ClusterConfig, LeaderClusterIdentifier};
