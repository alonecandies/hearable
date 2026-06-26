//! The `hearable` application library: pipeline orchestration that wires the trait-based
//! components together. The binary (`main.rs`) is a thin shell over this library, and the
//! integration tests link against it.

pub mod pipeline;

pub use pipeline::run_pipeline;
