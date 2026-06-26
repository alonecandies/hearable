//! Test utilities shared across the workspace's test suites.

use crate::{CaptionEvent, CaptionSink};
use std::sync::{Arc, Mutex};

/// A [`CaptionSink`] that records every emitted event for assertions.
#[derive(Default)]
pub struct BufferCaptionSink {
    events: Vec<CaptionEvent>,
}

impl BufferCaptionSink {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn events(&self) -> &[CaptionEvent] {
        &self.events
    }
}

impl CaptionSink for BufferCaptionSink {
    fn emit(&mut self, ev: CaptionEvent) {
        self.events.push(ev);
    }
}

/// A thread-safe, cloneable [`CaptionSink`] for testing concurrent pipelines: the worker
/// thread holds one clone and emits into the shared buffer; the test holds another clone to
/// read what was produced after joining.
#[derive(Clone, Default)]
pub struct SharedCaptionSink {
    inner: Arc<Mutex<Vec<CaptionEvent>>>,
}

impl SharedCaptionSink {
    pub fn new() -> Self {
        Self::default()
    }

    /// A copy of the events emitted so far.
    pub fn snapshot(&self) -> Vec<CaptionEvent> {
        self.inner.lock().unwrap().clone()
    }
}

impl CaptionSink for SharedCaptionSink {
    fn emit(&mut self, ev: CaptionEvent) {
        self.inner.lock().unwrap().push(ev);
    }
}
