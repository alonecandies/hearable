//! Test utilities shared across the workspace's test suites.

use crate::{CaptionEvent, CaptionSink};

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
