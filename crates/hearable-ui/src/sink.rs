//! A [`CaptionSink`] that forwards caption events to the overlay's UI thread over a channel.

use hearable_core::{CaptionEvent, CaptionSink};
use std::sync::mpsc::Sender;

/// Sends caption events to the overlay (or any consumer) via an `mpsc` channel.
pub struct ChannelCaptionSink {
    tx: Sender<CaptionEvent>,
}

impl ChannelCaptionSink {
    pub fn new(tx: Sender<CaptionEvent>) -> Self {
        Self { tx }
    }
}

impl CaptionSink for ChannelCaptionSink {
    fn emit(&mut self, ev: CaptionEvent) {
        // The UI may have closed; dropping events is acceptable (the pipeline must not block).
        let _ = self.tx.send(ev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hearable_core::{ClusterId, SpeakerLabel, UtteranceId};
    use std::sync::mpsc;

    #[test]
    fn forwards_events_to_receiver() {
        let (tx, rx) = mpsc::channel();
        let mut sink = ChannelCaptionSink::new(tx);
        sink.emit(CaptionEvent {
            utt_id: UtteranceId(1),
            text: "hi".into(),
            lang: None,
            speaker: SpeakerLabel::Unknown {
                cluster_id: ClusterId(0),
                score: 1.0,
            },
            t0_ms: 0,
            t1_ms: 0,
            is_final: true,
        });
        assert_eq!(rx.recv().unwrap().text, "hi");
    }

    #[test]
    fn emit_after_receiver_dropped_does_not_panic() {
        let (tx, rx) = mpsc::channel();
        let mut sink = ChannelCaptionSink::new(tx);
        drop(rx);
        sink.emit(CaptionEvent {
            utt_id: UtteranceId(1),
            text: "x".into(),
            lang: None,
            speaker: SpeakerLabel::Unknown {
                cluster_id: ClusterId(0),
                score: 1.0,
            },
            t0_ms: 0,
            t1_ms: 0,
            is_final: true,
        });
    }
}
