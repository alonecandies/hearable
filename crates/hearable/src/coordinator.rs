//! Threaded realtime coordinator (spec §3.4).
//!
//! Capture + VAD run on one thread; ASR + embedding + identification run on an inference
//! thread. They are decoupled by a bounded queue with a **drop-oldest** overload policy: an
//! always-on listener must never block the capture path, and when the inference thread falls
//! behind we discard the *oldest* pending utterance so the freshest speech is captioned.
//!
//! Phase 1 keeps ASR and embedding sequential within the inference thread; splitting them
//! into parallel workers (spec §3.4) is a latency optimization deferred to a later iteration.

use hearable_core::{
    AsrEngine, AudioSource, CaptionEvent, CaptionSink, EmbeddingExtractor, Identifier, Utterance,
    VadSegmenter,
};
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

/// Shared, mutable identifier: the inference thread reads it every utterance while the UI
/// thread may promote a cluster to a named profile concurrently.
pub type SharedIdentifier<I> = Arc<Mutex<I>>;

struct QueueInner {
    q: VecDeque<Utterance>,
    closed: bool,
    dropped: u64,
}

/// A bounded utterance queue with a drop-oldest overload policy.
struct DropOldestQueue {
    cap: usize,
    inner: Mutex<QueueInner>,
    cv: Condvar,
}

impl DropOldestQueue {
    fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            inner: Mutex::new(QueueInner {
                q: VecDeque::new(),
                closed: false,
                dropped: 0,
            }),
            cv: Condvar::new(),
        }
    }

    /// Enqueue, discarding the oldest item first if the queue is at capacity. Never blocks.
    fn push(&self, item: Utterance) {
        {
            let mut g = self.inner.lock().unwrap();
            if g.q.len() >= self.cap {
                g.q.pop_front();
                g.dropped += 1;
            }
            g.q.push_back(item);
        }
        self.cv.notify_one();
    }

    /// Block until an item is available, or return `None` once closed and drained.
    fn recv(&self) -> Option<Utterance> {
        let mut g = self.inner.lock().unwrap();
        loop {
            if let Some(x) = g.q.pop_front() {
                return Some(x);
            }
            if g.closed {
                return None;
            }
            g = self.cv.wait(g).unwrap();
        }
    }

    /// Signal that no more items will be pushed; wakes a blocked receiver.
    fn close(&self) {
        self.inner.lock().unwrap().closed = true;
        self.cv.notify_all();
    }

    fn dropped(&self) -> u64 {
        self.inner.lock().unwrap().dropped
    }
}

/// Result of a pipeline run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineOutcome {
    /// Utterances discarded by the drop-oldest policy under overload.
    pub dropped: u64,
}

/// Run the realtime pipeline across two threads, blocking until the audio source is
/// exhausted and all queued utterances have been processed.
#[allow(clippy::too_many_arguments)]
pub fn run_threaded<A, V, G, E, I, S>(
    mut source: A,
    mut segmenter: V,
    mut asr: G,
    mut embedder: E,
    identifier: SharedIdentifier<I>,
    mut sink: S,
    capacity: usize,
) -> PipelineOutcome
where
    A: AudioSource + Send + 'static,
    V: VadSegmenter + Send + 'static,
    G: AsrEngine + Send + 'static,
    E: EmbeddingExtractor + Send + 'static,
    I: Identifier + Send + 'static,
    S: CaptionSink + Send + 'static,
{
    let queue = Arc::new(DropOldestQueue::new(capacity));

    let prod_q = Arc::clone(&queue);
    let producer = thread::spawn(move || {
        let q = &prod_q;
        let _ = source.start(&mut |frame| {
            for utt in segmenter.push(frame) {
                q.push(utt);
            }
        });
        for utt in segmenter.flush() {
            q.push(utt);
        }
        q.close();
    });

    let cons_q = Arc::clone(&queue);
    let consumer = thread::spawn(move || {
        while let Some(utt) = cons_q.recv() {
            let tr = match asr.transcribe(&utt) {
                Ok(t) => t,
                Err(_) => continue,
            };
            let emb = match embedder.embed(&utt) {
                Ok(e) => e,
                Err(_) => continue,
            };
            // Brief lock: the UI thread may promote a cluster between utterances.
            let speaker = identifier.lock().unwrap().identify(&emb);
            sink.emit(CaptionEvent {
                utt_id: utt.id,
                text: tr.text,
                lang: tr.lang,
                speaker,
                t0_ms: utt.t0_ms,
                t1_ms: utt.t1_ms,
                is_final: tr.is_final,
            });
        }
    });

    producer.join().expect("capture/VAD thread panicked");
    consumer.join().expect("inference thread panicked");
    PipelineOutcome {
        dropped: queue.dropped(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hearable_core::UtteranceId;

    fn utt(id: u64) -> Utterance {
        Utterance {
            id: UtteranceId(id),
            pcm16k: vec![],
            t0_ms: 0,
            t1_ms: 0,
        }
    }

    #[test]
    fn drop_oldest_discards_front_when_full() {
        let q = DropOldestQueue::new(2);
        q.push(utt(0));
        q.push(utt(1));
        q.push(utt(2)); // capacity 2 -> oldest (id 0) dropped
        assert_eq!(q.dropped(), 1);
        assert_eq!(q.recv().unwrap().id.0, 1);
        assert_eq!(q.recv().unwrap().id.0, 2);
        q.close();
        assert!(q.recv().is_none());
    }

    #[test]
    fn recv_returns_none_after_close_when_empty() {
        let q = DropOldestQueue::new(4);
        q.close();
        assert!(q.recv().is_none());
    }
}
