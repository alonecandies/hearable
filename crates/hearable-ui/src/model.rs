//! Pure caption-display view-model (no egui dependency, fully unit-testable).
//!
//! Implements the accessibility presentation rules from the design spec: a rolling window of
//! recent lines, a committed/partial split, and colorblind-safe per-speaker colors from the
//! Okabe-Ito palette. The egui layer (the `ui` feature) only renders what this produces.

use hearable_core::{CaptionEvent, SpeakerLabel};
use std::collections::VecDeque;

/// Five colorblind-safe speaker colors (Okabe-Ito subset; vermillion omitted to keep
/// distinctness across the common color-vision deficiencies).
pub const SPEAKER_PALETTE: [[u8; 3]; 5] = [
    [86, 180, 233],  // sky blue
    [230, 159, 0],   // orange
    [0, 158, 115],   // bluish green
    [0, 114, 178],   // blue
    [204, 121, 167], // reddish purple
];

/// One renderable caption line.
#[derive(Debug, Clone, PartialEq)]
pub struct DisplayLine {
    pub speaker: String,
    pub text: String,
    pub color: [u8; 3],
    pub is_partial: bool,
    /// `Some(id)` for an as-yet-unnamed cluster (offer a "name" affordance); `None` if known.
    pub cluster_id: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SpeakerKey {
    Known(String),
    Unknown(u64),
}

/// Maintains the rolling caption display state.
pub struct CaptionView {
    max_lines: usize,
    committed: VecDeque<DisplayLine>,
    partial: Option<DisplayLine>,
    order: Vec<SpeakerKey>,
}

impl CaptionView {
    pub fn new(max_lines: usize) -> Self {
        Self {
            max_lines: max_lines.max(1),
            committed: VecDeque::new(),
            partial: None,
            order: Vec::new(),
        }
    }

    fn key(speaker: &SpeakerLabel) -> (SpeakerKey, String) {
        match speaker {
            SpeakerLabel::Known { name, .. } => (SpeakerKey::Known(name.clone()), name.clone()),
            SpeakerLabel::Unknown { cluster_id, .. } => (
                SpeakerKey::Unknown(cluster_id.0),
                format!("Speaker {}", cluster_id.0 + 1),
            ),
        }
    }

    /// Stable color slot per speaker, assigned in first-appearance order (wraps past 5).
    fn color_index(&mut self, key: &SpeakerKey) -> usize {
        match self.order.iter().position(|k| k == key) {
            Some(i) => i % SPEAKER_PALETTE.len(),
            None => {
                self.order.push(key.clone());
                (self.order.len() - 1) % SPEAKER_PALETTE.len()
            }
        }
    }

    /// Feed a caption event: final events commit a line; non-final events update the partial tail.
    pub fn push(&mut self, ev: &CaptionEvent) {
        let (key, speaker) = Self::key(&ev.speaker);
        let color = SPEAKER_PALETTE[self.color_index(&key)];
        let cluster_id = match &ev.speaker {
            SpeakerLabel::Unknown { cluster_id, .. } => Some(cluster_id.0),
            SpeakerLabel::Known { .. } => None,
        };
        let line = DisplayLine {
            speaker,
            text: ev.text.clone(),
            color,
            is_partial: !ev.is_final,
            cluster_id,
        };
        if ev.is_final {
            self.partial = None;
            self.committed.push_back(line);
            while self.committed.len() > self.max_lines {
                self.committed.pop_front();
            }
        } else {
            self.partial = Some(line);
        }
    }

    /// Visible lines (committed, plus any partial tail), oldest first, capped to `max_lines`.
    pub fn visible(&self) -> Vec<DisplayLine> {
        let mut v: Vec<DisplayLine> = self.committed.iter().cloned().collect();
        if let Some(p) = &self.partial {
            v.push(p.clone());
        }
        let n = v.len();
        if n > self.max_lines {
            v.split_off(n - self.max_lines)
        } else {
            v
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hearable_core::{ClusterId, SpeakerLabel};

    fn ev(text: &str, speaker: SpeakerLabel, is_final: bool) -> CaptionEvent {
        CaptionEvent {
            utt_id: hearable_core::UtteranceId(0),
            text: text.into(),
            lang: None,
            speaker,
            t0_ms: 0,
            t1_ms: 0,
            is_final,
        }
    }

    fn unknown(c: u64) -> SpeakerLabel {
        SpeakerLabel::Unknown {
            cluster_id: ClusterId(c),
            score: 0.9,
        }
    }

    #[test]
    fn distinct_speakers_get_distinct_colors_same_speaker_stable() {
        let mut v = CaptionView::new(5);
        v.push(&ev("a", unknown(0), true));
        v.push(&ev("b", unknown(1), true));
        v.push(&ev("c", unknown(0), true));
        let lines = v.visible();
        assert_eq!(lines[0].color, SPEAKER_PALETTE[0]);
        assert_eq!(lines[1].color, SPEAKER_PALETTE[1]);
        assert_eq!(
            lines[2].color, SPEAKER_PALETTE[0],
            "speaker 0 keeps its color"
        );
    }

    #[test]
    fn unknown_label_is_one_based() {
        let mut v = CaptionView::new(3);
        v.push(&ev("hi", unknown(0), true));
        assert_eq!(v.visible()[0].speaker, "Speaker 1");
    }

    #[test]
    fn unknown_lines_carry_cluster_id_known_do_not() {
        let mut v = CaptionView::new(3);
        v.push(&ev("a", unknown(2), true));
        v.push(&ev(
            "b",
            SpeakerLabel::Known {
                name: "Mom".into(),
                score: 0.9,
            },
            true,
        ));
        let lines = v.visible();
        assert_eq!(lines[0].cluster_id, Some(2));
        assert_eq!(lines[1].cluster_id, None);
    }

    #[test]
    fn known_speaker_uses_name() {
        let mut v = CaptionView::new(3);
        v.push(&ev(
            "hi",
            SpeakerLabel::Known {
                name: "Mom".into(),
                score: 0.95,
            },
            true,
        ));
        assert_eq!(v.visible()[0].speaker, "Mom");
    }

    #[test]
    fn committed_lines_capped_at_max() {
        let mut v = CaptionView::new(3);
        for i in 0..6 {
            v.push(&ev(&format!("line {i}"), unknown(0), true));
        }
        let lines = v.visible();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0].text, "line 3");
        assert_eq!(lines[2].text, "line 5");
    }

    #[test]
    fn partial_is_shown_then_replaced_by_final() {
        let mut v = CaptionView::new(3);
        v.push(&ev("hel", unknown(0), false));
        let mid = v.visible();
        assert_eq!(mid.len(), 1);
        assert!(mid[0].is_partial);
        v.push(&ev("hello", unknown(0), true));
        let done = v.visible();
        assert_eq!(done.len(), 1);
        assert!(!done[0].is_partial);
        assert_eq!(done[0].text, "hello");
    }
}
