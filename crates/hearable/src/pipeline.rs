use hearable_core::{
    AsrEngine, AudioSource, CaptionEvent, CaptionSink, EmbeddingExtractor, Identifier, Result,
    Utterance, VadSegmenter,
};

/// Synchronous reference pipeline (Phase 0): pull frames from `source`, segment them into
/// utterances, then for each utterance transcribe + embed + identify and emit a
/// [`CaptionEvent`]. Phase 1 replaces this with the threaded coordinator described in
/// spec §3.4 — the trait seams stay identical, so this serves as the executable contract.
pub fn run_pipeline<A, V, E, I, S>(
    mut source: A,
    mut segmenter: V,
    mut asr: impl AsrEngine,
    mut embedder: E,
    mut identifier: I,
    sink: &mut S,
) -> Result<()>
where
    A: AudioSource,
    V: VadSegmenter,
    E: EmbeddingExtractor,
    I: Identifier,
    S: CaptionSink,
{
    let mut utts: Vec<Utterance> = Vec::new();
    source.start(&mut |frame| utts.extend(segmenter.push(frame)))?;
    utts.extend(segmenter.flush());

    for utt in utts {
        let tr = asr.transcribe(&utt)?;
        let emb = embedder.embed(&utt)?;
        let speaker = identifier.identify_with_duration(&emb, utt.duration_secs());
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
    Ok(())
}
