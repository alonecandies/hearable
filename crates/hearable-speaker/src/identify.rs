use hearable_core::{ClusterId, Embedding, Identifier, Profile, Result, SpeakerLabel};

/// Tuning for online leader clustering (defaults per spec §4.4).
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    /// Cosine similarity required to join an existing cluster for a normal-length utterance.
    pub threshold: f32,
    /// Relaxed threshold for short utterances (less reliable embeddings).
    pub short_threshold: f32,
    /// Utterances shorter than this (seconds) use `short_threshold`.
    pub short_secs: f32,
    /// Exponential-moving-average weight when updating a matched (unnamed) centroid.
    pub ema_alpha: f32,
    /// Cap on simultaneously-tracked clusters; when full, the least-recently-seen *unnamed*
    /// cluster is evicted so an always-on session doesn't grow memory/scan-time without bound.
    pub max_clusters: usize,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        // Tuned from a first real run: 0.70 was far too strict — short, real-world
        // utterances never cleared it, so the same speaker spawned a new cluster each turn.
        // 0.5 matches sherpa's own example threshold; short utterances relax further.
        Self {
            threshold: 0.50,
            // Real-run tuning: short single words give noisy embeddings that falsely split one
            // speaker into several. Relax the short-utterance threshold and treat more
            // utterances as "short" so they prefer joining an existing cluster.
            short_threshold: 0.35,
            short_secs: 3.0,
            ema_alpha: 0.05,
            max_clusters: 64,
        }
    }
}

struct Cluster {
    id: ClusterId,
    centroid: Embedding,
    name: Option<String>,
    /// Monotonic tick of the last utterance assigned here (for LRU eviction of unnamed clusters).
    last_seen: u64,
}

/// Online (incremental) speaker identifier: cosine leader clustering with known-profile
/// matching. O(K) per utterance over the current cluster set; no batch re-clustering.
pub struct LeaderClusterIdentifier {
    cfg: ClusterConfig,
    clusters: Vec<Cluster>,
    next_id: u64,
    tick: u64,
}

impl LeaderClusterIdentifier {
    /// Build an identifier, seeding one named cluster per known profile (centroid = mean of
    /// the profile's enrolment embeddings).
    pub fn new(cfg: ClusterConfig, profiles: Vec<Profile>) -> Self {
        let mut me = Self {
            cfg,
            clusters: Vec::new(),
            next_id: 0,
            tick: 0,
        };
        for p in profiles {
            if let Some(centroid) = mean(&p.embeddings) {
                let id = me.alloc_id();
                me.clusters.push(Cluster {
                    id,
                    centroid,
                    name: Some(p.name),
                    last_seen: 0,
                });
            }
        }
        me
    }

    fn alloc_id(&mut self) -> ClusterId {
        let id = ClusterId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Number of clusters currently tracked (named profiles + active unknown speakers).
    pub fn cluster_count(&self) -> usize {
        self.clusters.len()
    }

    /// The current centroid of a cluster, for persisting a promoted profile. `None` if the
    /// cluster id is unknown.
    pub fn centroid_of(&self, cluster: ClusterId) -> Option<Embedding> {
        self.clusters
            .iter()
            .find(|c| c.id == cluster)
            .map(|c| c.centroid.clone())
    }

    fn identify_at(&mut self, e: &Embedding, thr: f32) -> SpeakerLabel {
        self.tick += 1;
        let now = self.tick;

        let mut best: Option<(usize, f32)> = None;
        for (i, c) in self.clusters.iter().enumerate() {
            let s = c.centroid.cosine(e);
            match best {
                Some((_, bs)) if s <= bs => {}
                _ => best = Some((i, s)),
            }
        }
        match best {
            Some((i, s)) if s >= thr => {
                self.clusters[i].last_seen = now;
                // Anchor named/enrolled profiles: only let unnamed clusters drift via EMA, so a
                // recognized person's centroid doesn't wander toward whoever speaks near them.
                if self.clusters[i].name.is_none() {
                    ema_update(&mut self.clusters[i].centroid, e, self.cfg.ema_alpha);
                }
                let c = &self.clusters[i];
                match &c.name {
                    Some(name) => SpeakerLabel::Known {
                        name: name.clone(),
                        score: s,
                    },
                    None => SpeakerLabel::Unknown {
                        cluster_id: c.id,
                        score: s,
                    },
                }
            }
            _ => {
                self.evict_if_full();
                let id = self.alloc_id();
                self.clusters.push(Cluster {
                    id,
                    centroid: e.clone(),
                    name: None,
                    last_seen: now,
                });
                // A freshly-seeded cluster is, by definition, a perfect match for its own seed.
                SpeakerLabel::Unknown {
                    cluster_id: id,
                    score: 1.0,
                }
            }
        }
    }

    /// If at capacity, evict the least-recently-seen *unnamed* cluster (named profiles are
    /// never evicted). If every cluster is named we simply allow growth (named set is bounded
    /// by user actions).
    fn evict_if_full(&mut self) {
        if self.clusters.len() < self.cfg.max_clusters {
            return;
        }
        if let Some(idx) = self
            .clusters
            .iter()
            .enumerate()
            .filter(|(_, c)| c.name.is_none())
            .min_by_key(|(_, c)| c.last_seen)
            .map(|(i, _)| i)
        {
            self.clusters.remove(idx);
        }
    }
}

impl Identifier for LeaderClusterIdentifier {
    fn identify(&mut self, e: &Embedding) -> SpeakerLabel {
        let thr = self.cfg.threshold;
        self.identify_at(e, thr)
    }

    fn identify_with_duration(&mut self, e: &Embedding, duration_secs: f32) -> SpeakerLabel {
        let thr = if duration_secs < self.cfg.short_secs {
            self.cfg.short_threshold
        } else {
            self.cfg.threshold
        };
        self.identify_at(e, thr)
    }

    fn promote(&mut self, cluster: ClusterId, name: &str) -> Result<()> {
        for c in &mut self.clusters {
            if c.id == cluster {
                c.name = Some(name.to_string());
                return Ok(());
            }
        }
        Err(hearable_core::Error::Store(format!(
            "no cluster {cluster:?} to promote"
        )))
    }
}

fn ema_update(centroid: &mut Embedding, e: &Embedding, alpha: f32) {
    if centroid.0.len() != e.0.len() {
        return;
    }
    for (c, x) in centroid.0.iter_mut().zip(&e.0) {
        *c = (1.0 - alpha) * *c + alpha * *x;
    }
}

fn mean(es: &[Embedding]) -> Option<Embedding> {
    let dim = es.first()?.0.len();
    // Only average embeddings of the expected dimension (a corrupt/mixed-dim profile store
    // would otherwise silently produce a truncated, wrong centroid).
    let mut acc = vec![0.0f32; dim];
    let mut n = 0u32;
    for e in es.iter().filter(|e| e.0.len() == dim) {
        for (a, x) in acc.iter_mut().zip(&e.0) {
            *a += x;
        }
        n += 1;
    }
    if n == 0 {
        return None;
    }
    for a in &mut acc {
        *a /= n as f32;
    }
    Some(Embedding(acc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hearable_core::{ClusterId, Embedding, Identifier, Profile, SpeakerLabel};

    fn id_default() -> LeaderClusterIdentifier {
        LeaderClusterIdentifier::new(ClusterConfig::default(), vec![])
    }

    fn cluster_of(l: &SpeakerLabel) -> ClusterId {
        match l {
            SpeakerLabel::Unknown { cluster_id, .. } => *cluster_id,
            _ => panic!("not unknown"),
        }
    }

    #[test]
    fn first_embedding_creates_unknown_cluster() {
        let mut id = id_default();
        match id.identify(&Embedding(vec![1.0, 0.0, 0.0])) {
            SpeakerLabel::Unknown { cluster_id, .. } => assert_eq!(cluster_id.0, 0),
            _ => panic!("expected Unknown"),
        }
    }

    #[test]
    fn similar_embedding_joins_same_cluster() {
        let mut id = id_default();
        let a = id.identify(&Embedding(vec![1.0, 0.0, 0.0]));
        let b = id.identify(&Embedding(vec![0.98, 0.02, 0.0]));
        assert_eq!(cluster_of(&a), cluster_of(&b));
    }

    #[test]
    fn dissimilar_embedding_creates_new_cluster() {
        let mut id = id_default();
        let a = id.identify(&Embedding(vec![1.0, 0.0, 0.0]));
        let b = id.identify(&Embedding(vec![0.0, 1.0, 0.0]));
        assert_ne!(cluster_of(&a), cluster_of(&b));
    }

    #[test]
    fn known_profile_matches_by_name() {
        let prof = Profile {
            name: "Mom".into(),
            embeddings: vec![Embedding(vec![1.0, 0.0, 0.0])],
        };
        let mut id = LeaderClusterIdentifier::new(ClusterConfig::default(), vec![prof]);
        match id.identify(&Embedding(vec![0.97, 0.0, 0.0])) {
            SpeakerLabel::Known { name, .. } => assert_eq!(name, "Mom"),
            other => panic!("expected Known(Mom), got {other:?}"),
        }
    }

    #[test]
    fn promote_makes_future_utterances_known() {
        let mut id = id_default();
        let label = id.identify(&Embedding(vec![1.0, 0.0, 0.0]));
        let cid = match label {
            SpeakerLabel::Unknown { cluster_id, .. } => cluster_id,
            _ => unreachable!(),
        };
        id.promote(cid, "Tom").unwrap();
        match id.identify(&Embedding(vec![0.99, 0.0, 0.0])) {
            SpeakerLabel::Known { name, .. } => assert_eq!(name, "Tom"),
            other => panic!("expected Known(Tom), got {other:?}"),
        }
    }

    #[test]
    fn promote_unknown_cluster_errors() {
        let mut id = id_default();
        assert!(id.promote(ClusterId(999), "Nobody").is_err());
    }

    #[test]
    fn cluster_count_is_capped_with_lru_eviction() {
        let cfg = ClusterConfig {
            threshold: 0.99,
            short_threshold: 0.99,
            short_secs: 0.0,
            ema_alpha: 0.05,
            max_clusters: 3,
        };
        let mut id = LeaderClusterIdentifier::new(cfg, vec![]);
        // Four near-orthogonal embeddings, high threshold -> four new-cluster attempts.
        for v in [
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ] {
            id.identify(&Embedding(v));
        }
        assert_eq!(
            id.cluster_count(),
            3,
            "cluster count must be capped at max_clusters"
        );
    }

    #[test]
    fn named_cluster_centroid_is_anchored() {
        let mut id = id_default();
        let cid = match id.identify(&Embedding(vec![1.0, 0.0, 0.0])) {
            SpeakerLabel::Unknown { cluster_id, .. } => cluster_id,
            _ => unreachable!(),
        };
        id.promote(cid, "Mom").unwrap();
        let before = id.centroid_of(cid).unwrap();
        // A nearby embedding matches "Mom" but, being named, must not drift her centroid.
        id.identify(&Embedding(vec![0.9, 0.1, 0.0]));
        assert_eq!(
            id.centroid_of(cid).unwrap(),
            before,
            "named centroid must stay anchored"
        );
    }

    #[test]
    fn new_cluster_reports_full_self_confidence() {
        let mut id = id_default();
        match id.identify(&Embedding(vec![1.0, 0.0, 0.0])) {
            SpeakerLabel::Unknown { score, .. } => assert_eq!(score, 1.0),
            _ => unreachable!(),
        }
    }

    #[test]
    fn centroid_of_returns_first_member_then_none_for_unknown() {
        let mut id = id_default();
        let e = Embedding(vec![1.0, 0.0, 0.0]);
        id.identify(&e);
        assert_eq!(id.centroid_of(ClusterId(0)), Some(e));
        assert_eq!(id.centroid_of(ClusterId(42)), None);
    }

    #[test]
    fn short_utterance_uses_relaxed_threshold() {
        // Explicit config so this stays meaningful regardless of the shipped defaults.
        // Cosine(a, b) ≈ 0.66 lands between short_threshold (0.62) and threshold (0.70):
        // a long utterance starts a new cluster, a short one joins the existing one.
        let cfg = ClusterConfig {
            threshold: 0.70,
            short_threshold: 0.62,
            short_secs: 1.5,
            ema_alpha: 0.05,
            max_clusters: 64,
        };
        let a = Embedding(vec![1.0, 0.0]);
        let b = Embedding(vec![0.66, 0.75]);

        let mut long_id = LeaderClusterIdentifier::new(cfg.clone(), vec![]);
        long_id.identify_with_duration(&a, 3.0);
        let long_label = long_id.identify_with_duration(&b, 3.0);
        assert_eq!(cluster_of(&long_label).0, 1, "long utterance: new cluster");

        let mut short_id = LeaderClusterIdentifier::new(cfg, vec![]);
        short_id.identify_with_duration(&a, 0.5);
        let short_label = short_id.identify_with_duration(&b, 0.5);
        assert_eq!(
            cluster_of(&short_label).0,
            0,
            "short utterance: joins cluster 0"
        );
    }

    proptest::proptest! {
        #[test]
        fn identify_never_panics(v in proptest::collection::vec(-10f32..10.0, 1..64)) {
            let mut id = id_default();
            let _ = id.identify(&Embedding(v));
        }
    }
}
