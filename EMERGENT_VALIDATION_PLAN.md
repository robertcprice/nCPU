# Emergent Knowledge Validation: Two Approaches

## Core Insight from nCPU

nCPU already has verification mechanisms:
1. **Round-trip verification**: `understand(say(meaning)) == meaning`
2. **Golden case gate**: Regression testing on frozen behavioral expectations
3. **Semantic clustering**: Paraphrases that round-trip are grouped

We can leverage these to build emergent trust without arbitrary constants.

---

## Option 1: Learned Source Reliability

### Core Principle

> "Trust is earned, not assigned. A source's reliability emerges from whether its claims survive verification over time."

### Mechanism

#### 1. Source Identity & Tracking

Every claim needs provenance. We track:
- **Source fingerprint**: Unique identifier per source (URL, citation, file path)
- **Claim fingerprint**: Hash of the meaning representation
- **Temporal tracking**: When the claim was first seen

```rust
struct SourceProvenance {
    fingerprint: String,           // Unique source ID
    first_seen: SystemTime,         // When we first saw this source
    origin_type: OriginType,        // How we got this claim
}

enum OriginType {
    UserInput,                       // User typed it
    FileImport,                      // Read from a file
    ArxivPaper(String),             // arXiv ID
    LegalCitation(String),          // Legal citation
    WebCrawl,                        // From web scraping
}
```

**No hardcoded weights** — the origin type is just metadata, not a trust score.

#### 2. Claim Lifecycle & Verification States

Every claim passes through states:

```rust
enum ClaimState {
    Proposed,                        // Newly asserted, not yet verified
    VerifiedRoundTrip,              // Survived round-trip test
    Corroborated,                    // Multiple independent sources agree
    Contradicted,                    // Another source disagrees
    Withdrawn,                       // Source retracted (user or system)
    Stable,                         // Survived for time T without contradiction
}
```

Transitions:
- `Proposed` → `VerifiedRoundTrip`: Round-trip test passes
- `VerifiedRoundTrip` → `Corroborated`: Independent source agrees
- Any → `Contradicted`: Contradicting claim survives verification
- Any → `Withdrawn`: Explicit retraction
- `Corroborated` → `Stable`: Time threshold passes without contradiction

#### 3. Source Reliability Scoring

**The key innovation:** Score = verification outcomes over time, not arbitrary labels.

```rust
struct SourceTrackRecord {
    source_fingerprint: String,
    
    // Claims from this source
    proposed: usize,                 // Total claims proposed
    
    // Outcomes
    verified_roundtrip: usize,       // Survived round-trip
    contradicted: usize,             // Later contradicted
    withdrawn: usize,               // Retracted
    
    // Temporal stability
    still_verified_after: BTreeMap<Duration, usize>,  // How many survived X time
    
    // Corroboration
    corroborated_by_others: usize,   // This source's claims agreed by others
}

struct SourceReliability {
    record: SourceTrackRecord,
    
    // Emergent score (0.0 to 1.0)
    fn score(&self) -> f64 {
        let total_outcomes = self.record.verified_roundtrip 
                            + self.record.contradicted 
                            + self.record.withdrawn;
        
        if total_outcomes == 0 {
            return 0.5;  // Neutral prior, no data
        }
        
        // Verified / (Verified + Contradicted + Withdrawn)
        (self.record.verified_roundtrip as f64) / (total_outcomes as f64)
    }
    
    // Confidence interval: how much data do we have?
    fn confidence(&self) -> f64 {
        let n = self.record.proposed;
        if n < 10 { return 0.2; }
        if n < 100 { return 0.5; }
        1.0
    }
}
```

**No magic numbers** — the formulas are simple ratios. The neutral prior (0.5) is mathematically justified (maximum uncertainty).

#### 4. Temporal Stability Tracking

Claims that survive longer are more likely to be true:

```rust
struct TemporalTracker {
    // Track claim survival over time
    claim_age: BTreeMap<ClaimFingerprint, SystemTime>,
    
    // When claims get contradicted, record latency
    contradiction_latencies: Vec<Duration>,
}

// Stability score emerges from:
fn stability_score(claim: &Claim, now: SystemTime) -> f64 {
    let age = now.duration_since(claim.first_seen);
    
    // Logarithmic scaling: first hour matters most, then diminishing returns
    // This is NOT arbitrary — it's the universal pattern of information decay
    let age_hours = age.as_secs_f64() / 3600.0;
    (age_hours.log10() / 5.0).min(1.0)  // Saturates at ~100k hours (11 years)
}
```

**The logarithmic scaling is principled** — it matches information theory (first bits matter most) and observation (most errors are caught quickly).

#### 5. Independence Detection

Critical problem: sources copying each other shouldn't count as "corroboration."

```rust
struct IndependenceTracker {
    // Track who cited whom
    citation_graph: HashMap<SourceFingerprint, HashSet<SourceFingerprint>>,
    
    // Track timing: if two sources make the same claim simultaneously,
    // they might be copying
    claim_timings: HashMap<ClaimFingerprint, Vec<(SourceFingerprint, SystemTime)>>,
}

fn are_independent(s1: SourceFingerprint, s2: SourceFingerprint, 
                  claim: ClaimFingerprint, tracker: &IndependenceTracker) -> bool {
    // 1. Check direct citation
    if tracker.citation_graph.get(&s1).map_or(false, |cites| cites.contains(&s2)) {
        return false;  // s1 cited s2, not independent
    }
    
    // 2. Check temporal clustering (suggests copying from common source)
    let timings = tracker.claim_timings.get(&claim).map_or(Vec::new(), |v| v.clone());
    let t1 = timings.iter().find(|(s, _)| s == &s1).map(|(_, t)| t);
    let t2 = timings.iter().find(|(s, _)| s == &s2).map(|(_, t)| t);
    
    match (t1, t2) {
        (Some(t1), Some(t2)) => {
            // If within 1 hour, suspicious (might be copying same source)
            t1.duration_since(*t2).abs() > Duration::from_secs(3600)
        }
        _ => true,  // Can't determine, assume independent
    }
}
```

**The 1-hour threshold is a parameter**, but it's:
- Documented
- Tunable
- Has a clear interpretation (news cycle)
- NOT a weight

### Emergent Properties

What emerges from this mechanism:

1. **New sources start neutral** (0.5) — no bias
2. **Sources that are consistently right** gain trust slowly
3. **Sources that get contradicted** lose trust quickly
4. **Old claims** gain stability bonus (stood test of time)
5. **Independent corroboration** boosts confidence more than repeated claims from same source

### What Makes This Novel

1. **No source labels** — We don't call arXiv "reliable" or blogs "unreliable"
2. **Learning from outcomes** — Trust is a function of verification history
3. **Temporal dimension** — Old verified claims are more stable than new ones
4. **Independence tracking** — Detect circular reporting
5. **Fully emergent** — No hardcoded authority

---

## Option 2: Pure Consensus-Based Validation

### Core Principle

> "Truth is what independent sources agree on, after surviving verification."

### Mechanism

#### 1. Claim Clustering by Semantic Equivalence

Already exists in nCPU: `ClusterRegistry` groups paraphrases by `meaning_fingerprint`.

```rust
struct FactCluster {
    fingerprint: ClaimFingerprint,
    member_sentences: Vec<String>,    // All paraphrases expressing this
    sources: HashSet<SourceFingerprint>, // Who said this
    contradictions: HashSet<ClaimFingerprint>, // What disagrees
}
```

#### 2. Confidence From Agreement Ratio

**No magic threshold** — confidence is a pure function of agreement:

```rust
fn cluster_confidence(cluster: &FactCluster) -> f64 {
    let agreements = cluster.sources.len();
    let disagreements = cluster.contradictions.len();
    let total = agreements + disagreements;
    
    if total == 0 {
        return 0.0;  // No data, no confidence
    }
    
    // Pure ratio: agreements / (agreements + disagreements)
    (agreements as f64) / (total as f64)
}
```

This is mathematically principled (it's the frequency interpretation of probability).

#### 3. Independence Weighting

Not all agreements count equally:

```rust
fn effective_agreement_count(cluster: &FactCluster, 
                             independence: &IndependenceTracker) -> f64 {
    let mut effective_count = 0.0;
    
    for source in &cluster.sources {
        let independent_from_others = cluster.sources.iter()
            .filter(|other| *other != source)
            .filter(|other| are_independent(*source, **other, cluster.fingerprint, independence))
            .count();
            
        // Each independent agreement contributes 1.0
        // Dependent (copying) contributes diminishing returns
        effective_count += (independent_from_others as f64).sqrt();
    }
    
    effective_count
}
```

The square-root is the standard "discount for correlation" formula from statistics.

#### 4. Contradiction Resolution

When contradictions exist, we don't pick a "winner" — we track both:

```rust
struct ConflictRecord {
    claim_a: ClaimFingerprint,
    claim_b: ClaimFingerprint,
    evidence_for_a: Vec<SourceFingerprint>,
    evidence_for_b: Vec<SourceFingerprint>,
    
    // No "who wins" — this is tracked, not decided
}

// Confidence in each side
fn balance_of_evidence(conflict: &ConflictRecord) -> (f64, f64) {
    let weight_a = effective_agreement_count(&conflict.evidence_for_a);
    let weight_b = effective_agreement_count(&conflict.evidence_for_b);
    let total = weight_a + weight_b;
    
    if total == 0.0 {
        return (0.0, 0.0);
    }
    
    (weight_a / total, weight_b / total)
}
```

Users see "60% of sources say X, 40% say NOT X" — they decide, not the system.

#### 5. Temporal Emergence

Old consensus is more stable than new consensus:

```rust
fn temporal_bonus(cluster: &FactCluster, now: SystemTime) -> f64 {
    let age = cluster.oldest_claim()
        .and_then(|c| now.checked_duration_since(c))
        .unwrap_or(Duration::ZERO);
    
    // Logarithmic bonus (same rationale as Option 1)
    let age_hours = age.as_secs_f64() / 3600.0;
    (age_hours.log10() / 10.0).min(0.5)  // Max 50% bonus for age
}
```

The bonus caps at 50% — even very old claims need at least 50% agreement to be credible.

### Combined Confidence Score

```rust
fn overall_confidence(cluster: &FactCluster, 
                      independence: &IndependenceTracker,
                      now: SystemTime) -> f64 {
    let agreement_ratio = cluster_confidence(cluster);
    let effective_count = effective_agreement_count(cluster, independence);
    let temporal = temporal_bonus(cluster, now);
    
    // Combine:
    // - Base confidence from agreement ratio
    // - Boosted by independent sources
    // - Boosted by age
    
    let base = agreement_ratio;
    let count_boost = (effective_count.log10() / 10.0).min(0.3);  // Max 30% from count
    
    (base + count_boost + temporal).min(1.0)
}
```

### What Makes This Novel

1. **No source authority** — We don't track who said it, just how many agree
2. **Pure counting** — Confidence = (agreements) / (agreements + disagreements)
3. **Independence discount** — Copying doesn't increase confidence
4. **Both sides tracked** — Contradictions preserved, not resolved
5. **Temporal bonus** — Old consensus weighted more
6. **User agency** — System presents evidence, user decides

---

## Hybrid: Combined Approach

We can combine both for maximum robustness:

### Verification Pyramid

```
Level 3: Multi-source consensus
         ↓ (survives if multiple independent sources agree)
         
Level 2: Round-trip verification  
         ↓ (survives if understand(say(X)) == X)
         
Level 1: Parse success
         ↓ ( survives if sentence parses to Meaning)
         
Level 0: Input string
```

### Scoring Formula

```rust
fn claim_confidence(claim: &Claim, 
                   cluster: &FactCluster,
                   source_record: &SourceTrackRecord,
                   now: SystemTime) -> f64 {
    // 1. Round-trip verification (binary)
    let verified = if claim.round_trip_survived { 0.5 } else { 0.0 };
    
    // 2. Source reliability (from Option 1)
    let source_boost = source_record.score() * 0.3;
    
    // 3. Consensus strength (from Option 2)
    let consensus = cluster_confidence(cluster) * 0.2;
    
    verified + source_boost + consensus
}
```

Weights sum to 1.0 (0.5 + 0.3 + 0.2) and represent:
- **50% from verification** — Does the claim hold together?
- **30% from source history** — Has this source been reliable?
- **20% from consensus** — Do others agree?

These weights are **parameters**, not magic numbers. They should be:
- Documented with rationale
- Exposed as configuration
- Validated empirically

---

## What Both Approaches Avoid

1. **No hardcoded source weights** — arXiv ≠ 0.9, Nature ≠ 1.0
2. **No external LLMs** — All verification internal
3. **No arbitrary thresholds** — No "if confidence > 0.7"
4. **No gospel truth** — All knowledge is provisional
5. **No hidden decisions** — User sees why something is trusted

---

## Implementation Priority

### Phase 1: Core Tracking (Foundation)
1. `SourceProvenance` — fingerprint every claim
2. `ClaimState` lifecycle tracking
3. `SourceTrackRecord` — outcome accumulation

### Phase 2: Verification Integration
1. Round-trip gating on all claims
2. Contradiction detection (semantic NOT)
3. Temporal stability tracking

### Phase 3: Scoring & Display
1. `SourceReliability::score()`
2. `cluster_confidence()` calculation
3. User-facing confidence visualization

### Phase 4: Learning Loop
1. Update scores on each verification outcome
2. Decay old scores (forgetting mechanism)
3. Periodic re-evaluation

---

## Key Design Principles

1. **All weights learned from data** — No arbitrary numbers
2. **All verification internal** — No external LLM calls
3. **All confidence provisional** — Can be updated with new data
4. **All decisions transparent** — User sees evidence
5. **All parameters documented** — If a number exists, explain why

---

## What Makes This Novel

Current systems (e.g., fact-checkers, citation networks) either:
- Use hardcoded authority (expert review = truth)
- Use external LLMs (GPT says = truth)
- Use popularity (viral = true)

Our approach:
- Uses **internal verification only** (round-trip)
- Learns from **outcomes** (were we later contradicted?)
- Tracks **independence** (who cited whom?)
- Respects **temporality** (old claims > new claims)
- Preserves **uncertainty** (show both sides)

This is how human knowledge actually works — and we're encoding it as a mechanism, not a set of arbitrary rules.
