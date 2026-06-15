# Stage 6: Module Composition — Design Index

**Goal**: Synthesize programs by composing verified library functions (filter, map, reduce, scan) instead of writing monolithic function bodies.

**Status**: Design phase — ready for implementation

---

## Three Core Documents

### 1. [STAGE_6_COMPOSITION.md](STAGE_6_COMPOSITION.md)
**The Vision & Strategy**

- High-level goal: break monolithic synthesis into function pipelines
- Why it matters: reusability, reduced synthesis time, clearer intent
- Architecture overview: library templates, metadata extension, solver integration
- 4 implementation phases with clear milestones
- Challenge analysis with solution options
- 8-12 benchmark specifications
- Integration checklist
- 2.5-week estimated timeline

**Read this first to understand the "why" and "what".**

---

### 2. [STAGE_6_ROADMAP.md](STAGE_6_ROADMAP.md)
**The Implementation Plan**

- Sprint structure: 2-3 sprints, 1-2.5 weeks
- Detailed day-by-day breakdown with deliverables
- File-by-file modification checklist
- Milestone tracking (6 major milestones)
- Risk mitigation table
- Success criteria (hard and soft)
- Post-implementation tasks

**Use this to guide the actual coding work.**

---

### 3. [STAGE_6_TECHNICAL_SPEC.md](STAGE_6_TECHNICAL_SPEC.md)
**The Implementation Details**

- Data structures: `CompositionPattern`, `CompositionTemplate`, `PatternMatch`
- Pattern detection algorithm with pseudocode for each pattern (filter, map, reduce, scan, sort)
- Inline codegen functions for each pattern
- Multi-function call codegen
- Verification strategy
- Teacher integration into solver pipeline
- Composition library template specifications
- Configuration & tuning parameters
- Complete walk-through example

**Reference this while coding to implement each component.**

---

## Quick Start

### For Understanding
1. Read **STAGE_6_COMPOSITION.md** sections 1-3 (vision, architecture, phases)
2. Skim **STAGE_6_ROADMAP.md** section 1 (sprints overview)
3. Reference **STAGE_6_TECHNICAL_SPEC.md** when coding

### For Implementation
1. Follow **STAGE_6_ROADMAP.md** sprint-by-sprint
2. Use **STAGE_6_TECHNICAL_SPEC.md** as code reference
3. Cross-check with **STAGE_6_COMPOSITION.md** integration checklist

### For Testing
1. **STAGE_6_ROADMAP.md** section "Detailed Work Breakdown" → test targets
2. **STAGE_6_TECHNICAL_SPEC.md** section 7.2 → template verification
3. **STAGE_6_COMPOSITION.md** section 5 → benchmark specs

---

## Key Concepts at a Glance

### Composition Patterns

| Pattern | Input | Output | Example |
|---------|-------|--------|---------|
| **Filter** | Array → smaller array | Remove elements | `[1,-2,3] → [1,3]` (x > 0) |
| **Map** | Array → array (same size) | Transform each element | `[1,2,3] → [2,4,6]` (×2) |
| **Reduce** | Array → scalar | Fold to single value | `[1,2,3] → 6` (sum) |
| **Scan** | Array → array (cumulative) | Cumulative fold | `[1,2,3] → [1,3,6]` (cumsum) |
| **Sort** | Array → sorted array | Reorder | `[3,1,2] → [1,2,3]` |

### Implementation Stages

1. **Phase 1 (inline)**: Generate code with inlined patterns, no function calls
   - 8 benchmarks, ~50 hours, 1 week
   
2. **Phase 2 (calls)**: Generate multi-function code with actual function calls
   - 4+ benchmarks, ~26 hours, 1.5 days
   
3. **Phase 3 (advanced)**: Nested compositions, learned biases, type checking
   - Optional, ~37 hours, 1 week
   
4. **Phase 4 (polish)**: Paper, final regression, optimization
   - ~20 hours, included in phase 2/3

---

## Deliverables by Document

### STAGE_6_COMPOSITION.md Delivers
✓ Architectural vision  
✓ Problem motivation & scope  
✓ Challenge analysis (3 major challenges with 3 solution options each)  
✓ Benchmark specifications (8 Phase 1 + 4 Phase 2 + 4 Phase 3)  
✓ Integration checklist  
✓ Open questions  
✓ References & related work  

### STAGE_6_ROADMAP.md Delivers
✓ Sprint structure (3 sprints)  
✓ Day-by-day deliverables  
✓ File-by-file modification checklist  
✓ 6 milestone gates  
✓ Risk mitigation strategy  
✓ Success criteria (hard & soft)  
✓ Timeline visualization  
✓ Post-implementation roadmap  

### STAGE_6_TECHNICAL_SPEC.md Delivers
✓ Complete data structures (Rust code)  
✓ Pattern detection pseudocode (all 6 patterns)  
✓ Inline codegen for each pattern  
✓ Multi-function call codegen  
✓ Verification strategy  
✓ Library template examples  
✓ Integration points in solver  
✓ Error handling & graceful fallback  
✓ Walk-through example (filter_sum_positive)  

---

## How These Docs Relate

```
STAGE_6_COMPOSITION.md
  ├─ Vision & scope
  ├─ Architecture overview
  ├─ Phase breakdown
  └─ Integration checklist
      └ Points to → STAGE_6_ROADMAP.md
            ├─ Sprint structure
            ├─ Day-by-day tasks
            └─ File checklist
                └ Points to → STAGE_6_TECHNICAL_SPEC.md
                      ├─ Data structures
                      ├─ Algorithm pseudocode
                      └─ Code examples
```

**Read top-to-bottom for understanding; read sideways for cross-references.**

---

## Integration with Main ARCHITECTURE.md

Once implemented, add to `docs/ARCHITECTURE.md` under "Solver teachers":

```markdown
- [STAGE_6_COMPOSITION.md](STAGE_6_COMPOSITION.md) — Function composition
  synthesis using verified library templates (filter, map, reduce, scan, sort).
  Detects composition patterns from input/output examples, emits inline or
  multi-function code. Replaces monolithic synthesis with reusable pipelines.
  Teacher: `search_composition` (between enumerative and gradient).
```

---

## Estimated Impact

### Coverage
- **Current**: 105/105 existing benchmarks
- **Post-Stage 6**: 105/105 (regression-free) + 12-16 new composition benchmarks

### Performance
- Phase 1 (inline): ~2-3x faster than monolithic for composition problems
- Phase 2 (calls): ~1.5-2x faster with function overhead
- Non-composition problems: no change (neutral)

### Code Quality
- Reduced synthesis time: composition solves in <2s vs. 10-30s for gradient
- Improved readability: composed code shows intent (filter → sum)
- Better reusability: library functions used across many problems

---

## Next Steps

1. **Review**: Read all three documents, open questions/feedback
2. **Start Implementation**: Follow STAGE_6_ROADMAP.md Day 1-2 (metadata + library)
3. **Iterate**: Sprint 1 → Sprint 2 → Sprint 3 (optional)
4. **Validate**: Run regression suite (105 + 12 benchmarks)
5. **Publish**: Add paper section, update architecture docs

---

## Document Status

| Document | Status | Pages | Lines |
|----------|--------|-------|-------|
| STAGE_6_COMPOSITION.md | Complete | 12 | 600+ |
| STAGE_6_ROADMAP.md | Complete | 15 | 700+ |
| STAGE_6_TECHNICAL_SPEC.md | Complete | 16 | 800+ |
| STAGE_6_INDEX.md | This file | 2 | 200+ |
| **Total** | **Ready** | **45** | **2300+** |

---

## Questions?

See "Open Questions" in STAGE_6_COMPOSITION.md section 7:
1. Should composition be enabled by default?
2. Should library functions be synthesized or hand-coded?
3. How to handle edge cases in composed calls?
4. Can composition help with template slowdown?

---

**Date**: June 15, 2026  
**Author**: Bobby Price  
**Version**: 1.0 (Design Complete, Ready for Implementation)
