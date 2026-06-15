<claude-mem-context>
# Memory Context

# [nCPU] recent context, 2026-06-14 8:28pm EDT

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (22,130t read) | 686,500t work | 97% savings

### Jun 11, 2026
S6385 Evaluate nCPU repo — current state, what's needed to finish, publish, and extract SaaS/commercial value (Jun 11 at 9:48 AM)
S6427 nCPU session: compile NPCoT WASM runtime, integrate into ncpu-site demo page, fix production build, commit both repos (Jun 11 at 9:50 AM)
S6443 Evaluate nCPU/nsynth repo, determine current state, identify what's needed to finish, publish, and extract SaaS/commercial value (Jun 11 at 10:48 AM)
S6445 nsynth repo evaluation — architecture tiers, extension roadmap, and SaaS/publication value extraction (Jun 11 at 11:23 AM)
S7104 nCPU/nsynth + LinguaGenesis integration — can nsynth's code-reasoning engine learn English rules from linguigenesis's rule-generated examples? (Jun 11 at 11:27 AM)
### Jun 13, 2026
S7117 LINGUAGENESIS_BRIDGE.md Written — Complete Integration Reference Doc (Jun 13 at 8:35 PM)
17360 8:38p 🔵 nsynth search_text_families.rs — Only 6 String Teachers, All String→i64, 124 Lines Total
17361 " 🔵 LinguaGenesis morphology.py — Complete Pluralization and Conjugation Rules Mapped
17362 8:39p 🔵 nsynth Search Teacher Infrastructure Fully Mapped — Pattern for Adding Morphology Teacher is Clear
17369 8:41p ⚖️ nsynth+LinguaGenesis Integration — 3-Task Spike Plan Committed and Source Locations Pinned
17370 " 🟣 nsynth search_suffix_class Teacher Implemented and Registered — First Morphology-Capable Synthesis Rule
17377 8:42p 🟣 nsynth search_suffix_class Verified Live — Rediscovers English Sibilant Rule from Examples in 0.0s
17379 8:43p 🟣 LinguaGenesis→nsynth Full Cross-Project Loop Completed — Exact Sibilant Rule Synthesized
17383 8:45p 🟣 Regression Test search_suffix_class_learns_sibilant_plural_rule Added to tests.rs
17384 8:46p 🟣 Regression Test Passes — search_suffix_class_learns_sibilant_plural_rule Green
17394 8:47p ✅ LINGUAGENESIS_BRIDGE.md Written — Complete Integration Reference Doc
S7122 linguagenesis_bridge.py Rewritten — Curriculum-Only Oracle, Two Task Modes (Jun 13 at 8:47 PM)
17398 8:53p ⚖️ LinguaGenesis nsynth Integration — Curriculum-Driven Teaching Directive
17402 8:56p 🔵 LinguaGenesis v2 Curriculum — Full Module Map and Generator/Validator API
17403 8:57p 🔵 LinguaGenesis Stage3 Generator — Live Output Confirms Rule-Tagged Paired Examples
17404 " 🔵 nsynth — 8 Pre-Existing Test Failures in Orchestrator and Interactive Solver
17405 9:02p 🟣 linguagenesis_bridge.py Rewritten — Curriculum-Only Oracle, Two Task Modes
17406 " 🔵 nsynth Solves verb_3sg_es But Overgeneralizes to ends_with("h") — Verb Lexicon Lacks Hard Negatives
S7123 nsynth + LinguaGenesis integration — ensure language learning follows the actual curriculum, not arbitrary rules invented in the bridge (Jun 13 at 9:02 PM)
S7153 LinguaGenesis × nsynth fusion — can nsynth's code-reasoning engine learn English grammar through rules coded into the curriculum? (Jun 13 at 9:02 PM)
17407 9:03p 🔵 LinguaGenesis Has word_tokenizer.py and morpheme_tokenizer.py — Directly Relevant to Sentence-Level Task
17408 " 🔵 LinguaGenesis MorphemeTokenizer — Splits Inflected Forms into Stem + Suffix Tokens Using Curriculum Lexicon
17419 9:11p 🟣 nsynth Now Recovers Full ch/sh/ss/x/z/o Rule — Bridge Expanded to Include Noun Lexicon and -o Verbs
17420 " 🔵 MorphemeTokenizer Produces Suffix Tokens on Stage-3 Sentences — But Only Discriminates 3sg, Not Past/Gerund
17421 " 🟣 nsynth — code_array_member_class_search() Codegen Added to search_codegen.rs
17423 9:12p 🟣 nsynth search_array_member_class Teacher Wired into SEARCH_CANDIDATES Pipeline
17424 " 🟣 nsynth Array-Member-Class Teacher Pipeline Complete — Clean Release Build
17429 " 🟣 linguagenesis_bridge.py task_sentence_3sg Rewritten — Raw Strings Replaced with MorphemeTokenizer ID Arrays
17430 9:13p 🔵 nsynth Panics on sentence_3sg Array Input — universal_array.rs:547 Index Out of Bounds
17433 " 🔵 universal_array.rs Panic Root Cause — Sigmoid Early-Exit Fails When Array Length = MAX_ARR (16)
17434 9:14p 🔵 universal_array.rs Panic Root Cause Refined — arr_len Exceeds MAX_ARR When Input Array >16 Tokens
17435 " 🔵 universal_array Panic Root Cause Final — native_array.rs extract_arr_examples Sets arr_len Without MIN(MAX_ARR) Clamp
17441 9:19p ⚖️ LinguaGenesis/nsynth — Curriculum-Grounded Language Learning Directive
17460 9:22p ⚖️ oNeura + LinguaGenesis Architectural Fusion — Rule-Based Language Learning via nsynth
17461 " 🔵 nsynth Solver Architecture — Symbolic Search Preempts Gradient Distillation for Exact Solutions
17462 " 🔵 nsynth Preemption Whitelist — Conditional Entry for search_unary_range_loop
17472 9:27p 🟣 nsynth — search_suffix_class + search_array_member_class Added as Gradient-Preempting Solvers
17482 9:31p 🟣 LinguaGenesis Bridge — task_sentence_3sg Narrowed to Sibilant Slice + task_sentence_full Added as Frontier
17483 " 🟣 nsynth + LinguaGenesis Fusion — First Successful Sentence-Level Grammaticality Solve
17490 9:36p 🟣 nsynth — Unit Test Added for search_array_member_class on Morpheme Token Membership
17495 9:39p 🟣 nsynth — New morph_transduce.rs Module: Generative String-to-String Morphology Synthesis
17496 " 🟣 nsynth — morph_transduce Wired Into Crate + execute_str_function Added to Runtime
17497 " 🟣 nsynth CLI — --problem-json Now Routes String-Output Problems to Generative Morphology Path
17498 9:40p 🟣 nsynth CLI — try_morph_transduction() Implemented: Auto-Routes String-Output Problems to Generative Morphology
17500 9:44p 🔵 nsynth Interactive + Orchestrator Test Suites — All 36 Tests Failing
17501 9:45p 🔵 nsynth egdc Directory — No Python Files, mog_gradient_bridge.py Missing
17506 9:48p 🔵 nsynth Release Build Succeeds Despite 36 Runtime Test Failures
17507 9:49p 🔵 morph_transduce Sibilant Plural Test Panics — Suffix Append Partially Correct
17508 " 🔵 linguagenesis_bridge.py TypeError — sum() Called on String Examples for String-Return Tasks
17509 " 🟣 nsynth CLI Pluralization — morph_transduce_suffix_append Synthesizes English Plural Rules End-to-End
17517 9:50p 🔴 morph_transduce.rs — Default-Append Selection Refactored to Try-All + Verify Loop
17518 " 🔴 linguagenesis_bridge.py — TypeError on sum() Fixed for String-Return Tasks
17519 " 🟣 LinguaGenesis + nsynth End-to-End Pluralization Pipeline — Curriculum-Driven, Fully Automated
17520 9:51p 🟣 LinguaGenesis/nsynth Fusion — All 3 Solver Paths + 4 Regression Tests Green
S7154 LinguaGenesis × nsynth fusion — nsynth learns English grammar rules from the LinguaGenesis curriculum as verified executable Mog programs (Jun 13 at 9:53 PM)
**Investigated**: Examined nsynth solver architecture, LinguaGenesis curriculum modules (REGULAR_VERBS, _correct_3sg_form, morpheme_tokenizer, pluralize), and the bridge script. Confirmed 36 pre-existing interactive/orchestrator failures all trace to missing egdc/mog_gradient_bridge.py. Found the curriculum lexicon was missing all -z verbs (buzz/fizz/whizz) despite its own comment promising them. Confirmed release build clean; confirmed morph_transduce panic at line 233 was caused by frequency-based default-append selection failing on sibilant-heavy training sets.

**Learned**: - nsynth can learn LinguaGenesis curriculum rules across three modes: classify (string→0/1 via search_suffix_class), sentence-level (token-id array→0/1 via search_array_member_class), generative (string→string via morph_transduce).
    - LinguaGenesis curriculum REGULAR_VERBS was missing all -z verbs — added to both repos.
    - morph_transduce default-append selection by frequency alone picks wrong "elsewhere" class; fix: try each candidate default, keep first that verifies on holdouts.
    - linguagenesis_bridge.py sum() crashed on string-return tasks — fixed with isinstance type guard.
    - Full sentence grammaticality requires AND-of-features over token stream; single token-id membership is insufficient; nsynth correctly declines rather than overfits.
    - Morpheme tokenizer makes suffix tokens explicit in token-id arrays, enabling sentence-level 3sg task to become tractable.

**Completed**: 1. morph_transduce.rs refactored — build_branches() + verify_transduction() helpers extracted; try-all-defaults loop replaces frequency-pick; panic at line 233 fixed; 2/2 tests pass.
    2. linguagenesis_bridge.py fixed — type-guards sum() call for string-return tasks.
    3. All three pipeline tasks verified end-to-end: verb_3sg_es→search_suffix_class, sentence_3sg→search_array_member_class True, pluralize_gen→morph_transduce_suffix_append True (87 train, 30 holdout).
    4. 4 new regression tests green; 0 new failures.
    5. Session memory written to ~/.claude/projects/-Users-bobbyprice-projects-nCPU/memory/linguagenesis_nsynth_bridge.md.
    6. MEMORY.md updated with LinguaGenesis × nsynth bridge entry.
    7. All changes in working tree — not yet committed.

**Next Steps**: Session ended at a decision point: commit now (separate commits per repo) or continue iterating. Likely next work: sentence-level conjunctions (AND-of-features over token stream for full Stage-3 grammaticality) and/or stem-changing transduction (city→cities needs substring ops, currently reported unsupported).


Access 687k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>