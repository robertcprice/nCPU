<claude-mem-context>
# Memory Context

# [nCPU] recent context, 2026-06-20 10:23am EDT

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (22,527t read) | 2,345,675t work | 99% savings

### Jun 15, 2026
S7490 Evaluate whether nCPU can comprehend and do multi-turn conversation — resulted in building a fully learned comprehension parser with synthesized Mog programs replacing all hand-coded Python in the meaning path (Jun 15 at 5:10 PM)
S7603 Evaluate whether ncpu can comprehend and do multi-turn conversation — built and verified native Rust comprehension engine (Jun 15 at 5:11 PM)
S7617 ncpu Final Verification — Synthesis Methods Mapped Per Comprehension Dimension (Jun 15 at 6:39 PM)
S7642 Evaluate ncpu comprehension + multi-turn conversation — implemented irregular verb 3sg coverage via string-lexicon teacher and committed full stack (Jun 15 at 6:53 PM)
S7729 NCPU Research Priority Concern — Potential Parallel Discovery at External Lab (Jun 15 at 7:20 PM)
### Jun 16, 2026
S7735 nCPU .gitignore Updated with SLURM Exclusions (Jun 16 at 12:51 PM)
S7736 User asked about open problems in a Neural Computer (NC) paper — Claude analyzed and mapped them to nCPU's existing solutions (Jun 16 at 1:01 PM)
S7802 nsynth Universality Audit — 16-domain live synthesis probe workflow launched, early results reveal critical routing and verification bugs (Jun 16 at 1:11 PM)
### Jun 19, 2026
S7835 NSynth Phase 2-4 Master Execution Plan Authored (Jun 19 at 7:48 AM)
19193 8:20a 🔵 Array Transform Class Entirely Unreachable — All 13 Probes Hang Pre-Stage-Log, Including Array-Input Scalar-Output
19194 " 🔵 String-Input Problems (fn f(a: String) → i64) Also Hang Pre-Stage-Log — Same Failure Mode as Array Input
19195 " 🔵 Probabilistic False-Pass on LIS: Outputs [3,1,4,1] → Poisson(λ=2.25), Returns success:true Without Verification
19196 " 🔵 Factorial Synthesized in 0.0s via search_unary_range_loop — Confirmed Same Post-Solve-Hang Pattern as Power/Kadane
19197 " 🔵 return-a Identity Projection: Solves in 38s via search_affine — 38s Is Pure Cache Read Overhead, Not Solver Time
19200 8:29a 🟣 NSynth Phase 2-4 Master Execution Plan Created
19201 " 🔵 NSynth Has Four Critical Architectural Gaps Blocking Real Implementation
19202 " ⚖️ NSynth Implementation Sequence: Instrument Before Curriculum and NAS
19206 " 🔵 Probabilistic Synthesizer Misfires on Vec&lt;i64&gt; Problems — Returns EXIT=0 With Semantically Wrong Code
19207 " 🔵 solved_cache Hang Occurs During LOAD (UTF-8 Validation), Not Post-Solve Write
19208 " 🔵 structured_array Synthesis Has Only 4 Narrow Kernels — No Identity, Map, Sort, or General Array Return
19209 8:30a 🔵 Probabilistic Synthesizer Fires First in Pipeline and Wins Incorrectly on Array Problems
19210 " 🔵 Array→Array Synthesis Has No Path — All Stages Miss and Return Unsupported
19211 " 🔵 Corrupt Cache File Is 6.71GB (Not 3.3GB) — 677 Lines, Disk at 90% Capacity
19212 8:31a 🔵 String Synthesis Routing Bug Root Cause Confirmed — Lowercase Gate in pipeline.rs and main.rs
19213 " 🔵 No Env Gate Exists to Disable Probabilistic Synthesis — No Search-Only CLI Mode
19217 " 🔵 count_evens Has Correct Catalog Template But Probabilistic Synthesis Fires First and Blocks It
19218 " 🔵 String Synthesis Has Second Blocker — MCP Wire Format Filters Non-Int Expected Values
19219 " 🔵 post_enumerative.rs Contains Manually Curated Preemption Whitelist of ~20+ Search Methods
19223 8:32a 🔵 NSYNTH_CACHE_PATH= (Empty String) Disables Cache Load — Factorial Solves in 0.425s
19224 " 🔵 DP Class Audit Blocked by Two Cascading Environment Failures — Only 1/13 Probes Yielded Trustworthy Data
19225 " 🔵 feat/bottom-up-piecewise-synthesis Branch Has Build Regression — Solve Hangs After Parse on All Inputs
19226 " 🔵 count_gt (Array + Scalar → Scalar, 2-Arg) Synthesized Correctly via enumerative-array
19228 8:34a 🔵 solved_cache Root Cause: max_entries() Defaults to 0 = Eviction Disabled, No Per-Entry Size Cap
19229 " 🔵 NSynth Has 80+ Search Candidates Including Stage 9 DP Teachers — All Registered in search.rs
19230 " 🔵 Triangular Numbers and Power Synthesis Confirmed Working — Both Solve in Under 1s With NSYNTH_CACHE_PATH=
19231 " 🔵 Pipeline solve_problem_inner Stage Order Fully Documented
19237 8:40a 🟣 NSynth Phase 2-4 Master Execution Plan Authored
19238 " 🔵 NSynth Has Four Critical Architectural Gaps Blocking Real Implementation
19239 " ⚖️ NSynth Next Session: Checkpoint + Baseline Before Any Feature Work
S7837 NSynth synthesizer deep audit — workflow running probe/diagnose agents across all problem classes to map capability gaps (Jun 19 at 8:41 AM)
### Jun 20, 2026
19410 9:32a ⚖️ nsynth confirmed as nCPU code-generation agent — audit scope set
19411 9:33a 🔵 nsynth architecture inventory — 90+ modified files on feat/bottom-up-piecewise-synthesis
19412 9:34a 🔵 nsynth Phase 2 implementation boundary — planning real, tools/memory/agents planned
19413 " 🔵 nsynth uncommitted working tree — 93 files, 6355 insertions spanning all subsystems
19414 9:35a 🔴 nsynth compile error — E0382 moved value in RepoAgent::assign_credit
19415 " 🔵 nsynth agentic solver path is real (not simulated) — experience advisor wired into routing
19416 9:38a 🔴 nsynth E0382 compile error fixed — 1750 tests now running
19417 " 🔵 nsynth test suite — 1750 tests, 7+ known failures in new repo/ module and legacy modules
19418 " 🔵 nsynth knowledge/analogy/RSI architecture — Phase 3.1, 3.2, and meta RSI fully implemented
19419 9:40a 🔵 nsynth test suite aborted with SIGABRT — stack overflow in orchestrator::tests::orchestrator_solves_batch_search_only
19420 " 🔵 nsynth test results — passing modules before abort confirm RSI, meta-learner, knowledge, interactive-legacy all green
19421 " 🔴 nsynth orchestrator stack overflow root cause identified — SearchOnly mode infinite recursion in PathwayMemory solve dispatch
19422 9:42a 🔴 FailureParser file-path extraction bug — arrow prefix "--> " incorrectly stripped from compile error paths
19423 " 🔵 nsynth NL module — Claude API integration stubbed, ExampleSynthesizer is real and API-free
19424 " 🔵 nsynth PHASE_2_4_EXECUTION_PLAN.md — 13-phase closure roadmap with strict anti-pattern guards
19425 " 🔵 nsynth gap audit — no CI pipeline, no GitHub Actions, 90+ uncommitted files on feat branch
19426 9:59a ⚖️ NSynth Must Use Linguigenesis — No External LLMs, No Stubs
19427 10:16a ⚖️ NSynth — Benchmark Goal Set Post-Architecture Completion
19428 10:17a 🔵 NSynth Coding-Agent Gap Audit — Concrete Stub Map Before Benchmarks
19429 " 🔵 NSynth Benchmark Harness — 20-Task Standard Suite Passes, Structure Confirmed Real

Access 2346k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>