//! Evaluation benchmarks for the understanding engine.
//!
//! This module hosts FraCaS-style **three-valued entailment** benchmarks: each
//! case carries one or more PREMISES and a yes/no HYPOTHESIS question, with a
//! gold label in `{Yes, No, Unknown}`. A case is scored by building a fresh
//! [`Mind`](crate::understanding::mind::Mind), reading every premise into its
//! world model, then asking the hypothesis as a yes/no question and bucketing
//! the answer back into the same three-valued space (leading `"Yes"` /
//! leading `"No"` / the open-world `"I don't know."`).
//!
//! The **soundness bar is zero wrong**: the open-world engine may answer
//! `Unknown` where a human would say `Yes`/`No` (under-determination is
//! permitted), but it must NEVER assert `Yes` for a `No`-gold case or vice
//! versa. A wrong answer is a soundness violation, not merely a missed point;
//! [`entailment::BenchReport::sound`] gates on `wrong == 0` while
//! [`entailment::BenchReport::accuracy`] reports the (separate) coverage.

pub mod entailment;
