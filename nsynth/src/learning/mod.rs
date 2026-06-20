//! Learning module for experience tracking and continuous improvement.
//!
//! This module provides capabilities for:
//! - Recording synthesis attempts and outcomes
//! - Extracting lessons from successful solves
//! - Querying historical data for similar problems
//! - Tracking effectiveness over time

pub mod experience;

pub use experience::{
    EffectivenessStats, Experience, ExperienceDB, ExperienceDay, Lesson, LessonAction,
    LessonPattern, ProblemComplexity, ProblemSnapshot, SolutionSnapshot, SolveMetadata,
    SolveOutcome,
};
