//! Experience tracking for continuous learning from synthesis attempts.
//!
//! This module implements a comprehensive experience database that:
//! - Records every solve attempt with problem context, solution, and outcome
//! - Extracts lessons from successful solves (patterns, actions, effectiveness)
//! - Provides queries to find similar problems and effective actions
//! - Tracks effectiveness over time to enable meta-learning
//!
//! # Architecture
//!
//! - **Experience**: Complete record of a solve attempt
//! - **Lesson**: Extracted knowledge pattern with effectiveness tracking
//! - **ExperienceDB**: Persistent storage and querying

use crate::benchmark::{Problem, Value};
use crate::solver::SolveResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// A complete experience record from a single solve attempt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Experience {
    /// Unique identifier for this experience
    pub id: u64,
    /// Timestamp when this experience was recorded
    pub timestamp: u64,

    /// Problem that was attempted
    pub problem: ProblemSnapshot,
    /// Solution that was found (or attempted)
    pub solution: SolutionSnapshot,
    /// Outcome of the solve attempt
    pub outcome: SolveOutcome,

    /// Lessons extracted from this experience
    pub lessons: Vec<Lesson>,

    /// Metadata about the solving process
    pub metadata: SolveMetadata,
}

/// Snapshot of a problem at the time of solving.
/// Stores the essential features for similarity matching.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProblemSnapshot {
    /// Function signature (e.g., "fn fn_name(i64, i64) -> i64")
    pub signature: String,
    /// Problem category/domain
    pub category: String,
    /// Number of input arguments
    pub arity: usize,
    /// Input type pattern (e.g., "II" for two ints, "AI" for array+int)
    pub input_pattern: String,
    /// Output type (Int, Float, Str, Bool, Array, etc.)
    pub output_type: String,
    /// Whether recursion is allowed/required
    pub recursive_allowed: bool,
    /// Whether tree input is involved
    pub tree_input: bool,
    /// Number of examples provided
    pub num_examples: usize,
    /// Complexity estimate based on example size
    pub complexity: ProblemComplexity,
}

/// Estimated complexity of a problem based on features.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProblemComplexity {
    Trivial,
    Simple,
    Medium,
    Complex,
    VeryComplex,
}

/// Snapshot of the solution that was produced.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolutionSnapshot {
    /// Method/route used to solve (e.g., "search", "gradient", "template")
    pub method: String,
    /// Generated code (may be partial/failed)
    pub code: String,
    /// Code size in characters
    pub code_size: usize,
    /// Key constructs used in the solution
    pub constructs: Vec<String>,
}

/// Outcome of a solve attempt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolveOutcome {
    Success,
    PartialSuccess,
    Failure(String),
    Timeout,
}

/// A lesson learned from experience.
/// Encapsulates a pattern-action pair with effectiveness tracking.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Lesson {
    /// Unique identifier for this lesson
    pub id: u64,
    /// When this lesson was first learned
    pub created_at: u64,
    /// When this lesson was last updated
    pub updated_at: u64,

    /// Pattern that triggers this lesson
    pub pattern: LessonPattern,
    /// Action to take when pattern matches
    pub action: LessonAction,

    /// Effectiveness score (0.0 to 1.0)
    pub effectiveness: f64,
    /// Number of times this lesson was applied
    pub applications: usize,
    /// Number of successful applications
    pub successes: usize,

    /// Confidence in this lesson (increases with more evidence)
    pub confidence: f64,
}

/// Pattern that identifies when a lesson is relevant.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LessonPattern {
    /// Pattern based on input/output types
    TypePattern {
        input_pattern: String,
        output_type: String,
    },
    /// Pattern based on problem category
    CategoryPattern { category: String },
    /// Pattern based on required constructs
    ConstructPattern { required_constructs: Vec<String> },
    /// Pattern based on complexity
    ComplexityPattern {
        min_complexity: ProblemComplexity,
        max_complexity: ProblemComplexity,
    },
    /// Composite pattern (all must match)
    CompositePattern { patterns: Vec<LessonPattern> },
}

/// Action to take when a lesson's pattern matches.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LessonAction {
    /// Try a specific solving method first
    PreferMethod { method: String },
    /// Apply a specific code construct
    UseConstruct { construct: String, template: String },
    /// Use a specific strategy
    UseStrategy { strategy: String },
    /// Configure solver parameters
    ConfigureSolver { params: HashMap<String, String> },
}

/// Metadata about the solving process.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolveMetadata {
    /// Time taken to solve (milliseconds)
    pub solve_time_ms: u64,
    /// Number of attempts before success
    pub attempts: usize,
    /// Solver version/configuration
    pub solver_version: String,
    /// Which route was taken through the solver
    pub route: Vec<String>,
}

/// Persistent experience database with querying capabilities.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExperienceDB {
    /// All experiences recorded
    experiences: Vec<Experience>,
    /// All lessons learned
    lessons: Vec<Lesson>,
    /// Path to the persisted database file
    db_path: PathBuf,
}

impl ExperienceDB {
    /// Create or load an experience database.
    pub fn new(db_path: PathBuf) -> Result<Self, String> {
        if db_path.exists() {
            let content =
                fs::read_to_string(&db_path).map_err(|e| format!("Failed to read DB: {}", e))?;
            let db: ExperienceDB =
                serde_json::from_str(&content).map_err(|e| format!("Failed to parse DB: {}", e))?;
            Ok(db)
        } else {
            // Ensure parent directory exists
            if let Some(parent) = db_path.parent() {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("Failed to create DB directory: {}", e))?;
            }
            Ok(Self {
                experiences: Vec::new(),
                lessons: Vec::new(),
                db_path,
            })
        }
    }

    /// Save the database to disk.
    pub fn save(&self) -> Result<(), String> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize DB: {}", e))?;
        fs::write(&self.db_path, content).map_err(|e| format!("Failed to write DB: {}", e))?;
        Ok(())
    }

    /// Record a new experience from a solve attempt.
    pub fn record_experience(
        &mut self,
        problem: &Problem,
        result: &SolveResult,
        solve_time_ms: u64,
    ) -> Result<(), String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| format!("Failed to get time: {}", e))?
            .as_secs();

        let id = self.experiences.len() as u64;

        let problem_snapshot = ProblemSnapshot::from_problem(problem);
        let solution_snapshot = SolutionSnapshot::from_result(result);
        let outcome = SolveOutcome::from_result(result);

        let metadata = SolveMetadata {
            solve_time_ms,
            attempts: 1, // TODO: Track actual attempts
            solver_version: env!("CARGO_PKG_VERSION").to_string(),
            route: vec![result.method.clone()],
        };

        // Extract lessons from this experience
        let lessons = self.extract_lessons(&problem_snapshot, &solution_snapshot, &outcome, now);

        let experience = Experience {
            id,
            timestamp: now,
            problem: problem_snapshot,
            solution: solution_snapshot,
            outcome,
            lessons: lessons.clone(),
            metadata,
        };

        self.experiences.push(experience);

        // Merge lessons into the main lesson database
        for lesson in lessons {
            self.merge_lesson(lesson);
        }

        self.save()?;
        Ok(())
    }

    /// Extract lessons from a solve experience.
    fn extract_lessons(
        &self,
        problem: &ProblemSnapshot,
        solution: &SolutionSnapshot,
        outcome: &SolveOutcome,
        timestamp: u64,
    ) -> Vec<Lesson> {
        let mut lessons = Vec::new();

        // Only learn from successes and partial successes
        if !matches!(
            outcome,
            SolveOutcome::Success | SolveOutcome::PartialSuccess
        ) {
            return lessons;
        }

        let id = self.lessons.len() as u64 + self.experiences.len() as u64;

        // Learn type-based lessons
        lessons.push(Lesson {
            id: id,
            created_at: timestamp,
            updated_at: timestamp,
            pattern: LessonPattern::TypePattern {
                input_pattern: problem.input_pattern.clone(),
                output_type: problem.output_type.clone(),
            },
            action: LessonAction::PreferMethod {
                method: solution.method.clone(),
            },
            effectiveness: if matches!(outcome, SolveOutcome::Success) {
                1.0
            } else {
                0.5
            },
            applications: 1,
            successes: if matches!(outcome, SolveOutcome::Success) {
                1
            } else {
                0
            },
            confidence: 0.5, // Starts low, increases with evidence
        });

        // Learn construct-based lessons
        for construct in &solution.constructs {
            lessons.push(Lesson {
                id: id + lessons.len() as u64,
                created_at: timestamp,
                updated_at: timestamp,
                pattern: LessonPattern::ConstructPattern {
                    required_constructs: vec![construct.clone()],
                },
                action: LessonAction::UseConstruct {
                    construct: construct.clone(),
                    template: format!("// Template for {}", construct),
                },
                effectiveness: if matches!(outcome, SolveOutcome::Success) {
                    1.0
                } else {
                    0.5
                },
                applications: 1,
                successes: if matches!(outcome, SolveOutcome::Success) {
                    1
                } else {
                    0
                },
                confidence: 0.5,
            });
        }

        lessons
    }

    /// Merge a lesson into the database, updating if similar exists.
    fn merge_lesson(&mut self, lesson: Lesson) {
        // Look for similar existing lessons
        for existing in &mut self.lessons {
            if existing.pattern == lesson.pattern && existing.action == lesson.action {
                // Update existing lesson with Bayesian learning
                existing.applications += lesson.applications;
                existing.successes += lesson.successes;
                existing.effectiveness = existing.successes as f64 / existing.applications as f64;
                existing.updated_at = lesson.updated_at;

                // Increase confidence with more data
                existing.confidence = (existing.confidence * 0.9)
                    + (0.1 * (1.0 - 1.0 / (existing.applications + 1) as f64));
                return;
            }
        }

        // No similar lesson found, add as new
        self.lessons.push(lesson);
    }

    /// Find experiences with similar problems.
    pub fn find_similar_problems(&self, problem: &Problem, limit: usize) -> Vec<&Experience> {
        let snapshot = ProblemSnapshot::from_problem(problem);

        let mut scored: Vec<_> = self
            .experiences
            .iter()
            .map(|exp| {
                let score = similarity_score(&snapshot, &exp.problem);
                (exp, score)
            })
            .filter(|(_, score)| *score > 0.0)
            .collect();

        // Sort by similarity score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored.into_iter().take(limit).map(|(exp, _)| exp).collect()
    }

    /// Get effective actions for a given problem.
    pub fn get_effective_actions(&self, problem: &Problem) -> Vec<&Lesson> {
        let snapshot = ProblemSnapshot::from_problem(problem);

        let mut relevant: Vec<_> = self
            .lessons
            .iter()
            .filter(|lesson| lesson.pattern.matches(&snapshot) && lesson.effectiveness > 0.5)
            .collect();

        // Sort by effectiveness and confidence
        relevant.sort_by(|a, b| {
            let score_a = a.effectiveness * a.confidence;
            let score_b = b.effectiveness * b.confidence;
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        relevant
    }

    /// Get effectiveness statistics over time.
    pub fn effectiveness_over_time(&self) -> EffectivenessStats {
        let mut stats = EffectivenessStats::default();

        for exp in &self.experiences {
            let day = exp.timestamp / 86400; // Days since epoch
            stats
                .daily_counts
                .entry(day)
                .or_insert_with(ExperienceDay::default);

            let day_stats = stats.daily_counts.get_mut(&day).unwrap();
            day_stats.total += 1;

            match &exp.outcome {
                SolveOutcome::Success => day_stats.successes += 1,
                SolveOutcome::PartialSuccess => day_stats.partials += 1,
                SolveOutcome::Failure(_) => day_stats.failures += 1,
                SolveOutcome::Timeout => day_stats.timeouts += 1,
            }
        }

        stats
    }

    /// Get the total number of experiences.
    pub fn len(&self) -> usize {
        self.experiences.len()
    }

    /// Get the total number of lessons.
    pub fn num_lessons(&self) -> usize {
        self.lessons.len()
    }

    /// Get all experiences.
    pub fn experiences(&self) -> &[Experience] {
        &self.experiences
    }

    /// Get all lessons.
    pub fn lessons(&self) -> &[Lesson] {
        &self.lessons
    }
}

/// Calculate similarity score between two problem snapshots.
fn similarity_score(a: &ProblemSnapshot, b: &ProblemSnapshot) -> f64 {
    let mut score = 0.0;

    // Exact signature match
    if a.signature == b.signature {
        score += 0.4;
    }

    // Input pattern match
    if a.input_pattern == b.input_pattern {
        score += 0.2;
    }

    // Output type match
    if a.output_type == b.output_type {
        score += 0.15;
    }

    // Category match
    if a.category == b.category {
        score += 0.1;
    }

    // Similar arity
    if a.arity == b.arity {
        score += 0.05;
    }

    // Complexity similarity
    let complexity_diff =
        complexity_score(&a.complexity) as i64 - complexity_score(&b.complexity) as i64;
    score += 0.1 - (complexity_diff.abs() as f64 * 0.02).max(0.0);

    score.max(0.0)
}

/// Get numeric score for complexity level.
fn complexity_score(complexity: &ProblemComplexity) -> u32 {
    match complexity {
        ProblemComplexity::Trivial => 0,
        ProblemComplexity::Simple => 1,
        ProblemComplexity::Medium => 2,
        ProblemComplexity::Complex => 3,
        ProblemComplexity::VeryComplex => 4,
    }
}

/// Effectiveness statistics over time.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EffectivenessStats {
    pub daily_counts: HashMap<u64, ExperienceDay>,
}

/// Experience counts for a single day.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ExperienceDay {
    pub total: usize,
    pub successes: usize,
    pub partials: usize,
    pub failures: usize,
    pub timeouts: usize,
}

impl ExperienceDay {
    /// Success rate (including partials as half success).
    pub fn success_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.successes + self.partials / 2) as f64 / self.total as f64
        }
    }
}

// Implementations for converting from/to existing types

impl ProblemSnapshot {
    pub fn from_problem(problem: &Problem) -> Self {
        use ProblemComplexity::*;

        let arity = problem
            .examples
            .first()
            .map(|e| e.inputs.len())
            .unwrap_or(0);

        let input_pattern = problem
            .examples
            .first()
            .map(|e| {
                e.inputs
                    .iter()
                    .map(|v| match v {
                        Value::Int(_) => 'I',
                        Value::Float(_) => 'F',
                        Value::Str(_) => 'S',
                        Value::Bool(_) => 'B',
                        Value::Array(_) => 'A',
                        Value::Pair(_, _) => 'P',
                        Value::Quad(_, _, _, _) => 'Q',
                        Value::Tree(_) => 'T',
                    })
                    .collect()
            })
            .unwrap_or_default();

        let output_type = match problem.examples.first().map(|e| &e.expected) {
            Some(Value::Int(_)) => "Int",
            Some(Value::Float(_)) => "Float",
            Some(Value::Str(_)) => "Str",
            Some(Value::Bool(_)) => "Bool",
            Some(Value::Array(_)) => "Array",
            Some(Value::Pair(_, _)) => "Pair",
            Some(Value::Quad(_, _, _, _)) => "Quad",
            Some(Value::Tree(_)) => "Tree",
            None => "Unknown",
        }
        .to_string();

        let complexity = if problem.examples.len() > 8 {
            VeryComplex
        } else if problem.examples.len() > 4 {
            Complex
        } else if problem.examples.len() > 2 {
            Medium
        } else if problem.examples.len() > 1 {
            Simple
        } else {
            Trivial
        };

        Self {
            signature: problem.signature.to_string(),
            category: problem.category.to_string(),
            arity,
            input_pattern,
            output_type,
            recursive_allowed: problem.recursive_allowed,
            tree_input: problem.tree_input,
            num_examples: problem.examples.len(),
            complexity,
        }
    }
}

impl SolutionSnapshot {
    pub fn from_result(result: &SolveResult) -> Self {
        let constructs = extract_constructs(&result.code);

        Self {
            method: result.method.clone(),
            code: result.code.clone(),
            code_size: result.code.len(),
            constructs,
        }
    }
}

impl SolveOutcome {
    pub fn from_result(result: &SolveResult) -> Self {
        if result.success {
            SolveOutcome::Success
        } else {
            match &result.error {
                Some(e) if e.contains("timeout") => SolveOutcome::Timeout,
                Some(e) => SolveOutcome::Failure(e.clone()),
                None => SolveOutcome::Failure("Unknown error".to_string()),
            }
        }
    }
}

impl LessonPattern {
    /// Check if this pattern matches a problem snapshot.
    pub fn matches(&self, problem: &ProblemSnapshot) -> bool {
        match self {
            Self::TypePattern {
                input_pattern,
                output_type,
            } => problem.input_pattern == *input_pattern && problem.output_type == *output_type,
            Self::CategoryPattern { category } => problem.category == *category,
            Self::ConstructPattern {
                required_constructs: _,
            } => {
                // TODO: Implement construct matching
                true
            }
            Self::ComplexityPattern {
                min_complexity,
                max_complexity,
            } => {
                let score = complexity_score(&problem.complexity);
                let min = complexity_score(min_complexity);
                let max = complexity_score(max_complexity);
                score >= min && score <= max
            }
            Self::CompositePattern { patterns } => patterns.iter().all(|p| p.matches(problem)),
        }
    }
}

/// Extract code constructs from source code.
fn extract_constructs(code: &str) -> Vec<String> {
    let mut constructs = Vec::new();

    // Common constructs to detect
    let patterns = [
        ("for", "loop"),
        ("while", "loop"),
        ("if", "conditional"),
        ("match", "pattern_matching"),
        (".map(", "map"),
        (".filter(", "filter"),
        (".fold(", "fold"),
        (".reduce(", "reduce"),
        ("recursion", "recursion"),
        ("vec![]", "vector"),
        ("HashMap", "hashmap"),
        ("HashSet", "hashset"),
    ];

    let code_lower = code.to_lowercase();

    for (pattern, name) in patterns {
        if code_lower.contains(pattern) {
            constructs.push(name.to_string());
        }
    }

    // Remove duplicates while preserving order
    let mut seen = std::collections::HashSet::new();
    constructs.retain(|x| seen.insert(x.clone()));

    constructs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_similarity_score() {
        let a = ProblemSnapshot {
            signature: "fn test(i64) -> i64".to_string(),
            category: "scalar".to_string(),
            arity: 1,
            input_pattern: "I".to_string(),
            output_type: "Int".to_string(),
            recursive_allowed: false,
            tree_input: false,
            num_examples: 2,
            complexity: ProblemComplexity::Simple,
        };

        let b = ProblemSnapshot {
            signature: "fn test(i64) -> i64".to_string(),
            category: "scalar".to_string(),
            arity: 1,
            input_pattern: "I".to_string(),
            output_type: "Int".to_string(),
            recursive_allowed: false,
            tree_input: false,
            num_examples: 3,
            complexity: ProblemComplexity::Medium,
        };

        // Should have high similarity
        let score = similarity_score(&a, &b);
        assert!(score > 0.7);
    }

    #[test]
    fn test_db_create_and_save() {
        let db_path = PathBuf::from("/tmp/test_experience_db.json");
        let _ = std::fs::remove_file(&db_path);

        let mut db = ExperienceDB::new(db_path.clone()).unwrap();
        assert_eq!(db.len(), 0);

        db.save().unwrap();
        assert!(db_path.exists());

        let db2 = ExperienceDB::new(db_path.clone()).unwrap();
        assert_eq!(db2.len(), 0);

        let _ = std::fs::remove_file(&db_path);
    }
}
