use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CreditCategory {
    Comprehension,
    Localization,
    Planning,
    Retrieval,
    ToolUse,
    Synthesis,
    Testing,
    Verification,
    FailureParsing,
    Memory,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreditAssignment {
    pub category: CreditCategory,
    pub score: f64,
    pub evidence: String,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct CreditLedger {
    assignments: Vec<CreditAssignment>,
}

impl CreditLedger {
    pub fn assign(&mut self, category: CreditCategory, score: f64, evidence: impl Into<String>) {
        self.assignments.push(CreditAssignment {
            category,
            score: score.clamp(0.0, 1.0),
            evidence: evidence.into(),
        });
    }

    pub fn assignments(&self) -> &[CreditAssignment] {
        &self.assignments
    }

    pub fn average_for(&self, category: CreditCategory) -> Option<f64> {
        let values: Vec<f64> = self
            .assignments
            .iter()
            .filter(|assignment| assignment.category == category)
            .map(|assignment| assignment.score)
            .collect();
        if values.is_empty() {
            None
        } else {
            Some(values.iter().sum::<f64>() / values.len() as f64)
        }
    }

    pub fn summary(&self) -> Vec<(CreditCategory, f64)> {
        let mut seen = Vec::new();
        for assignment in &self.assignments {
            if !seen.contains(&assignment.category) {
                seen.push(assignment.category);
            }
        }
        seen.into_iter()
            .filter_map(|category| {
                self.average_for(category)
                    .map(|average| (category, average))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_and_summarizes_credit() {
        let mut ledger = CreditLedger::default();
        ledger.assign(CreditCategory::Verification, 0.8, "tests passed");
        ledger.assign(CreditCategory::FailureParsing, 0.6, "classified timeout");
        assert_eq!(ledger.average_for(CreditCategory::Verification), Some(0.8));
        assert_eq!(ledger.summary().len(), 2);
    }
}
