use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Resource budget for an agent run (Phase 1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentRunBudget {
    pub max_attempts: u32,
    pub attempts_used: u32,
    pub max_wall_ms: u64,
    pub wall_ms_used: u64,
    pub max_synthesis_candidates: u32,
    pub synthesis_candidates_used: u32,
}

impl Default for AgentRunBudget {
    fn default() -> Self {
        Self {
            max_attempts: 8,
            attempts_used: 0,
            max_wall_ms: 120_000,
            wall_ms_used: 0,
            max_synthesis_candidates: 4,
            synthesis_candidates_used: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetExhausted {
    pub reason: String,
}

impl AgentRunBudget {
    pub fn record_attempt(&mut self) -> Result<(), BudgetExhausted> {
        self.attempts_used += 1;
        if self.attempts_used > self.max_attempts {
            return Err(BudgetExhausted {
                reason: format!(
                    "attempt budget exhausted ({}/{})",
                    self.attempts_used, self.max_attempts
                ),
            });
        }
        Ok(())
    }

    pub fn record_synthesis_candidate(&mut self) -> Result<(), BudgetExhausted> {
        self.synthesis_candidates_used += 1;
        if self.synthesis_candidates_used > self.max_synthesis_candidates {
            return Err(BudgetExhausted {
                reason: format!(
                    "synthesis candidate budget exhausted ({}/{})",
                    self.synthesis_candidates_used, self.max_synthesis_candidates
                ),
            });
        }
        Ok(())
    }

    pub fn tick_wall(&mut self, started: Instant) -> Result<(), BudgetExhausted> {
        self.wall_ms_used = started.elapsed().as_millis() as u64;
        if self.wall_ms_used > self.max_wall_ms {
            return Err(BudgetExhausted {
                reason: format!(
                    "wall clock budget exhausted ({}ms / {}ms)",
                    self.wall_ms_used, self.max_wall_ms
                ),
            });
        }
        Ok(())
    }

    pub fn exhausted(&self) -> bool {
        self.attempts_used >= self.max_attempts
            || self.wall_ms_used >= self.max_wall_ms
            || self.synthesis_candidates_used >= self.max_synthesis_candidates
    }
}
