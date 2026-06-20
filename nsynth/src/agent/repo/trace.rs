use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentStep {
    pub sequence: usize,
    pub name: String,
    pub input: String,
    pub output: String,
    pub result: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AgentTrace {
    steps: Vec<AgentStep>,
}

impl AgentTrace {
    pub fn push(
        &mut self,
        name: impl Into<String>,
        input: impl Into<String>,
        output: impl Into<String>,
        result: impl Into<String>,
    ) {
        let sequence = self.steps.len() + 1;
        self.steps.push(AgentStep {
            sequence,
            name: name.into(),
            input: input.into(),
            output: output.into(),
            result: result.into(),
        });
    }

    pub fn steps(&self) -> &[AgentStep] {
        &self.steps
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_ordered_steps() {
        let mut trace = AgentTrace::default();
        trace.push("mine", "repo", "signals", "ok");
        trace.push("gate", "diff", "paths", "ok");
        assert_eq!(trace.len(), 2);
        assert_eq!(trace.steps()[0].sequence, 1);
    }
}
