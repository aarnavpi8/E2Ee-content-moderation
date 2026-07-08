//! Loader for the quantised linear model exported by Phase 1 (`train.py`).

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Model {
    /// Hashed-feature dimension d.
    pub d: usize,
    /// Fixed-point scale used when quantising the float weights.
    pub scale: i64,
    /// Quantised weight vector, length `d`.
    pub theta_q: Vec<i64>,
    /// Quantised bias.
    pub bias_q: i64,
    /// Quantised threshold (score >= tau_q  =>  "allowed").
    pub tau_q: i64,
}

impl Model {
    pub fn from_json_str(s: &str) -> anyhow::Result<Self> {
        let m: Model = serde_json::from_str(s)?;
        anyhow::ensure!(
            m.theta_q.len() == m.d,
            "theta_q length {} != d {}",
            m.theta_q.len(),
            m.d
        );
        Ok(m)
    }

    pub fn from_json_file(path: &str) -> anyhow::Result<Self> {
        let s = std::fs::read_to_string(path)?;
        Self::from_json_str(&s)
    }

    /// Plain (out-of-circuit) integer score: theta_q . f + bias_q.
    pub fn score(&self, features: &[i64]) -> i64 {
        assert_eq!(features.len(), self.d);
        let mut acc: i64 = self.bias_q;
        for (w, x) in self.theta_q.iter().zip(features.iter()) {
            acc += w * x;
        }
        acc
    }

    /// Plain classifier predicate: true == "allowed" (ham).
    pub fn allowed(&self, features: &[i64]) -> bool {
        self.score(features) >= self.tau_q
    }
}
