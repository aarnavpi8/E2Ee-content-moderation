//! Loader for the quantised 1-hidden-layer MLP exported by Phase 1 (`train.py`).
//!
//! Architecture (mirrored exactly by the Plonky2 circuit):
//!   h1  = relu(W1 . f + b1)     W1: H x d,  b1: H          (scale S1)
//!   out =        W2 . h1 + b2   W2: H,      b2: scalar      (scale S1*S2)
//!   allowed (ham) iff out >= tau_q

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Model {
    /// Hashed-feature dimension d.
    pub d: usize,
    /// Hidden-layer width H.
    pub hidden_dim: usize,
    /// Layer-1 weights, `w1_q[i][j]` = input j -> hidden i. Shape H x d.
    pub w1_q: Vec<Vec<i64>>,
    /// Layer-1 biases, length H.
    pub b1_q: Vec<i64>,
    /// Layer-2 weights, length H.
    pub w2_q: Vec<i64>,
    /// Layer-2 bias (scalar).
    pub b2_q: i64,
    /// Decision threshold on the output logit (out >= tau_q => "allowed").
    pub tau_q: i64,
}

impl Model {
    pub fn from_json_str(s: &str) -> anyhow::Result<Self> {
        let m: Model = serde_json::from_str(s)?;
        anyhow::ensure!(m.w1_q.len() == m.hidden_dim, "w1_q rows != hidden_dim");
        anyhow::ensure!(m.b1_q.len() == m.hidden_dim, "b1_q len != hidden_dim");
        anyhow::ensure!(m.w2_q.len() == m.hidden_dim, "w2_q len != hidden_dim");
        for (i, row) in m.w1_q.iter().enumerate() {
            anyhow::ensure!(row.len() == m.d, "w1_q[{}] len {} != d {}", i, row.len(), m.d);
        }
        Ok(m)
    }

    pub fn from_json_file(path: &str) -> anyhow::Result<Self> {
        let s = std::fs::read_to_string(path)?;
        Self::from_json_str(&s)
    }

    /// Plain (out-of-circuit) integer MLP forward. Returns the output logit.
    pub fn score(&self, features: &[i64]) -> i64 {
        assert_eq!(features.len(), self.d);
        let mut out = self.b2_q;
        for i in 0..self.hidden_dim {
            let mut acc = self.b1_q[i];
            let row = &self.w1_q[i];
            for j in 0..self.d {
                acc += row[j] * features[j];
            }
            let h1 = acc.max(0); // integer ReLU
            out += self.w2_q[i] * h1;
        }
        out
    }

    /// Plain classifier predicate: true == "allowed" (ham).
    pub fn allowed(&self, features: &[i64]) -> bool {
        self.score(features) >= self.tau_q
    }
}
