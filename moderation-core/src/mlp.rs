//! Loader and ZK circuit for the quantised 1-hidden-layer MLP.
//!
//! Architecture (mirrored exactly by the Plonky2 circuit):
//!   h1  = relu(W1 . f + b1)     W1: H x d,  b1: H          (scale S1)
//!   out =        W2 . h1 + b2   W2: H,      b2: scalar      (scale S1*S2)
//!   allowed (ham) iff out >= tau_q

use anyhow::Result;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::hash::hash_types::HashOutTarget;
use plonky2::hash::poseidon::PoseidonHash;
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{CircuitConfig, CircuitData};
use plonky2::plonk::proof::ProofWithPublicInputs;
use serde::Deserialize;

use crate::{
    features, f_from_i64, pack_preimage, ProofBundle, C, D, F,
    FEATURE_ABS_MAX, FEATURE_RANGE_BITS, RELU_RANGE_BITS, DIFF_RANGE_BITS,
    NUM_MSG_ELEMS, MAX_MSG_BYTES
};

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

/// The compiled moderation circuit for one specific public MLP model.
pub struct Circuit {
    pub model: Model,
    data: CircuitData<F, C, D>,
    f_targets: Vec<Target>,
    /// Per-hidden-neuron ReLU positive/negative parts (set at proving time).
    relu_pos: Vec<Target>,
    relu_neg: Vec<Target>,
    msg_targets: Vec<Target>,
    r_target: Target,
}

impl Circuit {
    /// Build the circuit for a given MLP model. Expensive; do once at startup.
    pub fn new(model: Model) -> Self {
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        let h_dim = model.hidden_dim;

        // ---- Inputs: feature vector, range-checked ------------------------
        let f_targets: Vec<Target> = (0..model.d).map(|_| builder.add_virtual_target()).collect();
        let feat_offset = F::from_canonical_u64(FEATURE_ABS_MAX);
        for &f in &f_targets {
            let shifted = builder.add_const(f, feat_offset);
            builder.range_check(shifted, FEATURE_RANGE_BITS);
        }

        // ---- Layer 1 + ReLU:  h1_i = relu( W1_i . f + b1_i ) --------------
        let zero = builder.zero();
        let mut relu_pos = Vec::with_capacity(h_dim);
        let mut relu_neg = Vec::with_capacity(h_dim);
        let mut h1 = Vec::with_capacity(h_dim);
        for i in 0..h_dim {
            let mut acc = builder.constant(f_from_i64(model.b1_q[i]));
            for j in 0..model.d {
                let w = builder.constant(f_from_i64(model.w1_q[i][j]));
                acc = builder.mul_add(w, f_targets[j], acc);
            }
            // ReLU gadget: acc = pos - neg, with pos,neg >= 0 and pos*neg = 0.
            let pos = builder.add_virtual_target();
            let neg = builder.add_virtual_target();
            builder.range_check(pos, RELU_RANGE_BITS);
            builder.range_check(neg, RELU_RANGE_BITS);
            let diff = builder.sub(pos, neg);
            builder.connect(acc, diff);
            let prod = builder.mul(pos, neg);
            builder.connect(prod, zero);

            relu_pos.push(pos);
            relu_neg.push(neg);
            h1.push(pos); // relu(acc) = pos
        }

        // ---- Layer 2:  out = W2 . h1 + b2 ---------------------------------
        let mut out = builder.constant(f_from_i64(model.b2_q));
        for i in 0..h_dim {
            let w = builder.constant(f_from_i64(model.w2_q[i]));
            out = builder.mul_add(w, h1[i], out);
        }

        // margin = out - tau ; prove margin in [0, 2^58) (i.e. out >= tau).
        let tau = builder.constant(f_from_i64(model.tau_q));
        let margin = builder.sub(out, tau);
        builder.range_check(margin, DIFF_RANGE_BITS);

        // ---- Commitment: h = Poseidon(pack(m), r) -------------------------
        let msg_targets: Vec<Target> =
            (0..NUM_MSG_ELEMS).map(|_| builder.add_virtual_target()).collect();
        let r_target = builder.add_virtual_target();

        let mut hash_inputs = msg_targets.clone();
        hash_inputs.push(r_target);
        let h_target: HashOutTarget =
            builder.hash_n_to_hash_no_pad::<PoseidonHash>(hash_inputs);
        builder.register_public_inputs(&h_target.elements);

        let data = builder.build::<C>();
        Self { model, data, f_targets, relu_pos, relu_neg, msg_targets, r_target }
    }

    /// Prove that `message` passes the MLP classifier and commit to it under `r`.
    pub fn prove(&self, message: &str, r: u64) -> Result<ProofBundle> {
        let bytes = message.as_bytes();
        anyhow::ensure!(bytes.len() <= MAX_MSG_BYTES, "message too long");
        let m = &self.model;
        let features = features::feature_vector(message, m.d);
        anyhow::ensure!(
            m.allowed(&features),
            "message is classified as blocked; refusing to prove"
        );
        for &v in &features {
            anyhow::ensure!(
                v.unsigned_abs() < FEATURE_ABS_MAX,
                "feature magnitude exceeds circuit bound"
            );
        }

        let mut pw = PartialWitness::new();
        for (j, &t) in self.f_targets.iter().enumerate() {
            pw.set_target(t, f_from_i64(features[j]));
        }

        // Recompute hidden pre-activations to fill the ReLU pos/neg witnesses.
        let relu_bound = 1i64 << RELU_RANGE_BITS;
        for i in 0..m.hidden_dim {
            let mut acc = m.b1_q[i];
            let row = &m.w1_q[i];
            for j in 0..m.d {
                acc += row[j] * features[j];
            }
            anyhow::ensure!(acc.abs() < relu_bound, "hidden pre-activation exceeds ReLU bound");
            let (pos_v, neg_v) = if acc >= 0 { (acc, 0) } else { (0, -acc) };
            pw.set_target(self.relu_pos[i], f_from_i64(pos_v));
            pw.set_target(self.relu_neg[i], f_from_i64(neg_v));
        }

        let preimage = pack_preimage(bytes, r);
        for (i, &t) in self.msg_targets.iter().enumerate() {
            pw.set_target(t, preimage[i]);
        }
        pw.set_target(self.r_target, *preimage.last().unwrap());

        let proof = self.data.prove(pw)?;
        let h = [
            proof.public_inputs[0].to_canonical_u64(),
            proof.public_inputs[1].to_canonical_u64(),
            proof.public_inputs[2].to_canonical_u64(),
            proof.public_inputs[3].to_canonical_u64(),
        ];
        Ok(ProofBundle { proof_bytes: proof.to_bytes(), h })
    }

    /// Server-side verification.
    pub fn verify(&self, proof_bytes: &[u8], expected_h: &[u64; 4]) -> bool {
        let proof = match ProofWithPublicInputs::<F, C, D>::from_bytes(
            proof_bytes.to_vec(),
            &self.data.common,
        ) {
            Ok(p) => p,
            Err(_) => return false,
        };
        if proof.public_inputs.len() < 4 {
            return false;
        }
        for i in 0..4 {
            if proof.public_inputs[i].to_canonical_u64() != expected_h[i] {
                return false;
            }
        }
        self.data.verify(proof).is_ok()
    }
}
