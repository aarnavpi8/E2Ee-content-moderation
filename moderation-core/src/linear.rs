//! Loader and ZK circuit for the quantised linear classifier.
//!
//! Architecture (mirrored exactly by the Plonky2 circuit):
//!   out = theta_q . f + bias_q
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
    FEATURE_ABS_MAX, FEATURE_RANGE_BITS, DIFF_RANGE_BITS,
    NUM_MSG_ELEMS, MAX_MSG_BYTES
};

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

/// The compiled moderation circuit for one specific public linear model.
pub struct Circuit {
    pub model: Model,
    data: CircuitData<F, C, D>,
    f_targets: Vec<Target>,
    msg_targets: Vec<Target>,
    r_target: Target,
}

impl Circuit {
    /// Build the circuit for a given linear model.
    pub fn new(model: Model) -> Self {
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        // ---- Inputs: feature vector, range-checked ------------------------
        let f_targets: Vec<Target> = (0..model.d).map(|_| builder.add_virtual_target()).collect();
        let feat_offset = F::from_canonical_u64(FEATURE_ABS_MAX);
        for &f in &f_targets {
            let shifted = builder.add_const(f, feat_offset);
            builder.range_check(shifted, FEATURE_RANGE_BITS);
        }

        // ---- Linear combination: out = theta_q . f + bias_q ---------------
        let mut out = builder.constant(f_from_i64(model.bias_q));
        for i in 0..model.d {
            let w = builder.constant(f_from_i64(model.theta_q[i]));
            out = builder.mul_add(w, f_targets[i], out);
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
        Self { model, data, f_targets, msg_targets, r_target }
    }

    /// Prove that `message` passes the linear classifier and commit to it under `r`.
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
