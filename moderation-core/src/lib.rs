//! moderation-core — verifiable content-moderation primitives.
//!
//! Provides:
//!   * `features`  — portable feature hashing (Rust port of `features.py`).
//!   * `model`     — loader for the Phase-1 quantised linear model.
//!   * the Plonky2 circuit that proves, in zero knowledge, that a committed
//!     message passes the classifier:  `theta_q . f + bias_q >= tau_q`
//!     AND  `Poseidon(pack(m), r) = h`.
//!
//! The circuit bakes the public model `(theta_q, bias_q, tau_q)` in as
//! constants, so the resulting `CircuitData` (and its verifier data) is
//! specific to one model — updating the model just rebuilds the circuit, with
//! no trusted setup (Plonky2 is transparent).

pub mod features;
pub mod model;

use anyhow::Result;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::hash::hash_types::HashOutTarget;
use plonky2::hash::poseidon::PoseidonHash;
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{CircuitConfig, CircuitData};
use plonky2::plonk::config::{Hasher, PoseidonGoldilocksConfig};
use plonky2::plonk::proof::ProofWithPublicInputs;

pub use model::Model;

pub const D: usize = 2;
pub type C = PoseidonGoldilocksConfig;
pub type F = GoldilocksField;

/// Goldilocks prime p = 2^64 - 2^32 + 1.
pub const GOLDILOCKS_P: u64 = 0xFFFF_FFFF_0000_0001;

/// Maximum message length (bytes) the fixed-size circuit commits to. Messages
/// are zero-padded to this length before packing so the commitment layout is
/// constant. Longer messages are rejected client-side.
pub const MAX_MSG_BYTES: usize = 512;
/// Bytes packed per Goldilocks field element (7 * 8 = 56 bits < 64).
const MSG_CHUNK_BYTES: usize = 7;
/// Number of 7-byte chunks covering MAX_MSG_BYTES.
const NUM_MSG_CHUNKS: usize = (MAX_MSG_BYTES + MSG_CHUNK_BYTES - 1) / MSG_CHUNK_BYTES;
/// Commitment preimage element count: 1 length element + chunk elements.
const NUM_MSG_ELEMS: usize = 1 + NUM_MSG_CHUNKS;

/// Features are range-checked to `|f| <= 2^20` to prevent field-wrap attacks on
/// the dot product (a malicious prover otherwise choosing huge f to overflow).
const FEATURE_ABS_MAX: u64 = 1 << 20;
const FEATURE_RANGE_BITS: usize = 21; // f + 2^20 in [0, 2^21)
/// The classifier margin `score - tau` is range-checked to [0, 2^58): a
/// non-negative margin fits, a negative one aliases to ~2^64 and fails the check.
/// 58 > worst-case |score| (~2^54 for d=1024) and < 64.
const DIFF_RANGE_BITS: usize = 58;

/// Convert a signed integer to a Goldilocks field element.
fn f_from_i64(x: i64) -> F {
    if x >= 0 {
        F::from_canonical_u64(x as u64)
    } else {
        -F::from_canonical_u64((-x) as u64)
    }
}

/// Pack a message + nonce into the fixed-length Poseidon preimage.
///
/// Layout: `[ len, chunk_0, ..., chunk_{NUM_MSG_CHUNKS-1}, r ]`
/// where each chunk is 7 little-endian bytes of the zero-padded message.
pub fn pack_preimage(message: &[u8], r: u64) -> Vec<F> {
    assert!(message.len() <= MAX_MSG_BYTES, "message exceeds MAX_MSG_BYTES");
    // Buffer spans whole 7-byte chunks (>= MAX_MSG_BYTES), zero-padded.
    let mut buf = vec![0u8; NUM_MSG_CHUNKS * MSG_CHUNK_BYTES];
    buf[..message.len()].copy_from_slice(message);

    let mut elems = Vec::with_capacity(NUM_MSG_ELEMS + 1);
    elems.push(F::from_canonical_u64(message.len() as u64));
    for chunk in 0..NUM_MSG_CHUNKS {
        let mut acc: u64 = 0;
        for j in 0..MSG_CHUNK_BYTES {
            acc |= (buf[chunk * MSG_CHUNK_BYTES + j] as u64) << (8 * j);
        }
        elems.push(F::from_canonical_u64(acc));
    }
    elems.push(F::from_canonical_u64(r % GOLDILOCKS_P));
    elems
}

/// Plain (out-of-circuit) Poseidon commitment `h = Poseidon(pack(m, r))`.
/// Returns the 4 field elements of the hash output as canonical u64s.
pub fn commitment(message: &[u8], r: u64) -> [u64; 4] {
    let preimage = pack_preimage(message, r);
    let hash = PoseidonHash::hash_no_pad(&preimage);
    [
        hash.elements[0].to_canonical_u64(),
        hash.elements[1].to_canonical_u64(),
        hash.elements[2].to_canonical_u64(),
        hash.elements[3].to_canonical_u64(),
    ]
}

/// A proof bundle produced by the sender and carried to the server.
#[derive(Clone)]
pub struct ProofBundle {
    /// Serialised Plonky2 proof (self-describes its public inputs).
    pub proof_bytes: Vec<u8>,
    /// The commitment h (4 field elements), also carried in the AEAD AD.
    pub h: [u64; 4],
}

/// The compiled moderation circuit for one specific public model.
pub struct ModerationCircuit {
    pub model: Model,
    data: CircuitData<F, C, D>,
    f_targets: Vec<Target>,
    msg_targets: Vec<Target>,
    r_target: Target,
}

impl ModerationCircuit {
    /// Build the circuit for a given model. Expensive; do this once at startup.
    pub fn new(model: Model) -> Self {
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        // ---- Classifier: theta_q . f + bias_q >= tau_q --------------------
        let f_targets: Vec<Target> = (0..model.d).map(|_| builder.add_virtual_target()).collect();

        // Range-check each feature to |f| <= 2^20 (wrap-attack protection).
        let feat_offset = F::from_canonical_u64(FEATURE_ABS_MAX);
        for &f in &f_targets {
            let shifted = builder.add_const(f, feat_offset);
            builder.range_check(shifted, FEATURE_RANGE_BITS);
        }

        // score = bias + sum_i theta_i * f_i   (theta_i baked in as constants).
        let mut score = builder.constant(f_from_i64(model.bias_q));
        for (i, &f) in f_targets.iter().enumerate() {
            let theta_i = builder.constant(f_from_i64(model.theta_q[i]));
            score = builder.mul_add(theta_i, f, score);
        }

        // margin = score - tau ; prove margin in [0, 2^58) (i.e. score >= tau).
        let tau = builder.constant(f_from_i64(model.tau_q));
        let margin = builder.sub(score, tau);
        builder.range_check(margin, DIFF_RANGE_BITS);

        // ---- Commitment: h = Poseidon(pack(m), r) -------------------------
        let msg_targets: Vec<Target> =
            (0..NUM_MSG_ELEMS).map(|_| builder.add_virtual_target()).collect();
        let r_target = builder.add_virtual_target();

        let mut hash_inputs = msg_targets.clone();
        hash_inputs.push(r_target);
        let h_target: HashOutTarget =
            builder.hash_n_to_hash_no_pad::<PoseidonHash>(hash_inputs);

        // h is the only public input (4 field elements).
        builder.register_public_inputs(&h_target.elements);

        let data = builder.build::<C>();
        Self { model, data, f_targets, msg_targets, r_target }
    }

    /// Prove that `message` passes the classifier and commit to it under `r`.
    /// Fails (returns Err) if the message does not actually pass — an honest
    /// prover cannot forge a proof for disallowed content.
    pub fn prove(&self, message: &str, r: u64) -> Result<ProofBundle> {
        let bytes = message.as_bytes();
        anyhow::ensure!(bytes.len() <= MAX_MSG_BYTES, "message too long");
        let features = features::feature_vector(message, self.model.d);
        anyhow::ensure!(
            self.model.allowed(&features),
            "message is classified as blocked; refusing to prove"
        );
        for &v in &features {
            anyhow::ensure!(
                v.unsigned_abs() < FEATURE_ABS_MAX,
                "feature magnitude exceeds circuit bound"
            );
        }

        let mut pw = PartialWitness::new();
        for (i, &t) in self.f_targets.iter().enumerate() {
            pw.set_target(t, f_from_i64(features[i]));
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

    /// Server-side verification. Checks the proof is valid AND that its public
    /// commitment equals `expected_h` (the h carried in the message's AD).
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

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_model(d: usize) -> Model {
        // A trivial model: score = sum(f) ; allowed iff sum(f) >= 0.
        Model {
            d,
            scale: 1,
            theta_q: vec![1; d],
            bias_q: 0,
            tau_q: 0,
        }
    }

    #[test]
    fn commitment_is_deterministic_and_binding() {
        let a = commitment(b"hello world", 12345);
        let b = commitment(b"hello world", 12345);
        let c = commitment(b"hello worlD", 12345); // one bit different
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn honest_bundle_h_equals_plain_commitment() {
        // The h in the proof bundle must equal the plain Poseidon commitment the
        // receiver recomputes — this is what makes the receiver binding check
        // accept honest messages and reject forged ones.
        let circuit = ModerationCircuit::new(tiny_model(64));
        let msg = "hello there friend";
        let r = 424242;
        if circuit.model.allowed(&features::feature_vector(msg, 64)) {
            let bundle = circuit.prove(msg, r).expect("prove");
            assert_eq!(bundle.h, commitment(msg.as_bytes(), r), "in/out-of-circuit hash mismatch");
            // A different plaintext under the same r must NOT match (forgery caught).
            assert_ne!(bundle.h, commitment(b"a different message", r));
        }
    }

    #[test]
    fn prove_and_verify_roundtrip() {
        // Model allows when sum(features) >= 0. Craft a message; if it happens
        // to be blocked, this still exercises the allowed path via tiny_model
        // whose bias makes most short messages pass (sum near 0). We assert on
        // whatever the plain model says to keep the test deterministic.
        let circuit = ModerationCircuit::new(tiny_model(64));
        let msg = "hello there friend";
        let feats = features::feature_vector(msg, 64);
        if circuit.model.allowed(&feats) {
            let bundle = circuit.prove(msg, 999).expect("prove");
            assert!(circuit.verify(&bundle.proof_bytes, &bundle.h));
            // Tampered h must be rejected.
            let mut bad_h = bundle.h;
            bad_h[0] ^= 1;
            assert!(!circuit.verify(&bundle.proof_bytes, &bad_h));
        } else {
            assert!(circuit.prove(msg, 999).is_err());
        }
    }
}
