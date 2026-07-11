//! moderation-core — verifiable content-moderation primitives.
//!
//! Provides:
//!   * `features`  — portable feature hashing (Rust port of `features.py`).
//!   * `linear`    — loader and ZK circuit for the linear model.
//!   * `mlp`       — loader and ZK circuit for the 1-hidden-layer MLP model.

pub mod features;
pub mod linear;
pub mod mlp;

use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::hash::poseidon::PoseidonHash;
use plonky2::plonk::config::{Hasher, PoseidonGoldilocksConfig};

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
/// ReLU decomposition parts `pos, neg` are range-checked to [0, 2^48); this
/// bounds the hidden pre-activations (worst case ~2^42) and keeps them wrap-safe.
const RELU_RANGE_BITS: usize = 48;
/// The output margin `out - tau` is range-checked to [0, 2^58): a non-negative
/// margin fits, a negative one aliases to ~2^64 and fails. 58 > worst-case
/// |out| (~2^52) and < 64.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commitment_is_deterministic_and_binding() {
        let a = commitment(b"hello world", 12345);
        let b = commitment(b"hello world", 12345);
        let c = commitment(b"hello worlD", 12345); // one bit different
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
