//! Cross-language parity test: the Rust feature-hashing + linear classifier
//! must reproduce, bit-for-bit, the values Python wrote into
//! `moderation/models/test_vectors.json` (Phase 1). This is the guardrail that
//! keeps the circuit's notion of the model in lock-step with the trained one.

use std::collections::BTreeMap;

use moderation_core::{features::feature_vector, Model};
use serde::Deserialize;

#[derive(Deserialize)]
struct TestVector {
    message: String,
    d: usize,
    nonzero_features: BTreeMap<String, i64>,
    score_q: i64,
    prediction: String,
}

#[derive(Deserialize)]
struct TestVectors {
    model: String,
    vectors: Vec<TestVector>,
}

fn models_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("moderation")
        .join("models")
}

#[test]
fn rust_matches_python_reference() {
    let dir = models_dir();
    let tv_path = dir.join("test_vectors.json");
    let tv: TestVectors =
        serde_json::from_str(&std::fs::read_to_string(&tv_path).expect("test_vectors.json"))
            .expect("parse test_vectors");
    let model = Model::from_json_file(dir.join(&tv.model).to_str().unwrap())
        .expect("load model");

    for v in &tv.vectors {
        let phi = feature_vector(&v.message, v.d);

        // 1. Non-zero feature indices/values must match Python exactly.
        let mut rust_nz: BTreeMap<String, i64> = BTreeMap::new();
        for (i, &val) in phi.iter().enumerate() {
            if val != 0 {
                rust_nz.insert(i.to_string(), val);
            }
        }
        assert_eq!(rust_nz, v.nonzero_features, "feature mismatch for {:?}", v.message);

        // 2. Quantised score must match.
        assert_eq!(model.score(&phi), v.score_q, "score mismatch for {:?}", v.message);

        // 3. Prediction label must match.
        let pred = if model.allowed(&phi) { "ham/allowed" } else { "spam/blocked" };
        assert_eq!(pred, v.prediction, "prediction mismatch for {:?}", v.message);
    }
}
