//! Phase 5 - Benchmarking harness for both Linear and MLP classifiers.
//!
//! Measures the systems cost of verifiable moderation across feature
//! dimensions d in {64, 256, 1024}:
//!   * circuit build time (one-time)
//!   * proving time        (per message)   median + p95
//!   * proof size          (bytes)
//!   * verification time   (per message)   median + p95
//!   * end-to-end added latency (prove + verify) median + p95

use std::time::Instant;

const DIMS: [usize; 3] = [64, 256, 1024];
const ITERS: usize = 30;
const BASELINE_PROVE_S: f64 = 3.0;

const CANDIDATES: [&str; 5] = [
    "ok see you tomorrow",
    "hey are we still on for lunch",
    "sounds good talk later",
    "thanks let me know",
    "great see you soon",
];

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p * (sorted.len() as f64 - 1.0)).round() as usize).min(sorted.len() - 1);
    sorted[idx]
}

fn median(sorted: &[f64]) -> f64 {
    percentile(sorted, 0.5)
}

fn benchmark_linear() {
    println!("\n=== Benchmarking Linear Circuit ===");
    println!(
        "{:>6} | {:>10} | {:>14} | {:>10} | {:>16} | {:>18}",
        "d", "build(ms)", "prove med/p95", "proof(B)", "verify med/p95", "added lat med/p95"
    );
    println!("{}", "-".repeat(92));

    let mut json_rows = Vec::new();

    for &d in &DIMS {
        let path = format!("moderation/linear/models/model_d{}.json", d);
        let model = match moderation_core::linear::Model::from_json_file(&path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("skip d={}: {} ({})", d, e, path);
                continue;
            }
        };

        let t0 = Instant::now();
        let circuit = moderation_core::linear::Circuit::new(model);
        let build_ms = t0.elapsed().as_secs_f64() * 1e3;

        let msg = match CANDIDATES.iter().copied().find(|m| {
            let f = moderation_core::features::feature_vector(m, circuit.model.d);
            circuit.model.allowed(&f)
        }) {
            Some(m) => m,
            None => {
                eprintln!("skip d={}: no benign candidate is allowed by this model", d);
                continue;
            }
        };

        let mut prove_ms = Vec::with_capacity(ITERS);
        let mut verify_ms = Vec::with_capacity(ITERS);
        let mut total_ms = Vec::with_capacity(ITERS);
        let mut proof_size = 0usize;

        for i in 0..ITERS {
            let r = 1000 + i as u64;

            let tp = Instant::now();
            let bundle = circuit.prove(msg, r).expect("prove");
            let p_ms = tp.elapsed().as_secs_f64() * 1e3;

            proof_size = bundle.proof_bytes.len();

            let tv = Instant::now();
            let ok = circuit.verify(&bundle.proof_bytes, &bundle.h);
            let v_ms = tv.elapsed().as_secs_f64() * 1e3;
            assert!(ok, "verification failed");

            prove_ms.push(p_ms);
            verify_ms.push(v_ms);
            total_ms.push(p_ms + v_ms);
        }

        prove_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        verify_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        total_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let (p_med, p_p95) = (median(&prove_ms), percentile(&prove_ms, 0.95));
        let (v_med, v_p95) = (median(&verify_ms), percentile(&verify_ms, 0.95));
        let (t_med, t_p95) = (median(&total_ms), percentile(&total_ms, 0.95));

        println!(
            "{:>6} | {:>10.1} | {:>6.1}/{:>6.1} | {:>10} | {:>7.2}/{:>7.2} | {:>8.1}/{:>8.1}",
            d, build_ms, p_med, p_p95, proof_size, v_med, v_p95, t_med, t_p95
        );

        json_rows.push(serde_json::json!({
            "d": d,
            "build_ms": build_ms,
            "prove_ms_median": p_med,
            "prove_ms_p95": p_p95,
            "proof_bytes": proof_size,
            "verify_ms_median": v_med,
            "verify_ms_p95": v_p95,
            "added_latency_ms_median": t_med,
            "added_latency_ms_p95": t_p95,
            "prove_speedup_vs_baseline_3s": BASELINE_PROVE_S / (p_med / 1e3),
        }));
    }

    let summary = serde_json::json!({
        "iters": ITERS,
        "baseline_prove_seconds": BASELINE_PROVE_S,
        "results": json_rows,
    });
    let out = "moderation/linear/models/benchmark_results.json";
    if let Err(e) = std::fs::write(out, serde_json::to_string_pretty(&summary).unwrap()) {
        eprintln!("could not write {}: {}", out, e);
    } else {
        println!("Wrote {}", out);
    }
}

fn benchmark_mlp() {
    println!("\n=== Benchmarking MLP Circuit ===");
    println!(
        "{:>6} | {:>10} | {:>14} | {:>10} | {:>16} | {:>18}",
        "d", "build(ms)", "prove med/p95", "proof(B)", "verify med/p95", "added lat med/p95"
    );
    println!("{}", "-".repeat(92));

    let mut json_rows = Vec::new();

    for &d in &DIMS {
        let path = format!("moderation/mlp/models/model_d{}.json", d);
        let model = match moderation_core::mlp::Model::from_json_file(&path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("skip d={}: {} ({})", d, e, path);
                continue;
            }
        };

        let t0 = Instant::now();
        let circuit = moderation_core::mlp::Circuit::new(model);
        let build_ms = t0.elapsed().as_secs_f64() * 1e3;

        let msg = match CANDIDATES.iter().copied().find(|m| {
            let f = moderation_core::features::feature_vector(m, circuit.model.d);
            circuit.model.allowed(&f)
        }) {
            Some(m) => m,
            None => {
                eprintln!("skip d={}: no benign candidate is allowed by this model", d);
                continue;
            }
        };

        let mut prove_ms = Vec::with_capacity(ITERS);
        let mut verify_ms = Vec::with_capacity(ITERS);
        let mut total_ms = Vec::with_capacity(ITERS);
        let mut proof_size = 0usize;

        for i in 0..ITERS {
            let r = 1000 + i as u64;

            let tp = Instant::now();
            let bundle = circuit.prove(msg, r).expect("prove");
            let p_ms = tp.elapsed().as_secs_f64() * 1e3;

            proof_size = bundle.proof_bytes.len();

            let tv = Instant::now();
            let ok = circuit.verify(&bundle.proof_bytes, &bundle.h);
            let v_ms = tv.elapsed().as_secs_f64() * 1e3;
            assert!(ok, "verification failed");

            prove_ms.push(p_ms);
            verify_ms.push(v_ms);
            total_ms.push(p_ms + v_ms);
        }

        prove_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        verify_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        total_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let (p_med, p_p95) = (median(&prove_ms), percentile(&prove_ms, 0.95));
        let (v_med, v_p95) = (median(&verify_ms), percentile(&verify_ms, 0.95));
        let (t_med, t_p95) = (median(&total_ms), percentile(&total_ms, 0.95));

        println!(
            "{:>6} | {:>10.1} | {:>6.1}/{:>6.1} | {:>10} | {:>7.2}/{:>7.2} | {:>8.1}/{:>8.1}",
            d, build_ms, p_med, p_p95, proof_size, v_med, v_p95, t_med, t_p95
        );

        json_rows.push(serde_json::json!({
            "d": d,
            "build_ms": build_ms,
            "prove_ms_median": p_med,
            "prove_ms_p95": p_p95,
            "proof_bytes": proof_size,
            "verify_ms_median": v_med,
            "verify_ms_p95": v_p95,
            "added_latency_ms_median": t_med,
            "added_latency_ms_p95": t_p95,
            "prove_speedup_vs_baseline_3s": BASELINE_PROVE_S / (p_med / 1e3),
        }));
    }

    let summary = serde_json::json!({
        "iters": ITERS,
        "baseline_prove_seconds": BASELINE_PROVE_S,
        "results": json_rows,
    });
    let out = "moderation/mlp/models/benchmark_results.json";
    if let Err(e) = std::fs::write(out, serde_json::to_string_pretty(&summary).unwrap()) {
        eprintln!("could not write {}: {}", out, e);
    } else {
        println!("Wrote {}", out);
    }
}

fn main() {
    benchmark_linear();
    benchmark_mlp();
}
