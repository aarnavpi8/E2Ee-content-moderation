"""
Phase 5 helper - render benchmark plots + a markdown table from the JSON that
the Rust harness (`cargo run -p moderation-core --release --bin bench`) writes to
moderation/models/benchmark_results.json.

Usage:
    py -3.13 moderation/plot_benchmarks.py

Produces (if matplotlib is available):
    moderation/models/bench_proving_time.png
    moderation/models/bench_added_latency.png
and always prints a markdown comparison table to stdout.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "models")
RESULTS = os.path.join(OUT, "benchmark_results.json")


def markdown_table(rows):
    hdr = ("| d | build (ms) | prove med/p95 (ms) | proof (B) | "
           "verify med/p95 (ms) | added latency med/p95 (ms) | speedup vs 3s |")
    sep = "|---|---|---|---|---|---|---|"
    lines = [hdr, sep]
    for r in rows:
        lines.append(
            f"| {r['d']} | {r['build_ms']:.0f} | "
            f"{r['prove_ms_median']:.1f} / {r['prove_ms_p95']:.1f} | "
            f"{r['proof_bytes']} | "
            f"{r['verify_ms_median']:.2f} / {r['verify_ms_p95']:.2f} | "
            f"{r['added_latency_ms_median']:.1f} / {r['added_latency_ms_p95']:.1f} | "
            f"{r['prove_speedup_vs_baseline_3s']:.1f}x |"
        )
    return "\n".join(lines)


def main():
    if not os.path.exists(RESULTS):
        print(f"No benchmark data at {RESULTS}.")
        print("Run:  cargo run -p moderation-core --release --bin bench")
        return
    with open(RESULTS, encoding="utf-8") as fh:
        data = json.load(fh)
    rows = data["results"]

    print("\n### Benchmark results\n")
    print(markdown_table(rows))
    print(f"\n(baseline proof generation from ZK-middlebox literature: "
          f"~{data['baseline_prove_seconds']:.0f} s)\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots "
              "(`py -3.13 -m pip install matplotlib` to enable).")
        return

    dims = [r["d"] for r in rows]

    plt.figure()
    plt.plot(dims, [r["prove_ms_median"] for r in rows], "o-", label="prove median")
    plt.plot(dims, [r["prove_ms_p95"] for r in rows], "s--", label="prove p95")
    plt.xlabel("feature dimension d")
    plt.ylabel("proving time (ms)")
    plt.title("Plonky2 moderation proving time vs d")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUT, "bench_proving_time.png"), dpi=120, bbox_inches="tight")

    plt.figure()
    plt.plot(dims, [r["added_latency_ms_median"] for r in rows], "o-", label="added latency median")
    plt.plot(dims, [r["added_latency_ms_p95"] for r in rows], "s--", label="added latency p95")
    plt.xlabel("feature dimension d")
    plt.ylabel("prove + verify (ms)")
    plt.title("End-to-end added latency vs d")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUT, "bench_added_latency.png"), dpi=120, bbox_inches="tight")

    print(f"Wrote plots to {OUT}")


if __name__ == "__main__":
    main()
