"""
Comparative Benchmarking Plotting Tool.

Loads benchmark results from both the Linear and MLP models and produces:
1. A markdown table comparing their execution time, proof size, and latency.
2. Comparative plots (proving time and added latency) saved in the moderation directory.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LINEAR_RESULTS = os.path.join(HERE, "linear", "models", "benchmark_results.json")
MLP_RESULTS = os.path.join(HERE, "mlp", "models", "benchmark_results.json")


def load_results(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def print_comparison_table(linear, mlp):
    l_rows = {r["d"]: r for r in linear["results"]}
    m_rows = {r["d"]: r for r in mlp["results"]}

    hdr = ("| d | Model | Build (ms) | Prove Med/P95 (ms) | Proof Size (KB) | Verify Med/P95 (ms) | Added Lat Med/P95 (ms) | Speedup vs 3s |")
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [hdr, sep]

    for d in sorted(l_rows.keys()):
        lr = l_rows[d]
        mr = m_rows[d]

        lines.append(
            f"| {d} | **Linear** | {lr['build_ms']:.1f} | "
            f"{lr['prove_ms_median']:.1f} / {lr['prove_ms_p95']:.1f} | "
            f"{lr['proof_bytes'] / 1024.0:.1f} KB | "
            f"{lr['verify_ms_median']:.2f} / {lr['verify_ms_p95']:.2f} | "
            f"{lr['added_latency_ms_median']:.1f} / {lr['added_latency_ms_p95']:.1f} | "
            f"{lr['prove_speedup_vs_baseline_3s']:.1f}x |"
        )
        lines.append(
            f"| {d} | **MLP** | {mr['build_ms']:.1f} | "
            f"{mr['prove_ms_median']:.1f} / {mr['prove_ms_p95']:.1f} | "
            f"{mr['proof_bytes'] / 1024.0:.1f} KB | "
            f"{mr['verify_ms_median']:.2f} / {mr['verify_ms_p95']:.2f} | "
            f"{mr['added_latency_ms_median']:.1f} / {mr['added_latency_ms_p95']:.1f} | "
            f"{mr['prove_speedup_vs_baseline_3s']:.1f}x |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")

    # remove last separator line if redundant
    if lines[-1] == "|---|---|---|---|---|---|---|---|":
        lines.pop()

    return "\n".join(lines)


def main():
    linear = load_results(LINEAR_RESULTS)
    mlp = load_results(MLP_RESULTS)

    if not linear or not mlp:
        print("Missing benchmark results. Make sure to run the Rust bench binary first:")
        print("  cargo +nightly run -p moderation-core --release --bin bench")
        return

    print("\n### Comparative ZK Content-Moderation Benchmarks")
    print(print_comparison_table(linear, mlp))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed; skipping plots.")
        return

    dims = sorted([r["d"] for r in linear["results"]])
    
    l_prov_med = [next(r["prove_ms_median"] for r in linear["results"] if r["d"] == d) for d in dims]
    l_prov_p95 = [next(r["prove_ms_p95"] for r in linear["results"] if r["d"] == d) for d in dims]
    m_prov_med = [next(r["prove_ms_median"] for r in mlp["results"] if r["d"] == d) for d in dims]
    m_prov_p95 = [next(r["prove_ms_p95"] for r in mlp["results"] if r["d"] == d) for d in dims]

    l_lat_med = [next(r["added_latency_ms_median"] for r in linear["results"] if r["d"] == d) for d in dims]
    l_lat_p95 = [next(r["added_latency_ms_p95"] for r in linear["results"] if r["d"] == d) for d in dims]
    m_lat_med = [next(r["added_latency_ms_median"] for r in mlp["results"] if r["d"] == d) for d in dims]
    m_lat_p95 = [next(r["added_latency_ms_p95"] for r in mlp["results"] if r["d"] == d) for d in dims]

    # Plot 1: Proving Time Comparison
    plt.figure(figsize=(7, 5))
    plt.plot(dims, l_prov_med, "o-", color="C0", label="Linear (median)")
    plt.plot(dims, l_prov_p95, "o--", color="C0", alpha=0.6, label="Linear (p95)")
    plt.plot(dims, m_prov_med, "s-", color="C1", label="MLP (median)")
    plt.plot(dims, m_prov_p95, "s--", color="C1", alpha=0.6, label="MLP (p95)")
    plt.xscale("log")
    plt.xticks(dims, [str(d) for d in dims])
    plt.xlabel("Hashed Feature Dimension d")
    plt.ylabel("Proving Time (ms)")
    plt.title("Proving Time Comparison: Linear vs MLP (Plonky2)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(HERE, "comparative_proving_time.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Added Latency Comparison
    plt.figure(figsize=(7, 5))
    plt.plot(dims, l_lat_med, "o-", color="C0", label="Linear (median)")
    plt.plot(dims, l_lat_p95, "o--", color="C0", alpha=0.6, label="Linear (p95)")
    plt.plot(dims, m_lat_med, "s-", color="C1", label="MLP (median)")
    plt.plot(dims, m_lat_p95, "s--", color="C1", alpha=0.6, label="MLP (p95)")
    plt.xscale("log")
    plt.xticks(dims, [str(d) for d in dims])
    plt.xlabel("Hashed Feature Dimension d")
    plt.ylabel("End-to-End Latency (ms)")
    plt.title("End-to-End Latency Comparison: Linear vs MLP")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(HERE, "comparative_added_latency.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved plots to:\n  - {os.path.join(HERE, 'comparative_proving_time.png')}\n  - {os.path.join(HERE, 'comparative_added_latency.png')}")


if __name__ == "__main__":
    main()
