# Verifiable Client-Side Content Moderation over an E2EE Messenger

*Paper draft (Phase 7). Assembles the design (Phases 0, 2), the classifier
baseline (Phase 1), the ZK circuit (Phase 3), the CLI integration (Phase 4),
the benchmarks (Phase 5), and the adversarial evaluation (Phase 6).*

## Abstract

We add a *verifiable* content-moderation gate to a from-scratch Signal-style
end-to-end encrypted messenger (X3DH + Double Ratchet, in Rust). A sender proves
in zero knowledge — using Plonky2 — that the plaintext of a message passes a
public linear classifier, and commits to the content with a Poseidon hash. The
relay server acts as a zero-knowledge middlebox: it verifies the proof and
relays the ciphertext only if the proof is valid, without ever seeing the
plaintext. The recipient re-derives the commitment from the decrypted message,
making any sender who proves one message but encrypts another cryptographically
detectable and attributable. We report classifier baselines, proving-system
costs, and adversarial evasion rates, and we are explicit about what the design
does *not* solve — most importantly model governance and client substitution.

## 1. Introduction

End-to-end encryption removes the server's ability to inspect content, which is
also what makes server-side moderation impossible. We ask whether moderation can
be made *verifiable* rather than *visible*: can a server enforce that only
policy-compliant messages are relayed, while remaining plaintext-blind? Our
answer combines proactive zero-knowledge gating with reactive cryptographic
attribution.

## 2. System and threat model

See `phase2_protocol_design.md` for the full treatment. In brief:

* **Sender** computes `f = φ(m)` (feature hashing), samples nonce `r`, and
  produces a Plonky2 proof of `C_θ(f) = 1 ∧ Poseidon(pack(m), r) = h`. It
  encrypts `m` with `AD = (h, r)` and sends `(ciphertext, tag, h, r, π)`.
* **Server** verifies `π` against the public model and checks the proof's
  committed `h` equals the envelope `h`; relays on success, drops otherwise.
  It never decrypts.
* **Receiver** decrypts (AD binds `h, r` to the tag), recomputes
  `Poseidon(pack(m), r)`, and rejects on mismatch.

Guarantees (G1–G5) and non-goals (N1–N6) are enumerated in the design doc; the
headline non-goals are **model governance** (whoever controls the model controls
what is censored) and **client substitution** (a modified client can decouple
the proved features from the encrypted content; the receiver check still catches
`m`-vs-`h` mismatches but cannot force honest feature evaluation).

## 3. Classifier baseline (Phase 1)

Linear logistic-regression classifier over the portable signed feature-hash
`φ` (`moderation/features.py`), trained on the SMS Spam Collection (5,574
messages; 4,827 ham / 747 spam), 80/20 stratified split. Weights are quantised
to integers at scale `2^16`; quantisation is **lossless** relative to the float
model at every dimension (features are integers, so the integer dot product is
exact). Label 1 = ham ("allowed"); spam is the moderation-relevant positive class.

| d | accuracy | spam precision | spam recall | spam F1 |
|---|---|---|---|---|
| 64 | 0.9148 | 0.7288 | 0.5772 | 0.6442 |
| 256 | 0.9695 | 0.9021 | 0.8658 | 0.8836 |
| 1024 | 0.9749 | 0.9618 | 0.8456 | 0.9000 |

`d = 256` is the deployment default: it captures almost all of the accuracy of
`d = 1024` at a quarter of the circuit width. The integer model exported to JSON
is exactly the predicate the ZK circuit enforces; a Rust↔Python parity test
(`moderation-core/tests/parity.rs`) checks feature vectors, scores, and
predictions agree bit-for-bit.

## 4. Circuit (Phase 3)

The Plonky2 circuit (Goldilocks field) enforces, with the public model baked in
as constants:

1. **Feature range checks** — each `f_i` is proven in `|f_i| ≤ 2^20`, preventing
   a malicious prover from overflowing the field to fake a passing score.
2. **Classifier** — `score = Σ θ_q[i]·f_i + b_q`; the margin `score − τ_q` is
   range-checked to `[0, 2^58)`, which holds iff `score ≥ τ_q` (a negative margin
   aliases to ≈`2^64` and fails the check).
3. **Commitment** — the message is packed 7 bytes per field element (fixed-size,
   length-prefixed, zero-padded to 512 bytes) and hashed with native Poseidon
   together with `r`; the 4-element digest is the sole public input `h`.

Poseidon being native to Plonky2 keeps the hash nearly free in-circuit — the
dominant cost that a non-native hash (e.g. SHA-256) would have imposed is
avoided (see `phase0_feasibility.md`).

## 5. Benchmarks (Phase 5)

Harness: `cargo run -p moderation-core --release --bin bench` measures build
time, proving time (median/p95), proof size, and verification time across
`d ∈ {64, 256, 1024}`, and writes `moderation/models/benchmark_results.json`
(render with `moderation/plot_benchmarks.py`). Measured on a desktop-class CPU
(Windows, `x86_64-pc-windows-gnu`, release build, 30 iterations):

| d | build (ms) | prove med/p95 (ms) | proof (B) | verify med/p95 (ms) | added latency med/p95 (ms) | speedup vs 3 s |
|---|---|---|---|---|---|---|
| 64 | 45 | 20.7 / 41.0 | 94,060 | 2.69 / 3.19 | 23.4 / 43.7 | ~145× |
| 256 | 46 | 41.7 / 49.5 | 103,404 | 2.99 / 3.65 | 44.6 / 52.7 | ~72× |
| 1024 | 111 | 110.5 / 141.5 | 121,480 | 3.69 / 4.58 | 114.6 / 146.1 | ~27× |

The `d = 256` deployment default proves in **~42 ms median** — roughly **70×
faster** than the ~3 s proof-generation baseline reported in the ZK-middlebox
literature — with **~3 ms** verification and ~100 KB proofs. Proving time scales
sub-linearly-to-linearly in `d` as expected (the dot product dominates);
verification stays near-constant (a few ms). End-to-end added latency for a
message is dominated by proving and stays well under 150 ms even at `d = 1024`.
This empirically confirms the Phase 0 GO decision.

## 6. Adversarial evaluation (Phase 6)

Evasion rate = fraction of the 672 spam messages that `d = 256` correctly blocks
which a budget-`k` perturbation (k = max tokens edited) flips to "allowed",
targeting the most spam-indicative tokens first.

| perturbation | k=1 | k=2 | k=3 |
|---|---|---|---|
| synonym substitution | 4.3% | 7.0% | 7.9% |
| homoglyph insertion | 17.1% | 34.1% | 49.3% |
| whitespace / zero-width insertion | 16.4% | 31.6% | 41.4% |

Findings: character-level attacks (homoglyphs, zero-width splits) are far more
effective than synonym substitution against a bag-of-hashed-tokens linear model,
because they move a high-weight token to a *different* hash bucket with a single
edit. At a 3-token budget, roughly half of blocked spam evades via homoglyphs.
This is a property of the *classifier*, not the cryptography: the ZK gate
faithfully enforces whatever the public model decides, so classifier weakness
translates directly into evasion. Normalisation defences (Unicode confusable
folding, whitespace stripping before hashing) are obvious future work.

## 7. Limitations & future work

* **Model governance (N3).** The system verifiably enforces *a* model; it does
  not legitimise *which* model, nor constrain the model owner. Whoever controls
  `(θ, b, τ)` controls what is censorable. This is a governance problem, not a
  cryptographic one, and we do not solve it.
* **Client substitution / in-circuit feature binding (N1, N2).** `φ(m)` is
  computed outside the circuit; the circuit proves `C_θ(f)=1` for a witnessed
  `f` and `Poseidon(m,r)=h`, but not `f = φ(m)`. A modified client can therefore
  submit a benign `f` for a different `m`. The receiver's binding check catches
  `m`-vs-`h` substitution but cannot compel honest feature evaluation. Hashing
  the tokens in-circuit to bind `f` to `m` is the main cryptographic extension.
* **Classifier robustness.** Section 6 quantifies substantial evasion via
  character-level perturbations; input normalisation is needed for a deployable
  filter.
* **Scoped cuts (future work).** Small MLP classifiers with quantised ReLUs;
  Halo2/KZG and Groth16 comparisons; mobile / Termux / Raspberry-Pi
  benchmarking; paraphrase-based evasion; feature-dimension sweeps beyond 1024.

## 8. Artifacts

Open-source repository layout:

```
crypto-core/        X3DH + Double Ratchet + moderation envelope/AD helpers
moderation-core/    Plonky2 circuit, feature hashing (Rust), model loader, bench
moderation/         Python: train.py, features.py, adversarial.py, plot_benchmarks.py
server/             ZK-middlebox relay (verifies proofs before relaying)
client/             CLI with /send, /forge, /sendbad demo commands
phase0_feasibility.md, phase2_protocol_design.md, phase7_paper_draft.md
```
