# Verifiable Client-Side Content Moderation over an E2EE Messenger

*Paper draft (Phase 7). Assembles the design (Phases 0, 2), the classifier
baseline (Phase 1), the ZK circuit (Phase 3), the CLI integration (Phase 4),
the benchmarks (Phase 5), and the adversarial evaluation (Phase 6).*

## Abstract

We add a *verifiable* content-moderation gate to a from-scratch Signal-style
end-to-end encrypted messenger (X3DH + Double Ratchet, in Rust). A sender proves
in zero knowledge — using Plonky2 — that the plaintext of a message passes a
public MLP classifier (one hidden layer with quantised ReLU), and commits to the
content with a Poseidon hash. The
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

A small **multi-layer perceptron** — one hidden layer of width `H = 16` with
ReLU activation, single output logit — over the portable signed feature-hash
`φ` (`moderation/features.py`), trained (`sklearn.MLPClassifier`) on the SMS Spam
Collection (5,574 messages; 4,827 ham / 747 spam), 80/20 stratified split.
Weights are post-training quantised to integers at scale `2^12` per layer.
Because ReLU is scale-equivariant (`relu(s·x) = s·relu(x)`), quantisation
preserves the sign of the output margin, so the **integer MLP matches the float
MLP essentially exactly** (identical at d = 64, 256; marginally better at
d = 1024). Label 1 = ham ("allowed"); spam is the moderation-relevant class.

| d | accuracy | spam precision | spam recall | spam F1 |
|---|---|---|---|---|
| 64 | 0.9354 | 0.7730 | 0.7315 | 0.7517 |
| 256 | 0.9623 | 0.8543 | 0.8658 | 0.8600 |
| 1024 | 0.9767 | 0.9695 | 0.8523 | 0.9071 |

Relative to a plain linear classifier on the same features, the MLP notably
improves spam recall at low dimension (e.g. d = 64 recall 0.73 vs 0.58 linear).
`d = 256` is the deployment default. The integer model exported to JSON is
exactly the predicate the ZK circuit enforces; a Rust↔Python parity test
(`moderation-core/tests/parity.rs`) checks feature vectors, MLP logits, and
predictions agree bit-for-bit.

## 4. Circuit (Phase 3)

The Plonky2 circuit (Goldilocks field) enforces, with the public model baked in
as constants:

1. **Feature range checks** — each `f_i` is proven in `|f_i| ≤ 2^20`, preventing
   a malicious prover from overflowing the field to fake a passing score.
2. **MLP layer 1 + ReLU** — for each hidden neuron `i`, the pre-activation
   `a_i = Σ_j W1_q[i][j]·f_j + b1_q[i]` is computed, then ReLU is enforced with a
   decomposition gadget: witnesses `pos_i, neg_i` are range-checked to `[0, 2^48)`
   with `a_i = pos_i − neg_i` and `pos_i·neg_i = 0`, so `pos_i = relu(a_i)`. This
   is the extra machinery an MLP needs over a linear model (16 such gadgets).
3. **MLP layer 2 + threshold** — `out = Σ_i W2_q[i]·relu(a_i) + b2_q`; the margin
   `out − τ_q` is range-checked to `[0, 2^58)`, which holds iff `out ≥ τ_q` (a
   negative margin aliases to ≈`2^64` and fails the check).
4. **Commitment** — the message is packed 7 bytes per field element (fixed-size,
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
| 64 | 68 | 59.4 / 71.4 | 116,040 | 3.36 / 3.77 | 62.7 / 74.8 | ~50× |
| 256 | 113 | 279.9 / 596.3 | 121,480 | 9.48 / 14.48 | 290.1 / 603.6 | ~11× |
| 1024 | 498 | 550.6 / 585.4 | 126,984 | 10.45 / 10.98 | 560.0 / 596.1 | ~5× |

The `d = 256` deployment default proves in **~280 ms median** — still roughly
**11× faster** than the ~3 s proof-generation baseline reported in the
ZK-middlebox literature — with ~9 ms verification and ~120 KB proofs. The MLP is
markedly more expensive to prove than a linear model (~280 ms vs ~42 ms at
d = 256) because it adds `H·d` extra multiply-adds for the hidden layer plus 16
ReLU decomposition gadgets (each two 48-bit range checks and a product
constraint). Even so, end-to-end added latency stays under ~0.6 s through
d = 1024, comfortably inside the ~3 s budget — empirically confirming the Phase 0
GO decision for the MLP as well.

## 6. Adversarial evaluation (Phase 6)

Evasion rate = fraction of the 724 spam messages that the `d = 256` MLP correctly
blocks which a budget-`k` perturbation (k = max tokens edited) flips to
"allowed". Because the MLP is nonlinear, tokens are ranked by **occlusion** (the
score increase from removing each token) rather than a per-token weight.

| perturbation | k=1 | k=2 | k=3 |
|---|---|---|---|
| synonym substitution | 1.9% | 6.2% | 7.9% |
| homoglyph insertion | 15.1% | 34.8% | 51.1% |
| whitespace / zero-width insertion | 12.6% | 28.7% | 41.7% |

Findings: character-level attacks (homoglyphs, zero-width splits) are far more
effective than synonym substitution, and — crucially — **the MLP is no more
robust than the linear model was** (homoglyphs still evade ~51% at k = 3). The
attack exploits the *feature hashing / tokenisation*, moving a high-signal token
to a different hash bucket with a single edit, so a more expressive classifier on
top of the same features does not help. This is a property of the input
representation, not the cryptography: the ZK gate faithfully enforces whatever
the public model decides. Normalisation defences (Unicode confusable folding,
whitespace stripping before hashing) are the obvious next step.

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
* **Model expressiveness.** This build uses a one-hidden-layer MLP (H = 16,
  ReLU) — the plan's original MLP cut is **reversed** here. Deeper / wider
  networks are straightforward to add (more layers = more dot-products + ReLU
  gadgets) at higher proving cost; the ReLU decomposition gadget already
  generalises. In-circuit rescaling (to bound scale growth across many layers)
  would be the next piece for deeper nets.
* **Scoped cuts (future work).** Halo2/KZG and Groth16 comparisons; mobile /
  Termux / Raspberry-Pi benchmarking; paraphrase-based evasion; feature-dimension
  sweeps beyond 1024.

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
