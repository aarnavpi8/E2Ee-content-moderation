# Phase 0 — Feasibility Checkpoint

*Verifiable Client-Side Content Moderation — computational estimate and go/no-go decision.*

## 1. What the circuit must prove

For a message `m` with blinding nonce `r`, public model `(θ, b, τ)` and public
commitment `h`, the sender proves in zero knowledge:

```
(1)  Σ_i θ_q[i] · φ(m)[i] + b_q  ≥  τ_q        (classifier says "allowed")
(2)  Poseidon(m, r)  =  h                        (commitment binds the content)
```

where `φ(m)` is the length-`d` signed feature-hash vector, and `θ_q, b_q, τ_q`
are the integer-quantised weights exported in Phase 1.

## 2. Constraint estimate (d = 256, Plonky2 over Goldilocks)

Plonky2 works over the 64-bit Goldilocks field `p = 2^64 − 2^32 + 1` and counts
cost in **gates/rows** rather than R1CS constraints. Estimates below are for the
`d = 256` configuration plus a single Poseidon permutation.

| Component | Work | Approx. gates | Notes |
|---|---|---|---|
| Dot product `θ·φ(m)` | 256 mul-add | ~130–260 | `ArithmeticGate` packs multiple mul-adds per row; features are small ints so no overflow in Goldilocks. |
| Bias add + threshold | 1 add, 1 compare | ~40–70 | Comparison = one 64-bit range check on `(score − τ + offset)`. |
| Poseidon(m, r) | 1–2 permutations | ~15–30 | Poseidon is **native** in Plonky2 (`PoseidonGate`); each permutation is a handful of rows. Message chunked into field elements first. |
| Public-input / wiring overhead | — | ~50–100 | Copy constraints, public-input hashing. |
| **Total** | | **~250–500 gates** | Padded up to the next power of two (≈ 2^9–2^10 rows). |

### Cross-check against documented Plonky2 proving times

Public Plonky2 benchmarks on a desktop-class CPU report proving times on the
order of **tens of milliseconds up to a few hundred ms** for circuits in the
2^9–2^12 gate range (Poseidon-heavy recursion circuits, which are far larger,
prove in ~100–200 ms). Our moderation circuit is *smaller* than a single
recursion step, so a single-message proof should land comfortably **well under
1 second** on desktop — inside the ~3 s budget cited in the zero-knowledge
middlebox literature (Phase 5 will measure this empirically).

The `d = 1024` configuration scales the dot product by 4× (~1k–2k gates), still
below one recursion step; expected to remain sub-second.

## 3. Binding-target decision: `Poseidon(m, r)` vs `Poseidon(φ(m), r)`

**Decision: commit to the message, `h = Poseidon(m, r)`.**

Rationale:

* The receiver-side binding check (Phase 4) recomputes `Poseidon(m, r)` from the
  *decrypted plaintext* and compares against `h` in the AEAD associated data.
  This only works if `h` commits to `m` itself — the receiver never handles a
  raw `φ(m)`. Committing to `φ(m)` would leave the receiver unable to detect a
  sender who proved on `φ(m')` but encrypted `m`.
* `φ` is a public, deterministic function, so committing to `m` loses nothing:
  anyone (including the circuit) can recompute `φ(m)` from `m`.

### Known gap (documented, not solved here)

Feature extraction `φ(m)` is performed **outside** the circuit as a plain
computation, and the circuit takes `f = φ(m)` as a private witness. The circuit
therefore does *not* prove `f = φ(m)` — only that *some* `f` passes the
classifier and that `Poseidon(m, r) = h`. A malicious sender could in principle
feed a benign `f` while committing to a different `m`. This is mitigated, not
eliminated, by the receiver check (which catches `m` mismatches against `h`) and
is called out explicitly as a non-goal in the Phase 2 design document. Binding
`f` to `m` in-circuit would require re-deriving the feature hash inside the
circuit (hashing every token with an in-circuit hash), which is deferred as
future work.

## 4. Go / No-Go

**Decision: GO** with native Plonky2 circuit authoring.

* Constraint counts (~hundreds of gates) are ~orders of magnitude below what
  Plonky2 handles at sub-second proving time on desktop.
* Poseidon is native to Plonky2, removing the biggest cost risk (a non-native
  hash like SHA-256 would have dominated the circuit).
* The Phase 1 quantised classifier is **lossless** vs the float model
  (identical accuracy at d = 256: 96.95%), so the integer circuit enforces
  exactly the trained decision boundary — no accuracy is sacrificed to make the
  model circuit-friendly.

**Fallback checkpoint (from the plan):** if native Plonky2 circuit authoring
stalls by week 4, pivot to compiling the model with EZKL. Given Poseidon is
native and the arithmetic is a plain integer dot product, the fallback is not
expected to be triggered.
