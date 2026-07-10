# Phase 2 — Protocol Design: Verifiable Client-Side Content Moderation

*This document replaces Sections 3 and 4 of the original proposal. It formalises
the cryptographic binding, the threat model over X3DH + Double Ratchet, and an
explicit list of security guarantees and non-goals.*

---

## 1. Overview

The system augments a Signal-style end-to-end encrypted messenger (X3DH key
agreement + Double Ratchet, implemented in `crypto-core`) with a **verifiable
moderation gate**. A message is delivered only if its sender produced a valid
zero-knowledge proof that the plaintext passes a *public* linear classifier —
without the relay server ever seeing the plaintext.

The design combines two mechanisms:

1. **Proactive ZK gating** — the server refuses to relay any ciphertext not
   accompanied by a valid proof that the committed content passes the classifier.
2. **Reactive cryptographic attribution** — the recipient re-derives the
   content commitment from the decrypted plaintext and detects any sender who
   proved one message but encrypted another.

## 2. Notation

| Symbol | Meaning |
|---|---|
| `m` | plaintext message |
| `r` | random blinding nonce (per message) |
| `φ(m)` | length-`d` signed feature-hash vector (see `moderation/features.py`) |
| `θ_q, b_q, τ_q` | public integer-quantised classifier weights, bias, threshold |
| `C_θ(f)` | MLP classifier predicate: `1` iff `W2_q · relu(W1_q·f + b1_q) + b2_q ≥ τ_q` (== "allowed"), one hidden layer of width H=16 |
| `h` | Poseidon commitment `Poseidon(m, r)` |
| `π` | Plonky2 proof |
| `AD` | AEAD Associated Data field of the Double Ratchet message |

## 3. Message lifecycle

### 3.1 Sender (local)

1. Compute `f = φ(m)` in the clear (outside the circuit).
2. Sample random `r`.
3. Generate proof `π` with private witnesses `(m, r, f)` and public inputs
   `(θ_q, b_q, τ_q, h)`, where the circuit checks:
   `C_θ(f) = 1  ∧  Poseidon(m, r) = h`.
4. Encrypt with the Double Ratchet, setting **`AD = (h, r)`**. The AEAD tag now
   authenticates `(ciphertext, h, r)`.
5. Transmit `(ciphertext, tag, h, r, π)` to the server.

### 3.2 Server (zero-knowledge middlebox)

1. Run the Plonky2 **verifier** on public inputs `(θ_q, b_q, τ_q, h)` and `π`.
2. **Valid →** forward `(ciphertext, tag, h, r)` to the recipient's inbox.
   **Invalid →** drop the message.
3. The server never decrypts, never sees `m`, and holds no secret keys.

### 3.3 Receiver (decrypt + binding check)

1. AEAD-decrypt using `AD = (h, r)`. Because the tag covers the AD, any
   in-transit tampering with `h` or `r` makes decryption fail closed.
2. Recompute `h' = Poseidon(m, r)` from the decrypted `m` (plain computation, no
   circuit).
3. **`h' = h` →** accept: this is exactly the content the sender's proof cleared.
   **`h' ≠ h` →** reject and log a cryptographic-mismatch event: the sender
   proved validity for different content than they encrypted.

## 4. Where the binding comes from

* **Server ↔ proof binding:** the proof's public input `h` is the commitment;
  the server gates purely on `verify(θ_q, b_q, τ_q, h, π)`. No valid proof, no
  delivery.
* **Transport binding:** `h` and `r` travel in the authenticated AD field, so
  the AEAD tag cryptographically ties them to the exact ciphertext. Flipping any
  bit of `h` or `r` aborts decryption — no separate integrity mechanism needed.
* **Content binding:** `h = Poseidon(m, r)` is a hiding, binding commitment.
  Given the decrypted `m` and transmitted `r`, the receiver deterministically
  recomputes `h` and compares.

## 5. Security guarantees

G1. **No unproven delivery.** A ciphertext reaches a recipient only if the
   server verified a proof that the committed content passes the public
   classifier.

G2. **Server plaintext-blindness.** Verification uses only public inputs and the
   proof. The server learns nothing about `m` beyond "a valid proof exists".
   E2EE confidentiality of the underlying messenger is preserved.

G3. **Transport integrity of the commitment.** Any modification of `h` or `r`
   between sender and receiver causes AEAD decryption to fail (fail-closed).

G4. **Attributable content substitution.** A malicious sender who proves on one
   message but encrypts a different `m` is *detected* by the recipient's
   `Poseidon(m, r) = h` check, and the event is attributable to that sender.

G5. **No trusted setup.** Plonky2 is transparent; updating the public model
   (new `θ_q, b_q, τ_q`) requires no per-circuit ceremony.

## 6. Threat model

**In scope.** A malicious/curious relay server (honest-but-curious to actively
malicious for *confidentiality*, but relied upon only to *drop* unproven
messages); a malicious sender attempting to (a) deliver disallowed content or
(b) prove one message and send another; a network attacker tampering with
`(h, r, π, ciphertext)` in transit.

**Assumed honest / out of scope for the crypto.** The X3DH + Double Ratchet
layer is assumed correct and is the confidentiality/authenticity root (forward
secrecy, break-in recovery inherited unchanged). The public model `(θ, b, τ)` is
assumed authentically distributed to clients and server.

## 7. Non-goals (explicitly NOT solved)

N1. **Client substitution.** The guarantees hold only if the sender runs honest
   client software that actually evaluates `φ`/proves faithfully. A user running
   a modified client that computes an arbitrary passing `f` (decoupled from `m`)
   defeats G1 at the *classification* level; G4's receiver check still catches
   `m`-vs-`h` mismatches, but cannot force a sender to evaluate the real `φ(m)`.
   (See also the in-circuit `f = φ(m)` gap below.)

N2. **In-circuit feature binding.** `φ(m)` is computed outside the circuit; the
   circuit proves `C_θ(f)=1` for a witnessed `f` and `Poseidon(m,r)=h`, but does
   **not** prove `f = φ(m)`. Re-deriving the feature hash inside the circuit is
   deferred to future work.

N3. **Model governance.** Whoever controls the public model controls what is
   censorable. This system enforces *a* model verifiably; it does not decide
   *which* model is legitimate, nor prevent abuse of that power.

N4. **Classifier robustness.** The MLP classifier can be evaded by adversarial
   perturbations (synonyms, homoglyphs, zero-width characters). Phase 6
   quantifies this (the MLP is no more robust than a linear model, since the
   attack targets the feature hashing); the protocol does not claim a robust
   classifier.

N5. **Metadata privacy.** Sender/recipient identities, timing, and message sizes
   are handled exactly as in the base messenger — not improved by this work.

N6. **Denial of service.** A sender can always withhold a valid proof; the
   server simply drops such messages. Availability of the relay itself is out of
   scope.

## 8. Parameters (finalised)

* Classifier: 1-hidden-layer MLP, hidden width `H = 16`, ReLU activation.
* Feature dimensions swept: `d ∈ {64, 256, 1024}` (default/deployment: `d = 256`).
* Weight quantisation scale: `2^12` per layer (integer MLP matches float MLP).
* Commitment: Poseidon over Goldilocks, `h = Poseidon(m, r)`.
* Proof system: Plonky2 (transparent, native Rust, native Poseidon).
