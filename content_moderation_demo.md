# Verifiable Content Moderation — Build & Demo Guide

This extends the base E2EE messenger with a zero-knowledge moderation gate. See
`phase0_feasibility.md`, `phase2_protocol_design.md`, and `phase7_paper_draft.md`
for the design and results.

## Layout

| Path | Role |
|---|---|
| `moderation/` | Python: model training, feature spec, adversarial eval, plotting |
| `moderation-core/` | Rust: Plonky2 circuit, feature-hash port, model loader, `bench` bin |
| `crypto-core/` | Adds `ModerationData` + `moderation_ad()` to the transport |
| `server/` | Verifies the proof before relaying (zero-knowledge middlebox) |
| `client/` | `/send`, `/forge`, `/sendbad` commands |

## Step 1 — Train the model (Python, one-time)

```bash
py -3.13 -m pip install scikit-learn numpy
py -3.13 moderation/train.py
```

Writes `moderation/models/model_d{64,256,1024}.json`, `metrics.json`, and
`test_vectors.json`. The server and client load `model_d256.json` by default.

## Step 2 — Build the Rust workspace

> **Nightly Rust required.** Plonky2 0.2.2 uses `#![feature(specialization)]`, so
> a nightly toolchain is needed. On this machine a project-scoped rustup override
> is already set to `nightly-x86_64-pc-windows-gnu` (the MSVC toolchain is
> unusable here — no Visual Studio C++ build tools), so a plain `cargo` picks the
> right toolchain automatically. Because it's the GNU host, a MinGW-w64 `gcc`
> must be on `PATH` for linking — this repo used MSYS2 `ucrt64`:
> ```bash
> # ensure gcc is reachable, e.g.:
> export PATH="/c/GCC/ucrt64/bin:$PATH"   # (PowerShell: $env:Path = 'C:\GCC\ucrt64\bin;' + $env:Path)
> ```
> On Linux/macOS just `rustup override set nightly` in the repo (default host works).

```bash
cargo build --release
cargo test -p crypto-core        # existing crypto tests (12 pass)
cargo test -p moderation-core    # circuit prove/verify + Rust<->Python parity (6 pass)
```

Verified on this machine: `crypto-core` 12/12, `moderation-core` 5 lib + 1 parity
tests pass; the full workspace builds; the benchmark runs (see Step 4).

The parity test (`moderation-core/tests/parity.rs`) confirms the Rust feature
hashing and classifier reproduce the Python `test_vectors.json` exactly.

## Step 3 — Three-terminal demo

> For a clean run, delete any stale session state from before moderation was
> added: `rm -f alice_state.json bob_state.json` (they will be regenerated).


**Terminal 1 — server (ZK middlebox)**
```bash
cargo run -p server --release
# Loading moderation model ... Circuit ready (d = 256).
# Server running on http://127.0.0.1:3000
```

**Terminal 2 — Bob (recipient)**
```bash
cargo run -p client --release -- bob
```

**Terminal 3 — Alice (sender)**
```bash
cargo run -p client --release -- alice
```

### Scenario A — honest message (passes end-to-end)
In Alice's terminal:
```
/send bob hey are we still on for lunch
```
* Server logs `[moderation] ACCEPTED ... proof verified`.
* Bob prints the message and `✓ [binding check] Poseidon(m, r) == h`.

### Scenario B — server drops an invalid proof
```
/sendbad bob hey are we still on for lunch
```
`/sendbad` corrupts the envelope commitment `h` after proving, so the proof's
committed hash no longer matches. The server logs
`[moderation] DROPPED ... proof invalid or missing` and Alice sees
`Server dropped it as expected`. Bob never receives it.

### Scenario C — forged content caught by the recipient
```
/forge bob hey lunch today | malicious different payload
```
Alice proves the benign left half but encrypts the right half. The server's
proof check passes (the proof is valid for the committed `h`), so it relays.
Bob decrypts the payload, recomputes `Poseidon(m, r)`, finds it ≠ `h`, and
prints `[BINDING CHECK FAILED] Sender proved one message but encrypted another!`
— an attributable detection.

### Scenario D — locally blocked content (classifier gate)
```
/send bob FREE entry! WIN a prize now, txt WIN to 80086 to claim your cash
```
The local classifier marks this spam, so `prove()` refuses and nothing is sent:
`Not sent: blocked by local classifier / proof failed`.

## Step 4 — Benchmarks (Phase 5)

```bash
cargo run -p moderation-core --release --bin bench
py -3.13 moderation/plot_benchmarks.py   # optional: needs matplotlib
```

## Step 5 — Adversarial evaluation (Phase 6)

```bash
py -3.13 moderation/adversarial.py
```
Writes `moderation/models/adversarial_results.json`.
