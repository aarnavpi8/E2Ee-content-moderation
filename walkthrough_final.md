# Walkthrough - Side-by-Side Linear vs. MLP Comparison

We have successfully integrated the linear classifier back into the codebase, organized the project systematically in folders for both Python and Rust, and generated comparative datasets, tables, and charts to support your research paper.

---

## 1. Project Organization

We reorganized the workspace to separate the linear and MLP configurations completely:

### Python Pipeline (`moderation/`)
- [features.py](file:///home/agnibh/Desktop/vscode/E2Ee-content-moderation/moderation/features.py): Common text normalization and FNV-1a hashing.
- [linear/](file:///home/agnibh/Desktop/vscode/E2Ee-content-moderation/moderation/linear/): Linear model folder containing:
  - [train.py](file:///home/agnibh/Desktop/vscode/E2Ee-content-moderation/moderation/linear/train.py): Trains a Logistic Regression classifier, quantizes it, and saves outputs.
  - [adversarial.py](file:///home/agnibh/Desktop/vscode/E2Ee-content-moderation/moderation/linear/adversarial.py): Evaluates the linear model under adversarial perturbations.
- [mlp/](file:///home/agnibh/Desktop/vscode/E2Ee-content-moderation/moderation/mlp/): MLP model folder containing:
  - [train.py](file:///home/agnibh/Desktop/vscode/E2Ee-content-moderation/moderation/mlp/train.py): Trains the 1-hidden-layer MLP classifier, quantizes it, and saves outputs.
  - [adversarial.py](file:///home/agnibh/Desktop/vscode/E2Ee-content-moderation/moderation/mlp/adversarial.py): Evaluates the MLP model under adversarial perturbations.
- [plot_benchmarks.py](file:///home/agnibh/Desktop/vscode/E2Ee-content-moderation/moderation/plot_benchmarks.py): Parses benchmarks from both models, printing comparative markdown tables and generating comparative plots.

### Rust ZK library (`moderation-core/`)
- [src/lib.rs](file:///home/agnibh/Desktop/vscode/E2Ee-content-moderation/moderation-core/src/lib.rs): Contains shared cryptographic primitives (Poseidon hash, packing preimages, and the `ProofBundle` representation).
- [src/linear.rs](file:///home/agnibh/Desktop/vscode/E2Ee-content-moderation/moderation-core/src/linear.rs): Linear model loader and Plonky2 ZK circuit constraint builder.
- [src/mlp.rs](file:///home/agnibh/Desktop/vscode/E2Ee-content-moderation/moderation-core/src/mlp.rs): MLP model loader and Plonky2 ZK circuit constraint builder.
- [src/bin/bench.rs](file:///home/agnibh/Desktop/vscode/E2Ee-content-moderation/moderation-core/src/bin/bench.rs): Unified benchmark runner evaluating both models across all dimensions.
- [tests/parity_linear.rs](file:///home/agnibh/Desktop/vscode/E2Ee-content-moderation/moderation-core/tests/parity_linear.rs) & [tests/parity_mlp.rs](file:///home/agnibh/Desktop/vscode/E2Ee-content-moderation/moderation-core/tests/parity_mlp.rs): Cross-language parity tests verifying bit-for-bit equivalence between Rust and Python.

---

## 2. Accuracy & Adversarial Robustness

Below are the accuracy and adversarial robustness metrics after training on the SMS Spam dataset (adversarial evaluation conducted at $d = 256$):

### Classifier Performance
- **Linear ($d=256$)**: Accuracy: **96.95%**, Spam F1: **88.36%**
- **MLP ($d=256$)**: Accuracy: **96.23%**, Spam F1: **86.00%**
- **Linear ($d=1024$)**: Accuracy: **97.49%**, Spam F1: **90.00%**
- **MLP ($d=1024$)**: Accuracy: **97.58%**, Spam F1: **90.39%**

### Adversarial Evasion Rates (budget=3)
Thanks to the visual normalization logic, lookalike evasion attacks are fully neutralized for both models:

| Evasion Class | Linear Evasion Rate | MLP Evasion Rate |
|---|---|---|
| **Homoglyph** | **0.0%** (was 51.1%) | **0.0%** (was 51.1%) |
| **Zero-width / Space** | **0.0%** (was 41.7%) | **0.0%** (was 41.7%) |
| **Synonyms** | **7.9%** | **6.9%** |

---

## 3. ZK Performance Comparison

The following comparative table summarizes the Plonky2 ZK circuit execution overhead (run on native hardware in release mode, comparing Linear vs. MLP):

| d | Model | Build (ms) | Prove Med/P95 (ms) | Proof Size (KB) | Verify Med/P95 (ms) | Added Lat Med/P95 (ms) | Speedup vs 3s |
|---|---|---|---|---|---|---|---|
| 64 | **Linear** | 20.4 | 46.8 / 68.5 | 91.9 KB | 7.79 / 10.54 | 54.6 / 76.4 | 64.1x |
| 64 | **MLP** | 93.9 | 158.0 / 222.0 | 113.3 KB | 10.13 / 13.62 | 168.7 / 236.1 | 19.0x |
|---|---|---|---|---|---|---|---|
| 256 | **Linear** | 55.4 | 92.1 / 183.1 | 101.0 KB | 8.77 / 13.02 | 101.0 / 191.7 | 32.6x |
| 256 | **MLP** | 203.1 | 279.7 / 335.7 | 118.6 KB | 10.72 / 12.35 | 290.7 / 346.4 | 10.7x |
|---|---|---|---|---|---|---|---|
| 1024 | **Linear** | 188.5 | 278.0 / 356.4 | 118.6 KB | 10.59 / 11.16 | 288.5 / 366.9 | 10.8x |
| 1024 | **MLP** | 420.2 | 537.8 / 621.7 | 124.0 KB | 10.66 / 11.44 | 548.7 / 632.3 | 5.6x |

### Proving Time Comparison Graph
![Proving Time Comparison](/home/agnibh/.gemini/antigravity/brain/9486a91f-4e36-4236-982e-0b2d247a2291/comparative_proving_time.png)

### End-to-End Latency Comparison Graph
![Added Latency Comparison](/home/agnibh/.gemini/antigravity/brain/9486a91f-4e36-4236-982e-0b2d247a2291/comparative_added_latency.png)
