"""
Phase 1 (MLP variant) - Classifier & Data Preparation.

Trains a small *multi-layer perceptron* (one hidden layer, ReLU) content-
moderation classifier on the SMS Spam Collection using the portable feature-
hashing scheme in features.py, sweeps the hashed feature dimension
d in {64, 256, 1024}, quantises the learned weights to integers, and exports
everything the Rust ZK circuit needs.
"""

import json
import os
import sys

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(HERE))
from features import feature_vector, FNV_OFFSET_BASIS, FNV_PRIME

DATA = os.path.join(os.path.dirname(HERE), "data", "sms.tsv")
OUT = os.path.join(HERE, "models")
DIMS = [64, 256, 1024]
HIDDEN_DIM = 16          # small MLP hidden layer
S1 = 1 << 12             # layer-1 weight quantisation scale (2^12)
S2 = 1 << 12             # layer-2 weight quantisation scale (2^12)
SEED = 42


def load_dataset():
    labels, texts = [], []
    with open(DATA, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            tag, _, text = line.partition("\t")
            if tag not in ("ham", "spam"):
                continue
            labels.append(1 if tag == "ham" else 0)   # 1 = ham/allowed
            texts.append(text)
    return texts, np.array(labels, dtype=np.int64)


def featurize(texts, d):
    return np.array([feature_vector(t, d) for t in texts], dtype=np.float64)


def evaluate(y_true, y_pred):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "spam_precision": round(float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),
        "spam_recall": round(float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),
        "spam_f1": round(float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),
        "confusion_matrix_[[tn,fp],[fn,tp]]": confusion_matrix(y_true, y_pred).tolist(),
    }


def quantize_mlp(clf):
    """Return integer MLP params from a trained 1-hidden-layer MLPClassifier."""
    W1f = clf.coefs_[0]        # (d, H)
    b1f = clf.intercepts_[0]   # (H,)
    W2f = clf.coefs_[1][:, 0]  # (H,)
    b2f = float(clf.intercepts_[1][0])

    w1_q = np.round(W1f.T * S1).astype(np.int64)      # (H, d)  w1_q[i][j]
    b1_q = np.round(b1f * S1).astype(np.int64)        # (H,)
    w2_q = np.round(W2f * S2).astype(np.int64)        # (H,)
    b2_q = int(round(b2f * S1 * S2))
    return w1_q, b1_q, w2_q, b2_q


def int_forward(F_int, w1_q, b1_q, w2_q, b2_q):
    """Exact integer MLP forward. F_int: (N, d) int64. Returns out logits (N,)."""
    h1_pre = F_int @ w1_q.T + b1_q          # (N, H), scale S1
    h1 = np.maximum(h1_pre, 0)              # integer ReLU
    out = h1 @ w2_q + b2_q                  # (N,), scale S1*S2
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    texts, y = load_dataset()
    print(f"Loaded {len(texts)} messages "
          f"({int((y == 1).sum())} ham / {int((y == 0).sum())} spam)")
    print(f"MLP: 1 hidden layer, H={HIDDEN_DIM}, ReLU; scales S1=S2={S1}\n")

    X_txt_tr, X_txt_te, y_tr, y_te = train_test_split(
        texts, y, test_size=0.2, random_state=SEED, stratify=y
    )

    combined = {"arch": "mlp-1hidden-relu", "hidden_dim": HIDDEN_DIM,
                "s1": S1, "s2": S2, "seed": SEED, "dims": {}}
    per_dim = {}

    for d in DIMS:
        Xtr = featurize(X_txt_tr, d)
        Xte = featurize(X_txt_te, d)

        clf = MLPClassifier(
            hidden_layer_sizes=(HIDDEN_DIM,), activation="relu",
            solver="adam", alpha=1e-3, max_iter=500, random_state=SEED,
        )
        clf.fit(Xtr, y_tr)

        # ---- Float baseline (decision = output logit >= 0 -> ham) -----------
        y_pred_float = clf.predict(Xte)
        m_float = evaluate(y_te, y_pred_float)

        # ---- Integer-quantised MLP (what the ZK circuit enforces) -----------
        w1_q, b1_q, w2_q, b2_q = quantize_mlp(clf)
        tau_q = 0
        out_te = int_forward(Xte.astype(np.int64), w1_q, b1_q, w2_q, b2_q)
        y_pred_int = (out_te >= tau_q).astype(np.int64)
        m_int = evaluate(y_te, y_pred_int)

        print(f"=== d = {d} ===")
        print(f"  float MLP : {m_float}")
        print(f"  int   MLP : {m_int}")

        model = {
            "arch": "mlp-1hidden-relu",
            "d": d,
            "hidden_dim": HIDDEN_DIM,
            "s1": S1, "s2": S2,
            "w1_q": w1_q.tolist(),
            "b1_q": b1_q.tolist(),
            "w2_q": w2_q.tolist(),
            "b2_q": b2_q,
            "tau_q": tau_q,
            "label_meaning": {"1": "ham/allowed", "0": "spam/blocked"},
            "fnv": {"offset_basis": FNV_OFFSET_BASIS, "prime": FNV_PRIME},
            "metrics_float": m_float,
            "metrics_quantized": m_int,
        }
        per_dim[d] = (model, w1_q, b1_q, w2_q, b2_q)
        with open(os.path.join(OUT, f"model_d{d}.json"), "w", encoding="utf-8") as fh:
            json.dump(model, fh, indent=2)
        combined["dims"][str(d)] = {"metrics_float": m_float, "metrics_quantized": m_int}
        print()

    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(combined, fh, indent=2)

    # ---- Cross-check vectors for the Rust port (d=256 MLP) ------------------
    model256, w1_q, b1_q, w2_q, b2_q = per_dim[256]
    samples = ["Free entry now, WIN a prize!!!", "hey are we still on for lunch",
               "URGENT! call 09061701461 to claim", "ok see you tomorrow"]
    vectors = []
    for msg in samples:
        phi = np.array(feature_vector(msg, 256), dtype=np.int64)
        out = int(int_forward(phi.reshape(1, -1), w1_q, b1_q, w2_q, b2_q)[0])
        vectors.append({
            "message": msg,
            "d": 256,
            "nonzero_features": {str(i): int(v) for i, v in enumerate(phi) if v != 0},
            "score_q": out,
            "prediction": "ham/allowed" if out >= 0 else "spam/blocked",
        })
    with open(os.path.join(OUT, "test_vectors.json"), "w", encoding="utf-8") as fh:
        json.dump({"model": "model_d256.json", "vectors": vectors}, fh, indent=2)

    print("Wrote MLP models + metrics + test_vectors to", OUT)


if __name__ == "__main__":
    main()
