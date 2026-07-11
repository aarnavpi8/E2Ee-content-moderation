"""
Phase 1 (Linear variant) - Classifier & Data Preparation.

Trains a *linear* content-moderation classifier on the SMS Spam Collection
using the portable feature-hashing scheme in features.py, sweeps the hashed
feature dimension d in {64, 256, 1024}, quantises the learned weights to
integers, and exports everything the Rust ZK circuit needs.
"""

import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
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
SCALE = 1 << 16          # fixed-point scale for weight quantisation (2^16)
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


def main():
    os.makedirs(OUT, exist_ok=True)
    texts, y = load_dataset()
    print(f"Loaded {len(texts)} messages "
          f"({int((y == 1).sum())} ham / {int((y == 0).sum())} spam)")
    print(f"Linear Classifier: scales SCALE={SCALE}\n")

    X_txt_tr, X_txt_te, y_tr, y_te = train_test_split(
        texts, y, test_size=0.2, random_state=SEED, stratify=y
    )

    combined = {"arch": "linear-logistic", "scale": SCALE, "seed": SEED, "dims": {}}
    per_dim_models = {}

    for d in DIMS:
        Xtr = featurize(X_txt_tr, d)
        Xte = featurize(X_txt_te, d)

        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, y_tr)

        theta = clf.coef_[0]          # shape (d,)
        bias = float(clf.intercept_[0])

        # ---- Float baseline (decision boundary score >= 0 -> predict ham) ----
        score_te = Xte @ theta + bias
        y_pred_float = (score_te >= 0.0).astype(np.int64)
        m_float = evaluate(y_te, y_pred_float)

        # ---- Integer-quantised model (what the ZK circuit actually enforces) -
        theta_q = np.round(theta * SCALE).astype(np.int64)
        bias_q = int(round(bias * SCALE))
        tau_q = 0   # threshold folded into the bias

        # Integer scores: features are already integers -> exact integer dot product
        Xte_int = Xte.astype(np.int64)
        score_q = Xte_int @ theta_q + bias_q
        y_pred_q = (score_q >= tau_q).astype(np.int64)
        m_quant = evaluate(y_te, y_pred_q)

        print(f"=== d = {d} ===")
        print(f"  float  : {m_float}")
        print(f"  quant  : {m_quant}")

        model = {
            "arch": "linear-logistic",
            "d": d,
            "scale": SCALE,
            "theta_q": theta_q.tolist(),
            "bias_q": bias_q,
            "tau_q": tau_q,
            "label_meaning": {"1": "ham/allowed", "0": "spam/blocked"},
            "fnv": {"offset_basis": FNV_OFFSET_BASIS, "prime": FNV_PRIME},
            "metrics_float": m_float,
            "metrics_quantized": m_quant,
        }
        per_dim_models[d] = (model, X_txt_te, y_te)
        with open(os.path.join(OUT, f"model_d{d}.json"), "w", encoding="utf-8") as fh:
            json.dump(model, fh, indent=2)
        combined["dims"][str(d)] = {
            "metrics_float": m_float, "metrics_quantized": m_quant
        }
        print()

    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(combined, fh, indent=2)

    # ---- Cross-check vectors for the Rust port (use the d=256 model) ---------
    model256, txt_te, _ = per_dim_models[256]
    theta_q = np.array(model256["theta_q"], dtype=np.int64)
    bias_q = model256["bias_q"]
    samples = ["Free entry now, WIN a prize!!!", "hey are we still on for lunch",
               "URGENT! call 09061701461 to claim", "ok see you tomorrow"]
    vectors = []
    for msg in samples:
        phi = feature_vector(msg, 256)
        score = int(np.array(phi, dtype=np.int64) @ theta_q + bias_q)
        vectors.append({
            "message": msg,
            "d": 256,
            "nonzero_features": {str(i): int(v) for i, v in enumerate(phi) if v != 0},
            "score_q": score,
            "prediction": "ham/allowed" if score >= 0 else "spam/blocked",
        })
    with open(os.path.join(OUT, "test_vectors.json"), "w", encoding="utf-8") as fh:
        json.dump({"model": "model_d256.json", "vectors": vectors}, fh, indent=2)

    print("Wrote linear models + metrics + test_vectors to", OUT)


if __name__ == "__main__":
    main()
