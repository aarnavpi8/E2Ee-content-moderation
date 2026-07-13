import os
import sys
import json
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns

# Setup paths
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
from features import feature_vector

DATA_SMS = os.path.join(HERE, "data", "sms.tsv")
DATA_ENRON_ZIP = os.path.join(HERE, "enron_spam_data.zip")
ARTIFACT_DIR = "/home/agnibh/.gemini/antigravity/brain/937cae6b-188c-4d28-ae44-9990291470a8"

DIMS = [64, 256, 1024]
SCALE = 1 << 16  # Linear scale (2^16)
S1 = 1 << 12     # MLP scale 1 (2^12)
S2 = 1 << 12     # MLP scale 2 (2^12)
HIDDEN_DIM = 16
SEED = 42

# Adversarial attack parameters
SYNONYMS = {
    "free": "complimentary", "win": "earn", "winner": "recipient",
    "prize": "reward", "cash": "funds", "urgent": "timely",
    "call": "phone", "claim": "request", "offer": "deal",
    "guaranteed": "assured", "credit": "balance", "txt": "message",
    "text": "message", "reply": "respond", "stop": "halt",
    "mobile": "cell", "customer": "client", "won": "received",
}

HOMOGLYPHS = {
    "a": "а", "c": "с", "e": "е", "o": "о",
    "p": "р", "x": "х", "y": "у", "i": "і",
}
ZERO_WIDTH = "​"

def load_sms_dataset():
    labels, texts = [], []
    with open(DATA_SMS, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            tag, _, text = line.partition("\t")
            if tag not in ("ham", "spam"):
                continue
            labels.append(1 if tag == "ham" else 0)   # 1 = ham/allowed, 0 = spam/blocked
            texts.append(text)
    return texts, np.array(labels, dtype=np.int64)

def load_enron_dataset():
    # Load and clean Enron dataset
    df = pd.read_csv(DATA_ENRON_ZIP, compression='zip')
    df = df.dropna(subset=['Message', 'Spam/Ham'])
    df = df[df['Spam/Ham'].isin(['ham', 'spam'])]
    df['label'] = df['Spam/Ham'].map({'ham': 1, 'spam': 0})
    df['Message'] = df['Message'].astype(str)
    
    # Stratified sample: 2500 ham, 2500 spam
    df_sampled = df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(min(len(x), 2500), random_state=SEED)
    )
    return df_sampled['Message'].tolist(), df_sampled['label'].to_numpy(dtype=np.int64)

def featurize(texts, d):
    return np.array([feature_vector(t, d) for t in texts], dtype=np.float64)

def compute_metrics(y_true, y_pred):
    # Spam (0) is positive, Ham (1) is negative
    tp = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 0) & (y_pred == 1))
    
    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # spam recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # ham false positive rate (blocked ham)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # spam precision
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,
        "f1": f1
    }

def quantize_linear(clf):
    theta = clf.coef_[0]
    bias = float(clf.intercept_[0])
    theta_q = np.round(theta * SCALE).astype(np.int64)
    bias_q = int(round(bias * SCALE))
    return theta_q, bias_q

def quantize_mlp(clf):
    W1f = clf.coefs_[0]        # (d, H)
    b1f = clf.intercepts_[0]   # (H,)
    W2f = clf.coefs_[1][:, 0]  # (H,)
    b2f = float(clf.intercepts_[1][0])

    w1_q = np.round(W1f.T * S1).astype(np.int64)      # (H, d)
    b1_q = np.round(b1f * S1).astype(np.int64)        # (H,)
    w2_q = np.round(W2f * S2).astype(np.int64)        # (H,)
    b2_q = int(round(b2f * S1 * S2))
    return w1_q, b1_q, w2_q, b2_q

def int_forward_mlp(F_int, w1_q, b1_q, w2_q, b2_q):
    h1_pre = F_int @ w1_q.T + b1_q          # (N, H)
    h1 = np.maximum(h1_pre, 0)              # integer ReLU
    out = h1 @ w2_q + b2_q                  # (N,)
    return out

def score_linear(text, theta_q, bias_q, d):
    phi = feature_vector(text, d)
    return int(np.array(phi, dtype=np.int64) @ theta_q + bias_q)

def score_mlp(text, w1_q, b1_q, w2_q, b2_q, d):
    phi = feature_vector(text, d)
    h1_pre = np.array(phi, dtype=np.int64) @ w1_q.T + b1_q
    h1 = np.maximum(h1_pre, 0)
    out = h1 @ w2_q + b2_q
    return int(out)

# Adversarial Helper Functions
def token_spans(text):
    spans = []
    start = None
    for i, ch in enumerate(text):
        o = ord(ch.lower())
        alnum = (48 <= o <= 57) or (97 <= o <= 122)
        if alnum and start is None:
            start = i
        elif not alnum and start is not None:
            spans.append((text[start:i], start, i))
            start = None
    if start is not None:
        spans.append((text[start:], start, len(text)))
    return spans

def perturb_token(tok, kind):
    low = tok
    if kind == "synonym":
        return SYNONYMS.get(tok.lower())
    if kind == "homoglyph":
        out, changed = [], 0
        for ch in low:
            if changed < 2 and ch.lower() in HOMOGLYPHS:
                out.append(HOMOGLYPHS[ch.lower()])
                changed += 1
            else:
                out.append(ch)
        return "".join(out) if changed else None
    if kind == "whitespace":
        if len(low) < 2:
            return None
        mid = len(low) // 2
        return low[:mid] + ZERO_WIDTH + low[mid:]
    raise ValueError(kind)

def try_evade(text, kind, budget, score_fn):
    # Determine contributions by simple occlusion
    base = score_fn(text)
    spans = token_spans(text)
    scored = []
    for tok, s, e in spans:
        without = text[:s] + " " + text[e:]
        delta = score_fn(without) - base
        scored.append((-delta, tok, s, e))
    scored.sort(key=lambda t: t[0])
    
    chars = list(text)
    used = 0
    replacements = []
    for contrib, tok, s, e in scored:
        if used >= budget:
            break
        rep = perturb_token(tok, kind)
        if rep is None:
            continue
        replacements.append((s, e, rep))
        used += 1
    if not replacements:
        return False
    for s, e, rep in sorted(replacements, key=lambda t: t[0], reverse=True):
        chars[s:e] = list(rep)
    new_text = "".join(chars)
    # Evaded if predicted as Ham/Allowed (score >= 0)
    return score_fn(new_text) >= 0

def bootstrap_ci(evaded_flags, B=1000, alpha=0.05):
    evaded_flags = np.array(evaded_flags)
    n = len(evaded_flags)
    if n == 0:
        return 0.0, 0.0, 0.0
    boot_means = []
    for _ in range(B):
        sample = np.random.choice(evaded_flags, size=n, replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.sort(boot_means)
    low = np.percentile(boot_means, 100 * (alpha / 2))
    high = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return np.mean(evaded_flags), low, high

def main():
    print("Loading SMS and Enron datasets...")
    sms_texts, sms_labels = load_sms_dataset()
    enron_texts, enron_labels = load_enron_dataset()
    
    print(f"SMS dataset size: {len(sms_texts)} ({sum(sms_labels == 1)} ham / {sum(sms_labels == 0)} spam)")
    print(f"Enron sampled dataset size: {len(enron_texts)} ({sum(enron_labels == 1)} ham / {sum(enron_labels == 0)} spam)")
    
    # ----------------------------------------------------
    # Task 1: 5-Fold Cross Validation & McNemar's Test
    # ----------------------------------------------------
    print("\nRunning 5-Fold Cross Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    cv_results = {}
    
    # Track out-of-fold predictions at d=256 for McNemar's test
    oof_y_true = []
    oof_linear_pred = []
    oof_mlp_pred = []
    
    for d in DIMS:
        print(f"Featurizing SMS dataset at d={d}...")
        sms_features = featurize(sms_texts, d)
        
        cv_results[d] = {
            "linear_float": [], "linear_quant": [],
            "mlp_float": [], "mlp_quant": []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(sms_features, sms_labels)):
            X_tr, y_tr = sms_features[train_idx], sms_labels[train_idx]
            X_val, y_val = sms_features[val_idx], sms_labels[val_idx]
            
            # Train Linear
            clf_linear = LogisticRegression(max_iter=2000, C=1.0)
            clf_linear.fit(X_tr, y_tr)
            theta_q, bias_q = quantize_linear(clf_linear)
            
            # Linear predictions
            linear_pred_float = clf_linear.predict(X_val)
            linear_pred_quant = (X_val.astype(np.int64) @ theta_q + bias_q >= 0).astype(np.int64)
            
            # Train MLP
            clf_mlp = MLPClassifier(
                hidden_layer_sizes=(HIDDEN_DIM,), activation="relu",
                solver="adam", alpha=1e-3, max_iter=500, random_state=SEED
            )
            clf_mlp.fit(X_tr, y_tr)
            w1_q, b1_q, w2_q, b2_q = quantize_mlp(clf_mlp)
            
            # MLP predictions
            mlp_pred_float = clf_mlp.predict(X_val)
            mlp_pred_quant = (int_forward_mlp(X_val.astype(np.int64), w1_q, b1_q, w2_q, b2_q) >= 0).astype(np.int64)
            
            # Track out-of-fold predictions for McNemar's test at d=256
            if d == 256:
                oof_y_true.extend(y_val)
                oof_linear_pred.extend(linear_pred_quant)
                oof_mlp_pred.extend(mlp_pred_quant)
                
            # Compute metrics
            cv_results[d]["linear_float"].append(compute_metrics(y_val, linear_pred_float))
            cv_results[d]["linear_quant"].append(compute_metrics(y_val, linear_pred_quant))
            cv_results[d]["mlp_float"].append(compute_metrics(y_val, mlp_pred_float))
            cv_results[d]["mlp_quant"].append(compute_metrics(y_val, mlp_pred_quant))
            
    # Compute Cross-Validation summary statistics
    print("\n### 5-Fold Cross Validation Results (SMS Dataset) ###")
    rows = []
    for d in DIMS:
        for model_key in ["linear_float", "linear_quant", "mlp_float", "mlp_quant"]:
            metrics_list = cv_results[d][model_key]
            summary = {}
            for metric in ["accuracy", "tpr", "fpr", "precision", "f1"]:
                vals = [m[metric] for m in metrics_list]
                summary[metric] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
            row = {
                "d": d,
                "model": model_key,
                "accuracy": summary["accuracy"],
                "tpr": summary["tpr"],
                "fpr": summary["fpr"],
                "precision": summary["precision"],
                "f1": summary["f1"]
            }
            rows.append(row)
            
    df_cv = pd.DataFrame(rows)
    print(df_cv.to_markdown(index=False))
    
    # Run McNemar's Test at d=256
    print("\nRunning McNemar's Test at d=256...")
    y_true = np.array(oof_y_true)
    pred_lin = np.array(oof_linear_pred)
    pred_mlp = np.array(oof_mlp_pred)
    
    correct_lin = (pred_lin == y_true)
    correct_mlp = (pred_mlp == y_true)
    
    # Contingency Table:
    #                 MLP Correct | MLP Incorrect
    # Linear Correct      a       |      b
    # Linear Incorrect    c       |      d
    a_cell = np.sum(correct_lin & correct_mlp)
    b_cell = np.sum(correct_lin & ~correct_mlp)
    c_cell = np.sum(~correct_lin & correct_mlp)
    d_cell = np.sum(~correct_lin & ~correct_mlp)
    
    stat = (abs(b_cell - c_cell) - 1)**2 / (b_cell + c_cell) if (b_cell + c_cell) > 0 else 0.0
    p_val = chi2.sf(stat, 1) if (b_cell + c_cell) > 0 else 1.0
    
    print("\n### McNemar's Test Contingency Table (d=256) ###")
    print(f"Linear Correct   & MLP Correct   : {a_cell}")
    print(f"Linear Correct   & MLP Incorrect : {b_cell} (b)")
    print(f"Linear Incorrect & MLP Correct   : {c_cell} (c)")
    print(f"Linear Incorrect & MLP Incorrect : {d_cell}")
    print(f"McNemar Chi-Squared Statistic: {stat:.4f}")
    print(f"P-Value: {p_val:.6e}")
    
    # ----------------------------------------------------
    # Task 2: Distribution Shift / Generalization (Enron)
    # ----------------------------------------------------
    print("\nEvaluating Out-of-Distribution Generalization on Enron Corpus...")
    enron_results = []
    
    for d in DIMS:
        print(f"Featurizing Enron dataset at d={d}...")
        enron_features = featurize(enron_texts, d)
        sms_features = featurize(sms_texts, d)
        
        # Train full models on full SMS dataset
        clf_linear = LogisticRegression(max_iter=2000, C=1.0)
        clf_linear.fit(sms_features, sms_labels)
        theta_q, bias_q = quantize_linear(clf_linear)
        
        clf_mlp = MLPClassifier(
            hidden_layer_sizes=(HIDDEN_DIM,), activation="relu",
            solver="adam", alpha=1e-3, max_iter=500, random_state=SEED
        )
        clf_mlp.fit(sms_features, sms_labels)
        w1_q, b1_q, w2_q, b2_q = quantize_mlp(clf_mlp)
        
        # Predict on Enron
        lin_pred_float = clf_linear.predict(enron_features)
        lin_pred_quant = (enron_features.astype(np.int64) @ theta_q + bias_q >= 0).astype(np.int64)
        
        mlp_pred_float = clf_mlp.predict(enron_features)
        mlp_pred_quant = (int_forward_mlp(enron_features.astype(np.int64), w1_q, b1_q, w2_q, b2_q) >= 0).astype(np.int64)
        
        for name, pred in [("linear_float", lin_pred_float), 
                           ("linear_quant", lin_pred_quant), 
                           ("mlp_float", mlp_pred_float), 
                           ("mlp_quant", mlp_pred_quant)]:
            metrics = compute_metrics(enron_labels, pred)
            enron_results.append({
                "d": d,
                "model": name,
                "accuracy": f"{metrics['accuracy']:.4f}",
                "tpr": f"{metrics['tpr']:.4f}",
                "fpr": f"{metrics['fpr']:.4f}",
                "precision": f"{metrics['precision']:.4f}",
                "f1": f"{metrics['f1']:.4f}"
            })
            
    df_enron = pd.DataFrame(enron_results)
    print("\n### Enron Generalization Results ###")
    print(df_enron.to_markdown(index=False))
    
    # ----------------------------------------------------
    # Task 3: Adversarial Evasion Bootstrap CI (d=256)
    # ----------------------------------------------------
    print("\nRunning Adversarial Evasion with Bootstrap CIs (d=256)...")
    d = 256
    sms_features_256 = featurize(sms_texts, d)
    
    # Train full models
    clf_linear = LogisticRegression(max_iter=2000, C=1.0)
    clf_linear.fit(sms_features_256, sms_labels)
    theta_q, bias_q = quantize_linear(clf_linear)
    
    clf_mlp = MLPClassifier(
        hidden_layer_sizes=(HIDDEN_DIM,), activation="relu",
        solver="adam", alpha=1e-3, max_iter=500, random_state=SEED
    )
    clf_mlp.fit(sms_features_256, sms_labels)
    w1_q, b1_q, w2_q, b2_q = quantize_mlp(clf_mlp)
    
    # Select correctly blocked spam
    blocked_spam_linear = []
    blocked_spam_mlp = []
    
    for text, label in zip(sms_texts, sms_labels):
        if label == 0:  # spam
            if score_linear(text, theta_q, bias_q, d) < 0:
                blocked_spam_linear.append(text)
            if score_mlp(text, w1_q, b1_q, w2_q, b2_q, d) < 0:
                blocked_spam_mlp.append(text)
                
    print(f"Blocked spam count: Linear={len(blocked_spam_linear)}, MLP={len(blocked_spam_mlp)}")
    
    evasion_results = []
    
    for model_name, blocked_list, score_fn in [
        ("Linear (Quantized)", blocked_spam_linear, lambda txt: score_linear(txt, theta_q, bias_q, d)),
        ("MLP (Quantized)", blocked_spam_mlp, lambda txt: score_mlp(txt, w1_q, b1_q, w2_q, b2_q, d))
    ]:
        for kind in ["synonym", "homoglyph", "whitespace"]:
            evaded_flags = []
            for text in blocked_list:
                evaded = try_evade(text, kind, 3, score_fn)
                evaded_flags.append(1 if evaded else 0)
                
            mean_val, low, high = bootstrap_ci(evaded_flags, B=1000)
            evasion_results.append({
                "model": model_name,
                "attack": kind,
                "evasion_rate": mean_val,
                "ci_low": low,
                "ci_high": high
            })
            
    df_evasion = pd.DataFrame(evasion_results)
    print("\n### Adversarial Evasion Rates (budget=3) with 95% Bootstrap CIs ###")
    print(df_evasion.to_markdown(index=False))
    
    # Save JSON files with results to artifact directory
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with open(os.path.join(ARTIFACT_DIR, "cv_results.json"), "w") as fh:
        df_cv.to_json(fh, orient="records", indent=2)
    with open(os.path.join(ARTIFACT_DIR, "enron_results.json"), "w") as fh:
        df_enron.to_json(fh, orient="records", indent=2)
    with open(os.path.join(ARTIFACT_DIR, "evasion_results.json"), "w") as fh:
        df_evasion.to_json(fh, orient="records", indent=2)
    
    # ----------------------------------------------------
    # Task 4: Plotting
    # ----------------------------------------------------
    print("\nGenerating plots...")
    sns.set_theme(style="whitegrid")
    
    # Plot 1: K-Fold CV Accuracy & F1 Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Filter for quantized models
    cv_quant = df_cv[df_cv['model'].isin(['linear_quant', 'mlp_quant'])].copy()
    cv_quant['accuracy_val'] = cv_quant['accuracy'].apply(lambda x: float(x.split(" ")[0]))
    cv_quant['accuracy_std'] = cv_quant['accuracy'].apply(lambda x: float(x.split(" ")[2]))
    cv_quant['f1_val'] = cv_quant['f1'].apply(lambda x: float(x.split(" ")[0]))
    cv_quant['f1_std'] = cv_quant['f1'].apply(lambda x: float(x.split(" ")[2]))
    
    # Map model names for plotting
    cv_quant['Model Class'] = cv_quant['model'].map({'linear_quant': 'Linear (Quantized)', 'mlp_quant': 'MLP (Quantized)'})
    
    sns.barplot(data=cv_quant, x='d', y='accuracy_val', hue='Model Class', ax=axes[0], palette='muted')
    axes[0].set_title("5-Fold Cross-Validation Accuracy (SMS Dataset)")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xlabel("Feature Dimension (d)")
    axes[0].set_ylim(0.9, 1.0)
    
    sns.barplot(data=cv_quant, x='d', y='f1_val', hue='Model Class', ax=axes[1], palette='muted')
    axes[1].set_title("5-Fold Cross-Validation Spam F1-Score (SMS Dataset)")
    axes[1].set_ylabel("F1-Score")
    axes[1].set_xlabel("Feature Dimension (d)")
    axes[1].set_ylim(0.7, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "cv_comparison.png"), dpi=300)
    plt.close()
    
    # Plot 2: Enron Generalization Accuracy & F1 Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    enron_quant = df_enron[df_enron['model'].isin(['linear_quant', 'mlp_quant'])].copy()
    enron_quant['accuracy_val'] = enron_quant['accuracy'].astype(float)
    enron_quant['f1_val'] = enron_quant['f1'].astype(float)
    enron_quant['Model Class'] = enron_quant['model'].map({'linear_quant': 'Linear (Quantized)', 'mlp_quant': 'MLP (Quantized)'})
    
    sns.barplot(data=enron_quant, x='d', y='accuracy_val', hue='Model Class', ax=axes[0], palette='muted')
    axes[0].set_title("OOD Generalization Accuracy on Enron Dataset")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xlabel("Feature Dimension (d)")
    axes[0].set_ylim(0.5, 1.0)
    
    sns.barplot(data=enron_quant, x='d', y='f1_val', hue='Model Class', ax=axes[1], palette='muted')
    axes[1].set_title("OOD Generalization Spam F1-Score on Enron Dataset")
    axes[1].set_ylabel("F1-Score")
    axes[1].set_xlabel("Feature Dimension (d)")
    axes[1].set_ylim(0.5, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "enron_generalization.png"), dpi=300)
    plt.close()
    
    # Plot 3: Evasion Rates with 95% Bootstrap CI Error Bars
    plt.figure(figsize=(10, 6))
    
    # Compute error lengths
    df_evasion['err_low'] = df_evasion['evasion_rate'] - df_evasion['ci_low']
    df_evasion['err_high'] = df_evasion['ci_high'] - df_evasion['evasion_rate']
    
    # Bar plot with custom error bars
    models = df_evasion['model'].unique()
    attacks = df_evasion['attack'].unique()
    
    x = np.arange(len(attacks))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, model in enumerate(models):
        sub_df = df_evasion[df_evasion['model'] == model]
        rates = sub_df['evasion_rate'].values
        yerr = np.vstack([sub_df['err_low'].values, sub_df['err_high'].values])
        
        ax.bar(x + (i - 0.5)*width, rates, width, label=model, yerr=yerr, capsize=5)
        
    ax.set_title("Adversarial Evasion Rates (budget=3) with 95% Bootstrap CIs")
    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in attacks])
    ax.set_ylabel("Evasion Rate")
    ax.set_xlabel("Attack Type")
    ax.set_ylim(0.0, 0.7)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "evasion_bootstrap_ci.png"), dpi=300)
    plt.close()
    
    print("Plots saved to:", ARTIFACT_DIR)
    print("Done!")

if __name__ == "__main__":
    main()
