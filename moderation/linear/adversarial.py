"""
Phase 6 - Adversarial Evaluation (Linear variant).
"""

import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(HERE))
from features import feature_vector, fnv1a_64

DATA = os.path.join(os.path.dirname(HERE), "data", "sms.tsv")
OUT = os.path.join(HERE, "models")
MODEL_FILE = os.path.join(OUT, "model_d256.json")
BUDGETS = [1, 2, 3]

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


def load_model():
    with open(MODEL_FILE, encoding="utf-8") as fh:
        return json.load(fh)


def score(text, model):
    phi = feature_vector(text, model["d"])
    theta = np.array(model["theta_q"], dtype=np.int64)
    return int(np.array(phi, dtype=np.int64) @ theta + model["bias_q"])


def token_spans(text):
    """Return (token, start, end) for maximal ASCII-alphanumeric runs."""
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


def rank_spam_tokens(text, model):
    base = score(text, model)
    spans = token_spans(text)
    scored = []
    for tok, s, e in spans:
        without = text[:s] + " " + text[e:]
        delta = score(without, model) - base
        scored.append((-delta, tok, s, e))
    scored.sort(key=lambda t: t[0])
    return scored


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


def try_evade(text, model, kind, budget):
    ranked = rank_spam_tokens(text, model)
    chars = list(text)
    used = 0
    replacements = []
    for contrib, tok, s, e in ranked:
        if used >= budget:
            break
        rep = perturb_token(tok, kind)
        if rep is None:
            continue
        replacements.append((s, e, rep))
        used += 1
    if not replacements:
        return False, text
    for s, e, rep in sorted(replacements, key=lambda t: t[0], reverse=True):
        chars[s:e] = list(rep)
    new_text = "".join(chars)
    evaded = score(new_text, model) >= model["tau_q"]
    return evaded, new_text


def main():
    model = load_model()

    blocked_spam = []
    with open(DATA, encoding="utf-8") as fh:
        for line in fh:
            tag, _, text = line.rstrip("\n").partition("\t")
            if tag == "spam" and score(text, model) < model["tau_q"]:
                blocked_spam.append(text)

    print(f"Model d={model['d']} correctly blocks {len(blocked_spam)} spam messages.\n")

    results = {"d": model["d"], "num_blocked_spam": len(blocked_spam), "classes": {}}
    examples = {}
    for kind in ("synonym", "homoglyph", "whitespace"):
        results["classes"][kind] = {}
        for budget in BUDGETS:
            evaded = 0
            first_example = None
            for text in blocked_spam:
                ok, new_text = try_evade(text, model, kind, budget)
                if ok:
                    evaded += 1
                    if first_example is None:
                        first_example = {"original": text, "perturbed": new_text}
            rate = round(evaded / len(blocked_spam), 4) if blocked_spam else 0.0
            results["classes"][kind][f"budget_{budget}"] = {
                "evaded": evaded, "evasion_rate": rate,
            }
            print(f"  {kind:11s} budget={budget}: "
                  f"evaded {evaded:4d}/{len(blocked_spam)}  ({rate*100:5.1f}%)")
            if first_example and kind not in examples:
                examples[kind] = first_example
        print()

    results["examples"] = examples
    with open(os.path.join(OUT, "adversarial_results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print("Wrote adversarial_results.json")


if __name__ == "__main__":
    main()
