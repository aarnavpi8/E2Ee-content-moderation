"""
Phase 6 - Adversarial Evaluation.

Measures the *evasion rate* of the trained linear classifier under three
perturbation classes, each constrained by a fixed budget k (max number of
tokens perturbed):

  1. Synonym substitution   - swap spam-indicative words for benign synonyms.
  2. Homoglyph insertion    - replace ASCII letters with look-alike Unicode.
  3. Whitespace / zero-width insertion - split tokens with U+200B / spaces.

An "evasion" is a message the classifier originally BLOCKS (score < tau, i.e.
labelled spam) that a budget-k perturbation flips to ALLOWED (score >= tau)
while remaining human-legible.

Greedy strategy: perturb the tokens contributing the strongest spam signal
first (most negative theta contribution), up to the budget.

Output: moderation/models/adversarial_results.json
"""

import json
import os

from features import feature_vector, fnv1a_64

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "sms.tsv")
OUT = os.path.join(HERE, "models")
MODEL_FILE = os.path.join(OUT, "model_d256.json")
BUDGETS = [1, 2, 3]

# ---- Small hand-built synonym map (documented limitation, see write-up) -----
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
    acc = model["bias_q"]
    for i, v in enumerate(phi):
        if v:
            acc += model["theta_q"][i] * v
    return acc


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
    """Rank token spans by how strongly they signal spam (most negative first)."""
    d, theta = model["d"], model["theta_q"]
    scored = []
    for tok, s, e in token_spans(text):
        low = tok.lower()
        h_idx = fnv1a_64(low.encode())
        h_sign = fnv1a_64(b"\x01" + low.encode())
        idx = h_idx % d
        sign = 1 if (h_sign & 1) == 0 else -1
        contrib = sign * theta[idx]      # contribution to the "allowed" score
        scored.append((contrib, tok, s, e))
    scored.sort(key=lambda t: t[0])       # ascending: most spam-ish first
    return scored


def perturb_token(tok, kind):
    low = tok
    if kind == "synonym":
        return SYNONYMS.get(tok.lower())          # None if no synonym known
    if kind == "homoglyph":
        # Replace up to 2 letters with homoglyphs (enough to break the hash).
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
    """Apply greedy budget-k perturbation; return (evaded, perturbed_text)."""
    ranked = rank_spam_tokens(text, model)
    chars = list(text)
    used = 0
    # Rebuild from spans so indices stay valid: collect replacements first.
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
    # Apply right-to-left so earlier indices remain valid.
    for s, e, rep in sorted(replacements, key=lambda t: t[0], reverse=True):
        chars[s:e] = list(rep)
    new_text = "".join(chars)
    evaded = score(new_text, model) >= model["tau_q"]
    return evaded, new_text


def main():
    model = load_model()

    # Collect spam messages the model correctly BLOCKS.
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
