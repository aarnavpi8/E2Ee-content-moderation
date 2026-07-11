"""
Portable feature-hashing specification for the verifiable content-moderation
classifier.

This module is the SINGLE SOURCE OF TRUTH for how a plaintext message m is
turned into a fixed-dimension integer feature vector phi(m). The exact same
algorithm is re-implemented in Rust (moderation-core/src/features.rs) so that
the classifier evaluated in Python (Phase 1) and the one enforced inside the
Plonky2 circuit (Phase 3) agree bit-for-bit.

Design goals
------------
* Deterministic and language-portable: only integer arithmetic mod 2^64,
  ASCII tokenisation, no floats, no locale-dependent behaviour.
* Integer-valued output: the ZK circuit works over a prime field, so features
  must be (small) integers, not TF-IDF floats.
* Signed feature hashing (a la Weinberger et al. 2009): a second hash decides
  the sign of each token's contribution, which de-biases hash collisions.

Algorithm
---------
1. Lower-case the message and interpret it as UTF-8 bytes.
2. Tokenise into maximal runs of ASCII alphanumerics [a-z0-9]+. Every other
   byte (punctuation, whitespace, non-ASCII) is a separator.
3. For each token t:
       h_idx  = fnv1a_64(t)
       h_sign = fnv1a_64(b"\x01" + t)      # independent hash via 1-byte prefix
       idx    = h_idx  % d
       sign   = +1 if (h_sign & 1) == 0 else -1
       phi[idx] += sign
4. Return phi, a length-d vector of (possibly negative) integers.
"""

import unicodedata

FNV_OFFSET_BASIS = 14695981039346656037   # 0xcbf29ce484222325
FNV_PRIME = 1099511628211                 # 0x100000001b3
U64_MASK = (1 << 64) - 1


def fnv1a_64(data: bytes) -> int:
    """64-bit FNV-1a hash. Matches the Rust implementation exactly."""
    h = FNV_OFFSET_BASIS
    for byte in data:
        h ^= byte
        h = (h * FNV_PRIME) & U64_MASK
    return h


def normalize_text(text: str) -> str:
    """Normalize input string before tokenization and feature hashing to mitigate homoglyphs and zero-width characters."""
    decomposed = unicodedata.normalize('NFKD', text)
    homoglyphs = {
        'а': 'a', 'А': 'A',
        'с': 'c', 'С': 'C',
        'е': 'e', 'Е': 'E',
        'о': 'o', 'О': 'O',
        'р': 'p', 'Р': 'P',
        'х': 'x', 'Х': 'X',
        'у': 'y', 'У': 'Y',
        'і': 'i', 'І': 'I',
        'ј': 'j', 'Ј': 'J',
        'ѕ': 's', 'Ѕ': 'S',
    }
    
    out = []
    for ch in decomposed:
        o = ord(ch)
        # Strip zero-width / formatting/invisible characters
        if (0x200B <= o <= 0x200F) or (0x202A <= o <= 0x202E) or (0x2060 <= o <= 0x206F) or (o == 0xFEFF):
            continue
        # Map homoglyphs
        if ch in homoglyphs:
            out.append(homoglyphs[ch])
        else:
            out.append(ch)
            
    return "".join(out)


def tokenize(message: str):
    """Yield maximal ASCII-alphanumeric tokens (already lower-cased bytes)."""
    tokens = []
    cur = bytearray()
    for ch in message.lower():
        o = ord(ch)
        is_alnum = (48 <= o <= 57) or (97 <= o <= 122)  # 0-9 or a-z
        if is_alnum:
            cur.append(o)
        elif cur:
            tokens.append(bytes(cur))
            cur = bytearray()
    if cur:
        tokens.append(bytes(cur))
    return tokens


def feature_vector(message: str, d: int):
    """Compute the length-d integer feature vector phi(m)."""
    normalized = normalize_text(message)
    phi = [0] * d
    for tok in tokenize(normalized):
        h_idx = fnv1a_64(tok)
        h_sign = fnv1a_64(b"\x01" + tok)
        idx = h_idx % d
        sign = 1 if (h_sign & 1) == 0 else -1
        phi[idx] += sign
    return phi


if __name__ == "__main__":
    # Quick self-test / demo used to cross-check the Rust port.
    for msg in ["Free entry now!!", "hey are we still on for lunch", "Fr\u200bee\ufeff e\u200dntry", "асеорхуі", "𝖥𝗋𝖾𝖾 𝖾𝗇𝗍𝗋𝗒"]:
        print(f"Original: {repr(msg)} -> Normalized: {repr(normalize_text(msg))}")
        for d in (64, 256):
            phi = feature_vector(msg, d)
            nz = {i: v for i, v in enumerate(phi) if v != 0}
            print(f"d={d:4d} msg={msg!r:40s} nonzero={nz}")
