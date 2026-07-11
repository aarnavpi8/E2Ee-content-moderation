//! Rust port of the portable feature-hashing scheme.
//!
//! This MUST stay bit-for-bit identical to `moderation/features.py`. The
//! cross-check test vectors in `moderation/models/test_vectors.json` are used
//! by the integration test to guarantee the two implementations agree.

pub const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325; // 14695981039346656037
pub const FNV_PRIME: u64 = 0x0000_0100_0000_01b3; // 1099511628211

/// 64-bit FNV-1a hash (wrapping, mod 2^64).
pub fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h = FNV_OFFSET_BASIS;
    for &byte in data {
        h ^= byte as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Tokenise into maximal ASCII-alphanumeric runs, ASCII-lower-cased.
///
/// Matches the Python reference for ASCII text: Python lower-cases the whole
/// (Unicode) string then keeps only `[a-z0-9]`; for ASCII input this is exactly
/// ASCII lower-casing followed by the same class filter. Non-ASCII bytes never
/// fall in `[a-z0-9]` and are dropped identically on both sides.
pub fn tokenize(message: &str) -> Vec<Vec<u8>> {
    let mut tokens = Vec::new();
    let mut cur: Vec<u8> = Vec::new();
    for &b in message.as_bytes() {
        let lb = b.to_ascii_lowercase();
        let is_alnum = lb.is_ascii_digit() || (b'a'..=b'z').contains(&lb);
        if is_alnum {
            cur.push(lb);
        } else if !cur.is_empty() {
            tokens.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        tokens.push(cur);
    }
    tokens
}

pub fn normalize_text(input: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    let mut normalized = String::new();
    for ch in input.nfkd() {
        match ch {
            '\u{200b}'..='\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2060}'..='\u{206f}' | '\u{feff}' => {
                // strip / skip
            }
            // Lowercase Cyrillic homoglyphs
            'а' => normalized.push('a'),
            'с' => normalized.push('c'),
            'е' => normalized.push('e'),
            'о' => normalized.push('o'),
            'р' => normalized.push('p'),
            'х' => normalized.push('x'),
            'у' => normalized.push('y'),
            'і' => normalized.push('i'),
            'ј' => normalized.push('j'),
            'ѕ' => normalized.push('s'),
            // Uppercase Cyrillic homoglyphs
            'А' => normalized.push('A'),
            'С' => normalized.push('C'),
            'Е' => normalized.push('E'),
            'О' => normalized.push('O'),
            'Р' => normalized.push('P'),
            'Х' => normalized.push('X'),
            'У' => normalized.push('Y'),
            'І' => normalized.push('I'),
            'Ј' => normalized.push('J'),
            'Ѕ' => normalized.push('S'),
            other => normalized.push(other),
        }
    }
    normalized
}

/// Compute the length-`d` signed integer feature vector `phi(m)`.
pub fn feature_vector(message: &str, d: usize) -> Vec<i64> {
    let normalized = normalize_text(message);
    let mut phi = vec![0i64; d];
    for tok in tokenize(&normalized) {
        let h_idx = fnv1a_64(&tok);
        let mut sign_input = Vec::with_capacity(tok.len() + 1);
        sign_input.push(0x01u8);
        sign_input.extend_from_slice(&tok);
        let h_sign = fnv1a_64(&sign_input);
        let idx = (h_idx % d as u64) as usize;
        let sign: i64 = if h_sign & 1 == 0 { 1 } else { -1 };
        phi[idx] += sign;
    }
    phi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a_known_values() {
        // FNV-1a("") == offset basis; FNV-1a("a") == 0xaf63dc4c8601ec8c
        assert_eq!(fnv1a_64(b""), FNV_OFFSET_BASIS);
        assert_eq!(fnv1a_64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a_64(b"foobar"), 0x85944171f73967e8);
    }

    #[test]
    fn tokenizer_basic() {
        assert_eq!(
            tokenize("Free entry, NOW!!"),
            vec![b"free".to_vec(), b"entry".to_vec(), b"now".to_vec()]
        );
    }

    #[test]
    fn test_normalize_text() {
        // Zero-width / invisible formatting characters stripping
        assert_eq!(normalize_text("Fr\u{200b}ee\u{feff} e\u{200d}ntry"), "Free entry");

        // Cyrillic homoglyph folding
        assert_eq!(normalize_text("асеорхуі"), "aceopxyi");
        assert_eq!(normalize_text("АСЕОРХУІ"), "ACEOPXYI");

        // Styled mathematical characters (decomposed via NFKD)
        assert_eq!(normalize_text("𝖥𝗋𝖾𝖾 𝖾𝗇𝗍𝗋𝗒"), "Free entry");
    }
}
