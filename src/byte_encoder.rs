//! GPT-2 byte-level encoding
//! Direct port of `OpenAI`'s `bytes_to_unicode()` function

use std::collections::HashMap;
use std::sync::OnceLock;

/// Get the GPT-2 byte-to-unicode mapping
/// This maps bytes to unicode characters for byte-level BPE
pub fn bytes_to_unicode() -> &'static HashMap<u8, char> {
    static BYTE_ENCODER: OnceLock<HashMap<u8, char>> = OnceLock::new();
    BYTE_ENCODER.get_or_init(|| {
        // Start with printable ASCII and some extended characters
        let mut bs: Vec<u32> = Vec::new();

        // !"#$%&'()*+,-./0123456789:;<=>?@ABC...XYZ[\]^_`abc...xyz{|}~
        bs.extend(u32::from(b'!')..=u32::from(b'~'));

        // ¡¢£¤¥¦§¨©ª«
        bs.extend(0xA1..=0xAC);

        // ®¯°±²³´µ¶·¸¹º»¼½¾¿À...ÿ
        bs.extend(0xAE..=0xFF);

        let mut cs = bs.clone();
        let mut n = 0;

        // For bytes not in the printable set, map to high unicode (256+)
        for b in 0u32..256u32 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }

        // Create the mapping
        let mut result = HashMap::new();
        for (byte_val, &unicode_val) in bs.iter().zip(cs.iter()) {
            result.insert(*byte_val as u8, char::from_u32(unicode_val).unwrap());
        }

        result
    })
}

/// Get the reverse mapping (unicode char -> byte)
pub fn unicode_to_bytes() -> &'static HashMap<char, u8> {
    static BYTE_DECODER: OnceLock<HashMap<char, u8>> = OnceLock::new();
    BYTE_DECODER.get_or_init(|| bytes_to_unicode().iter().map(|(&k, &v)| (v, k)).collect())
}

/// Encode text bytes to GPT-2 unicode representation
#[must_use] 
pub fn encode_bytes(text: &str) -> String {
    let byte_encoder = bytes_to_unicode();
    text.bytes()
        .map(|b| byte_encoder.get(&b).copied().unwrap_or('�'))
        .collect()
}

/// Decode GPT-2 byte-encoded text back to a regular string.
/// Note: For very large inputs (>100MB), consider streaming to avoid memory spikes.
#[must_use] 
pub fn decode_bytes(text: &str) -> String {
    let byte_decoder = unicode_to_bytes();
    let bytes: Vec<u8> = text
        .chars()
        .filter_map(|c| byte_decoder.get(&c).copied())
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_space_encoding() {
        let encoded = encode_bytes(" ");
        // Space (byte 32/0x20) should map to Ġ (U+0120)
        assert_eq!(encoded.chars().next().unwrap() as u32, 0x0120);
    }

    #[test]
    fn test_hello() {
        let encoded = encode_bytes("Hello");
        println!("'Hello' encodes to: {encoded:?}");
        for (i, c) in encoded.chars().enumerate() {
            println!("  [{}] '{}' = U+{:04X}", i, c, c as u32);
        }
    }
}
