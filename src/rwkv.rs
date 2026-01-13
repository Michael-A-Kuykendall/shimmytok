//! RWKV tokenizer implementation.
//!
//! Port of llama.cpp RWKV tokenizer.
//! Source: llama_unescape_rwkv_token + llm_tokenizer_rwkv_session::tokenize
//!
//! Key features:
//! - Trie-based greedy longest match
//! - Special escape sequences: \n, \t, \r, \xNN

use crate::vocab::Vocabulary;
use crate::Error;
use std::collections::HashMap;

/// Trie node for byte-level matching.
#[derive(Clone, Default)]
struct TrieNode {
    next: HashMap<u8, usize>,
    value: Option<u32>,
}

/// Byte-level trie for token matching.
#[derive(Clone)]
struct NaiveTrie {
    nodes: Vec<TrieNode>,
}

impl NaiveTrie {
    fn new() -> Self {
        Self {
            nodes: vec![TrieNode::default()],
        }
    }

    fn insert_bytes(&mut self, bs: &[u8], id: u32) {
        let mut cur = 0usize;
        for &b in bs {
            let existing = self.nodes[cur].next.get(&b).copied();
            let nxt = match existing {
                Some(n) => n,
                None => {
                    let new_idx = self.nodes.len();
                    self.nodes.push(TrieNode::default());
                    self.nodes[cur].next.insert(b, new_idx);
                    new_idx
                }
            };
            cur = nxt;
        }
        self.nodes[cur].value = Some(id);
    }
}

/// RWKV tokenizer using trie-based greedy matching.
pub struct RwkvTokenizer {
    trie: NaiveTrie,
}

impl RwkvTokenizer {
    /// Create a new RWKV tokenizer from a vocabulary.
    pub fn new(vocab: &Vocabulary) -> Self {
        let mut trie = NaiveTrie::new();
        for id in 0..(vocab.n_tokens() as u32) {
            if let Some(txt) = vocab.get_token_text(id) {
                let raw = unescape_rwkv_token(txt);
                trie.insert_bytes(&raw, id);
            }
        }
        Self { trie }
    }

    /// Encode text into token IDs using trie-based greedy matching.
    pub fn encode(&self, text: &str, vocab: &Vocabulary) -> Result<Vec<u32>, Error> {
        let bs = text.as_bytes();
        let mut out: Vec<u32> = Vec::new();
        let mut pos = 0usize;

        while pos < bs.len() {
            // First byte must exist in trie
            let mut node = match self.trie.nodes[0].next.get(&bs[pos]).copied() {
                Some(n) => n,
                None => {
                    // Unknown byte - emit unk token
                    out.push(vocab.unk_token_id());
                    pos += 1;
                    continue;
                }
            };

            let mut best_id: u32 = 0;
            let mut best_end: usize = pos;

            // Check if current node has a value
            if let Some(v) = self.trie.nodes[node].value {
                best_id = v;
                best_end = pos + 1;
            }

            let mut i = pos + 1;
            while i < bs.len() {
                match self.trie.nodes[node].next.get(&bs[i]).copied() {
                    Some(n) => {
                        node = n;
                        i += 1;
                        if let Some(v) = self.trie.nodes[node].value {
                            best_id = v;
                            best_end = i;
                        }
                    }
                    None => break,
                }
            }

            // Check final node after loop
            if let Some(v) = self.trie.nodes[node].value {
                if i > best_end {
                    best_id = v;
                    best_end = i;
                }
            }

            if best_end > pos {
                out.push(best_id);
                pos = best_end;
            } else {
                // No match found, emit unk and advance
                out.push(vocab.unk_token_id());
                pos += 1;
            }
        }

        Ok(out)
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, tokens: &[u32], vocab: &Vocabulary) -> Result<String, Error> {
        let mut s = String::new();
        for &t in tokens {
            if let Some(txt) = vocab.get_token_text(t) {
                s.push_str(txt);
            }
        }
        Ok(s)
    }
}

/// Unescape RWKV token escape sequences.
///
/// Handles: \n, \t, \r, \xNN (hex byte)
pub fn unescape_rwkv_token(escaped: &str) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(escaped.len());
    let mut escaping = false;
    let mut hex_remaining: u8 = 0;
    let mut hex_acc: u8 = 0;

    for &b in escaped.as_bytes() {
        if hex_remaining != 0 {
            let v = match b {
                b'a'..=b'f' => b - b'a' + 10,
                b'A'..=b'F' => b - b'A' + 10,
                b'0'..=b'9' => b - b'0',
                _ => 0, // Invalid hex digit
            };
            hex_acc = (hex_acc << 4) + v;
            hex_remaining -= 1;
            if hex_remaining == 0 {
                out.push(hex_acc);
                hex_acc = 0;
            }
            continue;
        }

        if escaping {
            match b {
                b't' => out.push(b'\t'),
                b'n' => out.push(b'\n'),
                b'r' => out.push(b'\r'),
                b'x' => hex_remaining = 2,
                _ => out.push(b),
            }
            escaping = false;
            continue;
        }

        if b == b'\\' {
            escaping = true;
            continue;
        }

        out.push(b);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unescape_newline() {
        assert_eq!(unescape_rwkv_token("\\n"), vec![b'\n']);
    }

    #[test]
    fn test_unescape_tab() {
        assert_eq!(unescape_rwkv_token("\\t"), vec![b'\t']);
    }

    #[test]
    fn test_unescape_carriage_return() {
        assert_eq!(unescape_rwkv_token("\\r"), vec![b'\r']);
    }

    #[test]
    fn test_unescape_hex() {
        assert_eq!(unescape_rwkv_token("\\x41"), vec![b'A']);
        assert_eq!(unescape_rwkv_token("\\x00"), vec![0u8]);
        assert_eq!(unescape_rwkv_token("\\xff"), vec![0xFF]);
    }

    #[test]
    fn test_unescape_plain() {
        assert_eq!(unescape_rwkv_token("abc"), b"abc".to_vec());
    }

    #[test]
    fn test_unescape_mixed() {
        assert_eq!(
            unescape_rwkv_token("hello\\nworld"),
            b"hello\nworld".to_vec()
        );
    }
}
