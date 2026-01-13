//! UGM (Unigram) tokenizer implementation.
//!
//! Port of llama.cpp UGM tokenizer session (unigram / SentencePiece-style Viterbi).
//! Source: llm_tokenizer_ugm_session::tokenize
//!
//! Key features:
//! - Trie-based prefix matching
//! - Viterbi-style DP for optimal tokenization
//! - Score-based selection between competing tokenizations
//! - Unknown token handling with penalty score

use crate::vocab::{TokenType, Vocabulary};
use crate::Error;
use std::collections::HashMap;

/// Trie node for byte-level prefix matching.
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

    fn insert(&mut self, s: &str, id: Option<u32>) {
        let mut cur = 0usize;
        for &b in s.as_bytes() {
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
        if let Some(id) = id {
            self.nodes[cur].value = Some(id);
        }
    }

    fn traverse(&self, node: usize, b: u8) -> Option<usize> {
        self.nodes[node].next.get(&b).copied()
    }

    fn value(&self, node: usize) -> Option<u32> {
        self.nodes[node].value
    }
}

/// Fragment types during user-defined token preprocessing.
enum UgmFragment {
    /// A user-defined token that was matched
    UserDefined(u32),
    /// Regular text to be tokenized via Viterbi
    Text(String),
}

/// UGM tokenizer using Viterbi-style DP.
pub struct UgmTokenizer {
    trie: NaiveTrie,
    user_defined_trie: NaiveTrie,
    unknown_token_score: f64,
}

impl UgmTokenizer {
    /// Create a new UGM tokenizer from a vocabulary.
    pub fn new(vocab: &Vocabulary) -> Self {
        let mut trie = NaiveTrie::new();
        let mut user_defined_trie = NaiveTrie::new();

        let mut min_score = f64::INFINITY;

        for id in 0..(vocab.n_tokens() as u32) {
            let txt = match vocab.get_token_text(id) {
                Some(t) => t,
                None => continue,
            };
            let ttype = vocab.get_token_type(id);

            // Insert normal, user_defined, unused tokens into trie
            if matches!(
                ttype,
                TokenType::Normal | TokenType::UserDefined | TokenType::Unused
            ) {
                trie.insert(txt, Some(id));
            }
            if matches!(ttype, TokenType::UserDefined) {
                user_defined_trie.insert(txt, Some(id));
            }

            if matches!(ttype, TokenType::Normal) {
                let sc = vocab.get_token_score(id) as f64;
                min_score = min_score.min(sc);
            }
        }

        // Unknown token score = min_score - penalty (default 10.0)
        let unknown_token_score_penalty = 10.0f64;
        let unknown_token_score = if min_score.is_finite() {
            min_score - unknown_token_score_penalty
        } else {
            -10.0 // Fallback if no normal tokens
        };

        Self {
            trie,
            user_defined_trie,
            unknown_token_score,
        }
    }

    /// Encode text into token IDs using Viterbi DP.
    pub fn encode(&self, text: &str, vocab: &Vocabulary) -> Result<Vec<u32>, Error> {
        let normalized = normalize_ugm(text);
        if normalized.is_empty() {
            return Ok(Vec::new());
        }

        // Preprocess: split on user-defined tokens first (llama.cpp parity)
        // User-defined tokens like <|endoftext|> must be matched greedily before Viterbi
        let fragments = self.split_on_user_defined(&normalized);

        let mut result = Vec::new();
        for fragment in fragments {
            match fragment {
                UgmFragment::UserDefined(token_id) => {
                    result.push(token_id);
                }
                UgmFragment::Text(segment) => {
                    let tokens = self.encode_segment(&segment, vocab)?;
                    result.extend(tokens);
                }
            }
        }

        Ok(result)
    }

    /// Split text on user-defined tokens using greedy longest match.
    fn split_on_user_defined(&self, text: &str) -> Vec<UgmFragment> {
        let bytes = text.as_bytes();
        let n = bytes.len();
        let mut fragments = Vec::new();
        let mut pos = 0;
        let mut text_start = 0;

        while pos < n {
            // Try to match a user-defined token at this position
            let mut best_len = 0;
            let mut best_id = None;

            if let Some(mut node) = self.user_defined_trie.traverse(0, bytes[pos]) {
                let mut len = 1;
                if let Some(id) = self.user_defined_trie.value(node) {
                    best_len = len;
                    best_id = Some(id);
                }

                while pos + len < n {
                    match self.user_defined_trie.traverse(node, bytes[pos + len]) {
                        Some(next) => {
                            node = next;
                            len += 1;
                            if let Some(id) = self.user_defined_trie.value(node) {
                                best_len = len;
                                best_id = Some(id);
                            }
                        }
                        None => break,
                    }
                }
            }

            if let Some(token_id) = best_id {
                // Emit any text before this user-defined token
                if pos > text_start {
                    fragments.push(UgmFragment::Text(text[text_start..pos].to_string()));
                }
                fragments.push(UgmFragment::UserDefined(token_id));
                pos += best_len;
                text_start = pos;
            } else {
                pos += 1;
            }
        }

        // Emit remaining text
        if text_start < n {
            fragments.push(UgmFragment::Text(text[text_start..].to_string()));
        }

        fragments
    }

    /// Encode a text segment (no user-defined tokens) using Viterbi DP.
    fn encode_segment(&self, text: &str, vocab: &Vocabulary) -> Result<Vec<u32>, Error> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        // Best tokenization DP table
        #[derive(Clone, Copy)]
        struct Best {
            token: u32,
            start: usize,
            score: f64,
        }

        let n = text.len();
        let unk_id = vocab.unk_token_id();

        let mut best: Vec<Best> = vec![
            Best {
                token: unk_id,
                start: 0,
                score: f64::NEG_INFINITY
            };
            n + 1
        ];
        best[0] = Best {
            token: unk_id,
            start: 0,
            score: 0.0,
        };

        let bytes = text.as_bytes();
        let mut input_offset = 0usize;

        while input_offset < n {
            // Size of current UTF-8 codepoint in bytes
            let cp_len = utf8_cp_len(bytes[input_offset]).min(n - input_offset);

            let current_best = best[input_offset];

            // Traverse trie from this position
            let mut prefix_offset = input_offset;
            let mut node_opt = self.trie.traverse(0, bytes[prefix_offset]);
            prefix_offset += 1;

            let mut single_codepoint_token_found = false;

            while prefix_offset <= n {
                let node = match node_opt {
                    Some(x) => x,
                    None => break,
                };

                if let Some(token_id) = self.trie.value(node) {
                    if prefix_offset - input_offset == cp_len {
                        single_codepoint_token_found = true;
                    }

                    let token_score = match vocab.get_token_type(token_id) {
                        TokenType::UserDefined => 0.0, // User-defined tokens get score 0
                        _ => vocab.get_token_score(token_id) as f64,
                    };

                    let challenger = current_best.score + token_score;
                    if challenger > best[prefix_offset].score {
                        best[prefix_offset] = Best {
                            token: token_id,
                            start: input_offset,
                            score: challenger,
                        };
                    }
                }

                if prefix_offset == n {
                    break;
                }
                node_opt = self.trie.traverse(node, bytes[prefix_offset]);
                prefix_offset += 1;
            }

            // Unknown-token path if no token covers whole codepoint
            if !single_codepoint_token_found {
                let next = input_offset + cp_len;
                let challenger = current_best.score + self.unknown_token_score;
                if challenger > best[next].score {
                    best[next] = Best {
                        token: unk_id,
                        start: input_offset,
                        score: challenger,
                    };
                }
            }

            input_offset += cp_len;
        }

        // Backtrack from end to build tokens
        let mut out_rev: Vec<u32> = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let b = best[pos];
            out_rev.push(b.token);
            pos = b.start;
        }
        out_rev.reverse();
        Ok(out_rev)
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

/// Get the length of a UTF-8 codepoint from its first byte.
fn utf8_cp_len(first: u8) -> usize {
    match first {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    }
}

/// Placeholder normalization.
///
/// For exact llama.cpp parity, route this through GGUF charsmap/XCDA parsing.
fn normalize_ugm(text: &str) -> String {
    text.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_utf8_cp_len() {
        assert_eq!(utf8_cp_len(b'a'), 1);
        assert_eq!(utf8_cp_len(0xC2), 2); // Start of 2-byte sequence
        assert_eq!(utf8_cp_len(0xE0), 3); // Start of 3-byte sequence
        assert_eq!(utf8_cp_len(0xF0), 4); // Start of 4-byte sequence
    }

    #[test]
    fn test_trie_insert_and_traverse() {
        let mut trie = NaiveTrie::new();
        trie.insert("hello", Some(42));

        let node1 = trie.traverse(0, b'h').unwrap();
        let node2 = trie.traverse(node1, b'e').unwrap();
        let node3 = trie.traverse(node2, b'l').unwrap();
        let node4 = trie.traverse(node3, b'l').unwrap();
        let node5 = trie.traverse(node4, b'o').unwrap();

        assert_eq!(trie.value(node5), Some(42));
    }
}
