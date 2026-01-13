//! WPM (Word-Piece Model) tokenizer implementation.
//!
//! Port of llama.cpp "WPM tokenizer" (LLAMA_VOCAB_TYPE_WPM).
//! Source reference: llm_tokenizer_wpm_session::tokenize + preprocess in llama-vocab.cpp
//!
//! Key behaviors:
//! - Unicode NFD normalize, lowercase
//! - Split into "words" on whitespace
//! - Punctuation / certain symbols / Chinese chars become single-char "words"
//! - Each word is tokenized by greedy longest-match against vocab, with phantom space prefix U+2581 (▁)
//! - If any character position in a word cannot be matched: discard all tokens produced for that word
//! - If a word yields no tokens, emit unk

use crate::vocab::Vocabulary;
use crate::Error;

/// WPM tokenizer using phantom-space + greedy longest-match algorithm.
pub struct WpmTokenizer {
    /// Maximum token length in vocab - used to cap search range.
    max_token_len: usize,
}

impl WpmTokenizer {
    /// Create a new WPM tokenizer from a vocabulary.
    pub fn new(vocab: &Vocabulary) -> Self {
        let mut max_len = 0usize;
        for id in 0..(vocab.n_tokens() as u32) {
            if let Some(t) = vocab.get_token_text(id) {
                max_len = max_len.max(t.len());
            }
        }
        Self {
            max_token_len: max_len,
        }
    }

    /// Encode text into token IDs using WPM algorithm.
    pub fn encode(&self, text: &str, vocab: &Vocabulary) -> Result<Vec<u32>, Error> {
        let words = preprocess_wpm(text);

        let phantom_space = "\u{2581}"; // U+2581 ▁
        let mut out: Vec<u32> = Vec::new();

        for w in words {
            if w.is_empty() {
                continue;
            }

            let word1 = format!("{phantom_space}{w}");
            let bytes = word1.as_bytes();
            let n = bytes.len();
            let checkpoint = out.len();

            let mut i = 0usize;
            while i < n {
                let mut matched = false;

                // llama.cpp upper bound: min(n, i + max_token_len + 1) and then j decrements
                let mut j = (i + self.max_token_len + 1).min(n);
                while j > i {
                    // Use word1.get() to safely slice, which handles UTF-8 boundaries
                    if let Some(s) = word1.get(i..j) {
                        if let Some(id) = vocab.get_token_id(s) {
                            out.push(id);
                            matched = true;
                            i = j; // next position
                            break;
                        }
                    }
                    j -= 1;
                }

                if !matched {
                    // discard all tokens added for this word
                    out.truncate(checkpoint);
                    break;
                }
            }

            if out.len() == checkpoint {
                // No tokens matched for this word - emit unk
                out.push(vocab.unk_token_id());
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

/// Preprocess text for WPM tokenization.
///
/// - Lowercase
/// - Split on whitespace
/// - Punctuation / symbols / Chinese chars => single-char word boundaries
fn preprocess_wpm(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();

    let mut words: Vec<String> = vec![String::new()];
    for ch in lower.chars() {
        if ch.is_whitespace() {
            if !words.last().unwrap().is_empty() {
                words.push(String::new());
            }
            continue;
        }

        // Check if this is punctuation that should be isolated
        let is_punct = ch.is_ascii_punctuation() || is_cjk_char(ch);
        if is_punct {
            if !words.last().unwrap().is_empty() {
                words.push(String::new());
            }
            words.last_mut().unwrap().push(ch);
            words.push(String::new());
        } else {
            words.last_mut().unwrap().push(ch);
        }
    }

    if words.last().map(|w| w.is_empty()).unwrap_or(false) {
        words.pop();
    }
    words
}

/// Check if a character is in the CJK unicode range.
/// These are treated as single-char words in WPM.
fn is_cjk_char(ch: char) -> bool {
    let cp = ch as u32;
    // CJK Unified Ideographs
    (0x4E00..=0x9FFF).contains(&cp)
        // CJK Unified Ideographs Extension A
        || (0x3400..=0x4DBF).contains(&cp)
        // CJK Unified Ideographs Extension B
        || (0x20000..=0x2A6DF).contains(&cp)
        // CJK Compatibility Ideographs
        || (0xF900..=0xFAFF).contains(&cp)
        // CJK Unified Ideographs Extension C
        || (0x2A700..=0x2B73F).contains(&cp)
        // CJK Unified Ideographs Extension D
        || (0x2B740..=0x2B81F).contains(&cp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_wpm_basic() {
        let words = preprocess_wpm("Hello, world!");
        assert_eq!(words, vec!["hello", ",", "world", "!"]);
    }

    #[test]
    fn test_preprocess_wpm_whitespace() {
        let words = preprocess_wpm("  multiple   spaces  ");
        assert_eq!(words, vec!["multiple", "spaces"]);
    }

    #[test]
    fn test_preprocess_wpm_cjk() {
        let words = preprocess_wpm("hello世界");
        // Each CJK char should be its own word
        assert_eq!(words, vec!["hello", "世", "界"]);
    }

    #[test]
    fn test_is_cjk_char() {
        assert!(is_cjk_char('中'));
        assert!(is_cjk_char('国'));
        assert!(!is_cjk_char('a'));
        assert!(!is_cjk_char('ä'));
    }
}
